"""Client-side routing for the vision tool — KTD-V2 boot-time catalog.

Built at orchestrator startup from ``configs/vision.yaml``. Talks to
every configured server once (``GET /v1/catalog``), cross-checks each
returned model's ``manifest_sha`` against the committed config, and
caches the routing decision for the process lifetime.

KTD-V2 says it loud and clear: **no periodic refresh, no
force_refresh, no expiration**. Multi-server routing and hot reload
are deferred to v1.x when a second server actually lands. Operators
who swap a model file in place are expected to restart the
orchestrator — same posture as symptoms.

Routing algorithm (matches plan §"Approach"):

1. Look up ``disease_id`` in ``configs/vision.yaml::diseases``. Miss →
   :class:`UnknownDiseaseError`.
2. If the caller passed a ``model_id_hint`` and it appears in
   ``disease.effective_flow`` (primary + fallbacks), prefer it.
3. Otherwise use ``disease.primary_model_id``.
4. Look up the model's ``server_id`` → :class:`ServerSpec`. The
   ``ModelSpec`` validator in Unit 1 already cross-checks that every
   ``server_id`` references a known server, so this step never raises
   in practice; we add a defensive ``RuntimeError`` for surprise.

Each ``ServerSpec`` carries its own :class:`VisionHttpClient` so the
tool body can ``await registry.client_for(server_spec).detect(...)``.
The registry owns the clients' lifecycles; ``aclose()`` shuts them all
down in lockstep at orchestrator teardown.
"""

from __future__ import annotations

import logging
from typing import Mapping

import httpx

from claritymed.core.vision.client import VisionHttpClient
from claritymed.core.vision.schemas import (
    DiseaseSpec,
    ModelSpec,
    ServerSpec,
    VisionConfig,
)
from claritymed.errors import (
    UnknownDiseaseError,
    VisionCatalogMismatchError,
    VisionServerUnreachableError,
)

logger = logging.getLogger(__name__)


class VisionRegistry:
    """Catalog snapshot + routing for the vision tool.

    Constructed once at orchestrator boot. Synchronous routing methods;
    network only fires inside :meth:`bootstrap` (one round-trip per
    server).
    """

    def __init__(
        self,
        config: VisionConfig,
        *,
        clients: Mapping[str, VisionHttpClient] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args:
        config: Validated ``VisionConfig`` from ``configs/vision.yaml``.
        clients: Pre-built clients keyed by ``ServerSpec.id``. Tests
            inject these alongside a ``MockTransport`` so the
            registry can be exercised without an open socket. When
            omitted, the registry builds one client per server using
            the optional ``transport`` (test-only knob).
        transport: Shared ``httpx`` transport applied when ``clients``
            is None. Production passes ``None`` so each client opens
            a fresh ``AsyncClient``.
        """
        self._config = config
        self._diseases: dict[str, DiseaseSpec] = {d.id: d for d in config.diseases}
        self._models: dict[str, ModelSpec] = {m.id: m for m in config.models}
        self._servers: dict[str, ServerSpec] = {s.id: s for s in config.servers}
        if clients is not None:
            self._clients = dict(clients)
        else:
            self._clients = {
                server.id: VisionHttpClient(server.base_url, transport=transport)
                for server in config.servers
            }
        self._bootstrapped = False

    # --- introspection --------------------------------------------------

    @property
    def diseases(self) -> Mapping[str, DiseaseSpec]:
        return self._diseases

    @property
    def servers(self) -> Mapping[str, ServerSpec]:
        return self._servers

    def enabled_disease_ids(self) -> list[str]:
        return sorted(d.id for d in self._diseases.values() if d.enabled)

    def client_for(self, server: ServerSpec) -> VisionHttpClient:
        """Return the cached client for ``server``.

        Looked up by ``server.id`` rather than by reference so callers
        passing in a copy (e.g. ``model_copy``) still find the right
        client.
        """
        client = self._clients.get(server.id)
        if client is None:
            raise RuntimeError(
                f"no client registered for server_id {server.id!r}; "
                f"known: {sorted(self._clients)!r}"
            )
        return client

    # --- boot-time cross-check ------------------------------------------

    async def bootstrap(self) -> None:
        """Fetch ``/v1/catalog`` from every server and cross-check vs the config.

        Called once at orchestrator boot. Three outcomes per cross-check:

        * **Soft (warn-only):** server advertises a model the served set
          doesn't include, OR the served set has a model the server isn't
          loading. The all-or-nothing strict mode used to abort the whole
          feature on either; in practice a partially-deployed catalog
          is the common case during incremental rollout, so we degrade
          instead. When the missing model is a disease's **primary**, the
          disease is dropped from the in-memory catalog for this session
          (its tool-description entry vanishes, ``route`` raises
          ``UnknownDiseaseError`` if the LLM picks it from a stale
          prompt). When only a fallback is missing, runtime
          ``_run_fallback_flow`` already handles the unreachable hop.
        * **Hard (raises):** manifest_sha drift on a model both sides
          agree on — that is a data-integrity problem, not a deployment
          gap, and quietly serving mismatched weights is worse than
          taking the feature offline.

        Raises:
            VisionServerUnreachableError: A server didn't respond.
            VisionCatalogMismatchError: Manifest sha drift on a served
                model. (Missing-from-catalog / extra-on-server cases used
                to raise this too; they are now warnings.)
        """
        if self._bootstrapped:
            return
        missing_models: set[str] = set()
        for server in self._config.servers:
            client = self.client_for(server)
            try:
                catalog = await client.catalog()
            except VisionServerUnreachableError:
                logger.exception(
                    "vision boot cross-check: server %s unreachable", server.id
                )
                raise
            missing_models |= self._cross_check_server(server, catalog.models)
        if missing_models:
            affected = self._auto_disable_diseases_missing_primary(missing_models)
            if affected:
                logger.warning(
                    "vision: auto-disabled %d disease(s) for this session "
                    "because their primary model is not loaded on any "
                    "configured server: %s",
                    len(affected),
                    affected,
                )
        self._bootstrapped = True
        logger.info(
            "vision registry bootstrapped: diseases=%s servers=%s",
            sorted(self._diseases),
            sorted(self._servers),
        )

    def _cross_check_server(self, server: ServerSpec, catalog_models) -> set[str]:
        """Compare one server's catalog against the served set.

        Compares against the **served set** — models the server is
        expected to load right now: those listed in an enabled disease's
        ``flow``. Models present in ``configs/vision.yaml::models`` but
        outside any enabled flow (disabled-disease scaffolds, future
        fallback placeholders like ``breast_us_kaggle_resnet50_v1``) are
        intentionally ignored so the config can carry "ready to flip on"
        entries without breaking boot.

        Returns the set of served-set model_ids this server doesn't load.
        The caller aggregates these across servers and uses them to
        auto-disable affected diseases.

        Raises ``VisionCatalogMismatchError`` only on sha drift — the
        two-set-diff cases (extra on server / missing from server) are
        warnings now, see :meth:`bootstrap` for the rationale.
        """
        catalog_index = {m.model_id: m for m in catalog_models}
        served_set = {
            model_id
            for disease in self._diseases.values()
            if disease.enabled
            for model_id in disease.effective_flow
            if self._models.get(model_id)
            and self._models[model_id].server_id == server.id
        }
        extras = catalog_index.keys() - served_set
        if extras:
            # Server-side orphan — doesn't affect routing, just wastes
            # memory on the server. Log so a yaml ↔ deployment drift
            # surfaces, don't fail (the operator may be staging a new
            # disease whose yaml flip hasn't landed yet).
            logger.warning(
                "vision: server %r advertises %d model(s) outside the served set: %s",
                server.id,
                len(extras),
                sorted(extras),
            )
        missing = served_set - catalog_index.keys()
        if missing:
            # Caller decides per-disease whether to keep or drop based on
            # whether the missing model is a primary or just a fallback.
            logger.warning(
                "vision: server %r is missing %d served-set model(s): %s",
                server.id,
                len(missing),
                sorted(missing),
            )
        for model_id in catalog_index.keys() & served_set:
            spec = self._models[model_id]
            if catalog_index[model_id].manifest_sha != spec.manifest_sha256:
                raise VisionCatalogMismatchError(
                    f"manifest sha drift for model {model_id!r} on server "
                    f"{server.id!r}: config pins {spec.manifest_sha256}, "
                    f"server serves {catalog_index[model_id].manifest_sha}"
                )
        return missing

    def _auto_disable_diseases_missing_primary(
        self, missing_models: set[str]
    ) -> list[str]:
        """Drop enabled diseases whose primary model is missing from catalog.

        Only ``primary_model_id`` triggers auto-disable. A missing
        **fallback** is handled at request time inside
        :meth:`~claritymed.orchestrator.features.vision_plugin.VisionFeature._run_fallback_flow`
        — the HTTP call 404s, the flow steps to the next entry, the
        disease stays serviceable. Dropping the disease for a missing
        fallback would over-rotate: the primary works, the user gets
        no benefit from disabling the whole condition.

        Removing from ``self._diseases`` propagates everywhere downstream
        because both the tool description
        (``VisionFeature._build_tool_description``) and routing
        (``self.route``) read from this dict.
        """
        affected: list[str] = []
        for disease in list(self._diseases.values()):
            if not disease.enabled:
                continue
            if disease.primary_model_id in missing_models:
                self._diseases.pop(disease.id, None)
                affected.append(disease.id)
        affected.sort()
        return affected

    # --- routing --------------------------------------------------------

    def route(
        self, disease_id: str, model_id_hint: str | None = None
    ) -> tuple[ServerSpec, ModelSpec]:
        """Resolve a (disease, model_hint) pair to the (server, model) target.

        Raises:
            UnknownDiseaseError: ``disease_id`` is not in the catalog
                (or is disabled). Carries the list of enabled disease
                ids so the tool body can return structured guidance to
                the LLM.
        """
        disease = self._diseases.get(disease_id)
        if disease is None or not disease.enabled:
            raise UnknownDiseaseError(
                disease_id=disease_id,
                available=self.enabled_disease_ids(),
            )
        # Hint precedence: only honor when it appears in the effective
        # flow (primary + fallbacks). An out-of-flow hint is a soft
        # drop-through to primary so the LLM can pass speculative model
        # ids without breaking routing.
        if model_id_hint and model_id_hint in disease.effective_flow:
            target_model_id = model_id_hint
        else:
            target_model_id = disease.primary_model_id

        model = self._models.get(target_model_id)
        if model is None:
            # VisionConfig.cross_reference already catches this; raising
            # defensively in case a future config-mutation path forgets
            # to re-run validation.
            raise RuntimeError(
                f"model_id {target_model_id!r} resolved for disease "
                f"{disease_id!r} but is not in configs/vision.yaml::models"
            )
        server = self._servers.get(model.server_id)
        if server is None:
            raise RuntimeError(
                f"server_id {model.server_id!r} pinned by model "
                f"{model.id!r} not in configs/vision.yaml::servers"
            )
        return server, model

    # --- lifecycle ------------------------------------------------------

    async def aclose(self) -> None:
        """Close every client. Called at orchestrator teardown."""
        for client in self._clients.values():
            await client.aclose()


__all__ = ["VisionRegistry"]
