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

        Called once at orchestrator boot. Failures here are fatal — the
        orchestrator refuses to start so an operator notices the drift
        before any user request hits the vision tool.

        Raises:
            VisionServerUnreachableError: A server didn't respond.
            VisionCatalogMismatchError: A server's catalog disagrees
                with ``configs/vision.yaml``.
        """
        if self._bootstrapped:
            return
        for server in self._config.servers:
            client = self.client_for(server)
            try:
                catalog = await client.catalog()
            except VisionServerUnreachableError:
                logger.exception(
                    "vision boot cross-check: server %s unreachable", server.id
                )
                raise
            self._cross_check_server(server, catalog.models)
        self._bootstrapped = True
        logger.info(
            "vision registry bootstrapped: diseases=%s servers=%s",
            sorted(self._diseases),
            sorted(self._servers),
        )

    def _cross_check_server(self, server: ServerSpec, catalog_models) -> None:
        """For each catalog entry, find the matching ``ModelSpec`` and compare.

        Compares against the **served set** — models the server is
        expected to load right now: those listed in an enabled disease's
        ``flow``. Models present in ``configs/vision.yaml::models`` but
        outside any enabled flow (disabled-disease scaffolds, future
        fallback placeholders like ``breast_us_kaggle_resnet50_v1``) are
        intentionally ignored so the config can carry "ready to flip on"
        entries without breaking boot.

        Three failure modes:

        * Server advertises a model the served set doesn't include.
        * Served set has a model the server isn't loading.
        * Both sides agree on the model but ``manifest_sha`` differs.
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
        # Server-side surprise (served set doesn't include).
        for model_id in catalog_index.keys() - served_set:
            raise VisionCatalogMismatchError(
                f"server {server.id!r} advertises model {model_id!r} but "
                f"no enabled disease's flow lists it for this server"
            )
        # Config-side surprise (server doesn't load).
        for model_id in served_set - catalog_index.keys():
            raise VisionCatalogMismatchError(
                f"configs/vision.yaml expects model {model_id!r} on server "
                f"{server.id!r} (enabled disease's flow) but server's "
                f"/v1/catalog does not include it"
            )
        # Sha drift — only for models in the served set; orphan catalog
        # entries already failed above.
        for model_id in catalog_index.keys() & served_set:
            spec = self._models[model_id]
            if catalog_index[model_id].manifest_sha != spec.manifest_sha256:
                raise VisionCatalogMismatchError(
                    f"manifest sha drift for model {model_id!r} on server "
                    f"{server.id!r}: config pins {spec.manifest_sha256}, "
                    f"server serves {catalog_index[model_id].manifest_sha}"
                )

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
