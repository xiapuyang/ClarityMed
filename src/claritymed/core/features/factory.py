"""Build the list of active ``FeaturePlugin`` instances for an ask turn.

Adding a new feature is one entry here + one plugin class. The factory
fails loud on any active feature picking ``agentic`` mode in v1 — the
state-graph workflow is not implemented yet, and silently downgrading
to ``tool`` would mask the rollout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from claritymed.core.features.base import FeaturePlugin

if TYPE_CHECKING:
    from claritymed.core.rag.strategies.base import RagStrategy


def build_features(
    *,
    rag_mode: str = "tool",
    rag_strategy: "RagStrategy | None" = None,
    get_session_id: Callable[[], str | None] | None = None,
    ingest_factory: Callable[[], FeaturePlugin] | None = None,
    symptoms_factory: Callable[[], FeaturePlugin] | None = None,
    vision_factory: Callable[[], FeaturePlugin] | None = None,
    profile_context_factory: Callable[[], FeaturePlugin] | None = None,
) -> list[FeaturePlugin]:
    """Instantiate plugins from per-feature config.

    Args:
        rag_mode: ``rag.mode`` from retrieval.yaml.
        rag_strategy: Pre-built RAG strategy (``None`` when
            ``rag.enabled=false``); the RagFeature is still added so the
            tool can emit ``RAG disabled`` events for visibility.
        get_session_id: When set, enables ``AttachmentsFeature`` so OCR'd
            paste / upload text is splice-injected into the prompt. ``None``
            (the default) skips the plugin entirely — headless one-shot
            paths (CLI ``ask``, evals) have no session_id and would only
            get a no-op block.
        ingest_factory: When set, called once to build the ingest-tools
            plugin (the seven write tools behind an approval gate). The
            factory lives outside ``core/`` because the plugin depends on
            ``orchestrator.features.ingest_tools_plugin`` — passing a
            callable here keeps ``core.features`` import-free of
            orchestrator code while letting AskService wire the live
            dispatcher + settings store.
        symptoms_factory: When set, called once to build the symptoms
            plugin (``predict_disease_from_symptoms`` tool). Same
            indirection rationale as ``ingest_factory``: the plugin
            depends on ``orchestrator.features.symptoms_plugin``, so we
            pass a callable instead of importing the concrete class
            here. ``None`` (the default) means the symptoms feature is
            absent — operator either flagged
            ``configs/symptoms.yaml.datasets[0].enabled = false`` or
            never created the file.
        vision_factory: When set, called once to build the vision
            plugin (``detect_disease_from_image`` tool). Same
            indirection rationale as ``ingest_factory`` /
            ``symptoms_factory``: the plugin depends on
            ``orchestrator.features.vision_plugin``. ``None`` (the
            default) means the vision feature is absent — operator
            either flagged every disease ``enabled: false`` or the
            registry bootstrap failed.
        profile_context_factory: When set, called once to build the
            profile-context plugin. The factory lives in the caller
            (orchestrator layer) so ``core.features`` stays import-free
            of orchestrator code. ``None`` disables the plugin.

    Raises:
        NotImplementedError: any active feature requests ``agentic`` mode.
    """
    from claritymed.core.rag.feature import RagFeature

    plugins: list[FeaturePlugin] = [
        RagFeature(mode=rag_mode, strategy=rag_strategy),  # type: ignore[arg-type]
    ]
    if get_session_id is not None:
        from claritymed.core.attachments_feature import AttachmentsFeature

        plugins.append(AttachmentsFeature(get_session_id=get_session_id))
    if ingest_factory is not None:
        plugins.append(ingest_factory())
    if symptoms_factory is not None:
        plugins.append(symptoms_factory())
    if vision_factory is not None:
        plugins.append(vision_factory())
    if profile_context_factory is not None:
        plugins.append(profile_context_factory())

    agentic = [p for p in plugins if p.mode == "agentic"]
    if agentic:
        names = ", ".join(p.name for p in agentic)
        raise NotImplementedError(
            f"rag.mode=agentic (and any future <feature>.mode=agentic) is "
            f"reserved for the state-graph workflow and has no runtime yet. "
            f"Active agentic features: {names}. Set mode to 'tool' or "
            f"'deterministic'."
        )
    return plugins
