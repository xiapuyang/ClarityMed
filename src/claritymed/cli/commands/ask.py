"""``claritymed ask`` — stream a grounded answer to one question.

One-shot mirror of the TUI's ask path. The CLI never instantiates an
``Agent`` directly; all the orchestration goes through ``AskService``
so any service-level change (audit, scrub, event schema) reaches the
TUI and the CLI together.
"""

from __future__ import annotations

import logging
import typer

from claritymed.cli.common import (
    console,
    run_async,
    try_current_account,
)
from claritymed.cli.entry import inject_context
from claritymed.core.llm.model import build_model
from claritymed.orchestrator.services import (
    AskService,
    Done,
    Error,
    TokenChunk,
)
from claritymed.stores.models import resolve_provider

logger = logging.getLogger(__name__)


def _maybe_build_strategy(model=None):
    """Return a RagStrategy when ``rag.enabled=true``, else ``None``.

    Lazy-imports the retrieval factory so a disabled-RAG ``ask`` does not
    pay the import cost of qdrant / llama-index. Fail-loud once enabled —
    a missing dependency raises rather than silently degrading to LLM-only.

    ``model`` is forwarded to strategies that issue LLM calls during
    retrieval (HyDE). Strategies that don't need it (naive_hybrid,
    agentic) ignore the argument.
    """
    from claritymed.core.rag import load_retrieval_config

    cfg = load_retrieval_config()
    if not cfg.rag.enabled:
        return None
    from claritymed.core.rag import build_hybrid_retriever
    from claritymed.core.rag.strategies import build_strategy

    retriever = build_hybrid_retriever(cfg)
    return build_strategy(
        retriever,
        config=cfg.strategies,
        max_evidence=cfg.rag.max_evidence,
        model=model,
    )


def ask(
    question: str = typer.Argument(..., help="The medical question to ask."),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    provider_id: str | None = typer.Option(None, "--provider", "-p"),
) -> None:
    """Stream a grounded answer to ``question``."""

    async def _run() -> None:
        from claritymed.orchestrator.services import ChatSession

        with inject_context(
            user_id=user,
            language=language,
            command=f"ask q={question[:60]!r}",
            check_user_exists=True,
        ) as (_, uid, lang):
            from claritymed.errors import CloudOptInRequiredError

            account = try_current_account()
            try:
                provider = resolve_provider(override=provider_id, account=account)
            except CloudOptInRequiredError as exc:
                logger.error("%s", exc)
                raise typer.Exit(code=2) from exc
            model = build_model(provider)
            from claritymed.core.translation import make_translation_provider

            strategy = _maybe_build_strategy(model=model)
            from claritymed.core.rag import load_retrieval_config

            mode_name = load_retrieval_config().rag.mode
            service = AskService(
                model=model,
                language=lang,
                chat_session=ChatSession.new(uid),
                provider_id=provider.id,
                model_name=provider.model,
                strategy=strategy,
                provider_config=provider,
                translation_service=make_translation_provider(
                    model, phi_kind=provider.kind
                ),
                rag_mode=mode_name,
            )

            async for event in service.run(question, user_id=uid):
                if isinstance(event, TokenChunk):
                    console.print(event.text, end="")
                elif isinstance(event, Error):
                    logger.error("%s: %s", event.error_type, event.message)
                    raise typer.Exit(code=1)
                elif isinstance(event, Done):
                    console.print()  # newline after streaming text

    run_async(_run())
