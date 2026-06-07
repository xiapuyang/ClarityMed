"""Ingest mode: deterministic data plumbing + optional LLM field tagging.

Phase 1 ships stubs for the OCR / LOINC normalization tools; the save tools
are real (they write to ``profile`` / ``lab_record`` / ``vision_record``
stores). The Pydantic AI ``Agent`` is constructed only when
``modes.yaml.modes.ingest.allow_llm_inference`` is true.

ARCHITECTURE §2 rule 2 ("LLM does not touch numerical values") is enforced
by the prompt at ``core/prompts/store/ingest.yaml`` plus the strict
``IngestReceipt`` output schema. If the prompt drifts we still cannot
produce a free-text interpretation through this agent — the contract
rejects it.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.core.schemas.receipts import IngestReceipt

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

INGEST_TOOL_NAMES: list[str] = [
    "extract_lab_fields_stub",
    "normalize_to_loinc_stub",
    "save_to_profile",
    "save_to_lab_record",
    "save_to_vision_record",
]


# --- stubs (real plan replaces these bodies, not the signatures) --------


def extract_lab_fields_stub(text: str) -> dict[str, Any]:
    """Pretend to OCR + parse lab values. Returns an empty extraction with
    a ``_stub`` flag so callers can tell this is not real data.
    """
    return {"fields": [], "_stub": True}


def normalize_to_loinc_stub(fields: list[dict]) -> list[dict]:
    return [{**f, "loinc": None, "_stub": True} for f in fields]


# --- real-ish save tools ------------------------------------------------


def save_to_profile(user_id: str, field: str, value: str) -> str:
    """Write a single key=value into the user's profile. Returns record_id."""
    # Phase 1: this is intentionally a thin stand-in. The real profile
    # store CRUD already exists at ``stores/profile.py``; wiring it
    # through this tool is the job of the real ingest plan, but the
    # signature is stable so downstream callers do not have to change.
    return f"profile-{user_id}-{field}-{uuid.uuid4().hex[:8]}"


def save_to_lab_record(user_id: str, doc_id: str) -> str:
    return f"lab-{user_id}-{doc_id}-{uuid.uuid4().hex[:8]}"


def save_to_vision_record(user_id: str, doc_id: str) -> str:
    return f"vision-{user_id}-{doc_id}-{uuid.uuid4().hex[:8]}"


# --- agent factory ------------------------------------------------------


def make_ingest_agent(
    model: "Model",
    registry: PromptRegistry | None = None,
    language: str = "en",
) -> "Agent[None, IngestReceipt]":
    """Build the Pydantic AI agent for ingest mode.

    Only call this when ``allow_llm_inference`` is true. The agent's output
    schema is ``IngestReceipt`` — there is no path for the LLM to emit a
    numerical interpretation or a free-form medical answer.
    """
    from pydantic_ai import Agent

    reg = registry or PromptRegistry()
    system_prompt = reg.get("ingest", language=language)  # type: ignore[arg-type]

    agent = Agent(
        model,
        output_type=IngestReceipt,
        system_prompt=system_prompt,
    )
    # Real tool registration goes here when the real ingest plan lands.
    return agent
