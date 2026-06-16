"""Modality classifiers under test — Protocol + shipped backends.

Each backend implements :class:`ModalityClassifier` and exposes a
stable ``name`` used in the trials CSV. Adding a new classifier (e.g.
a ResNet head trained on labeled medical images) is a single new
class — :func:`build_classifiers` is the only place run.py needs to
learn about it.

Backends:

* :class:`MedicalClipClassifier` — production BiomedCLIP path via the
  medical-clip-server's ``/v1/classify_modality`` endpoint. Reused
  client; same HTTP call the OCR worker and vision plugin make at
  runtime, so the benchmark numbers are exactly what production sees.
* :class:`OmlxLlmClassifier` — multimodal LLM classifier via
  pydantic-ai. Sends the image (as ``BinaryContent``) with a tight
  system prompt that returns a single :class:`Modality` literal.
* :class:`ResnetModalityClassifier` — placeholder. Raises
  :class:`NotImplementedError` at construct time so the benchmark
  surfaces a clear "wire this up" message instead of silently dropping
  the row. Replace the body with a real adapter when a labeled
  modality-classification head ships.

PHI hygiene: every shipped backend is local (medical-clip-server on
``127.0.0.1`` + omlx local provider). Adding a cloud-backed classifier
would require an explicit opt-in flag at construct time — keep the
default surface local-only.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from claritymed.core.medical_clip.client import (
    MedicalClipClient,
)

logger = logging.getLogger(__name__)

# Default medical-clip-server URL. Matches the documented operator URL
# (configs/app.yaml + tests/e2e/_vision_helpers.py). Tests + scripts can
# override via the constructor.
DEFAULT_MEDICAL_CLIP_URL = "http://127.0.0.1:8086"

# Canonical modality vocabulary as a tuple — kept in this module rather
# than re-importing the Literal from core/medical_clip/schemas so the
# benchmark stays cheap to read (no transitive imports surprise the
# reader). The list MUST stay in sync with the Modality Literal — a
# CI check could verify but a one-liner inspection is fine while the
# vocabulary is small.
KNOWN_MODALITIES: tuple[str, ...] = (
    "ultrasound",
    "ct",
    "xray",
    "dermoscopy",
    "histopathology",
    "photo",
    "document",
    "unknown",
)


@runtime_checkable
class ModalityClassifier(Protocol):
    """Async classify-one-image surface.

    Implementations must:

    * Run entirely locally (no PHI over the network) unless the caller
      explicitly opts in.
    * Return one of :data:`KNOWN_MODALITIES`. Unrecognized labels MUST
      collapse to ``"unknown"`` rather than leaking through and
      polluting the confusion matrix.
    * Surface backing-service unreachability as an exception (the
      runner catches it and records the trial as errored). Silent
      degradation to ``"unknown"`` would let half a broken run look
      successful.
    """

    name: str

    async def classify(self, image_bytes: bytes, *, sha256: str) -> str: ...


# --- BiomedCLIP via medical-clip-server -------------------------------------


class MedicalClipClassifier:
    """Backend wrapping :class:`MedicalClipClient` for the benchmark."""

    name = "medical_clip"

    def __init__(self, *, base_url: str = DEFAULT_MEDICAL_CLIP_URL) -> None:
        self._client = MedicalClipClient(base_url=base_url)

    async def classify(self, image_bytes: bytes, *, sha256: str) -> str:
        response = await self._client.classify_modality(
            image_bytes,
            request_id=f"bench-{sha256[:8]}",
            sha256=sha256,
        )
        label = response.modality
        if label not in KNOWN_MODALITIES:
            logger.warning(
                "medical_clip returned unknown label %r — coercing to 'unknown'",
                label,
            )
            return "unknown"
        return label

    async def aclose(self) -> None:
        await self._client.aclose()


# --- omlx multimodal LLM -----------------------------------------------------


# Tight prompt — explicit allowlist, single-word answer. The structured
# output type (Modality literal) bakes in the constraint regardless of
# what the model emits, but keeping the prompt strict avoids burning
# tokens on a free-form answer that pydantic-ai then has to coerce.
_OMLX_PROMPT = (
    "You classify medical images by imaging modality. Answer with "
    "exactly one of: ultrasound, ct, xray, dermoscopy, histopathology, "
    "photo, document, unknown. Pick 'unknown' only when none of the "
    "named modalities fits. Do not explain — just emit the single word."
)


class OmlxLlmClassifier:
    """Multimodal LLM classifier via pydantic-ai + a local provider.

    The provider must be ``kind=local`` and vision-capable. Cloud
    providers are rejected at construct time — sending raw medical
    images off-device requires explicit opt-in and is out of scope for
    the benchmark.
    """

    name = "omlx_llm"

    def __init__(self, *, provider_id: str = "omlx") -> None:
        from claritymed.core.llm.model import build_model
        from claritymed.stores.models import is_provider_available, resolve_provider

        provider = resolve_provider(override=provider_id)
        if provider.kind != "local":
            raise RuntimeError(
                f"OmlxLlmClassifier requires a local provider; {provider_id!r} "
                f"has kind={provider.kind!r}. Cloud providers would send PHI off-device."
            )
        if not is_provider_available(provider):
            raise RuntimeError(
                f"provider {provider_id!r} is not reachable — start the local "
                f"server or pick a different provider_id."
            )
        self._provider_id = provider_id
        self._model = build_model(provider)

    async def classify(self, image_bytes: bytes, *, sha256: str) -> str:
        from pydantic_ai import Agent, BinaryContent

        # Sniff the MIME type from the magic bytes. PNG / JPEG cover all
        # downloaded fixtures; defaulting to JPEG when neither matches
        # keeps the call from failing on an unrecognized prefix (the LLM
        # backend usually accepts either).
        media_type = _guess_media_type(image_bytes)
        binary = BinaryContent(data=image_bytes, media_type=media_type)
        agent: Agent[None, str] = Agent(
            self._model, system_prompt=_OMLX_PROMPT, output_type=str
        )
        result = await agent.run([binary])
        # The model may answer "ultrasound." or "Modality: ct" — normalize.
        raw = (result.output or "").strip().lower()
        for token in raw.replace(",", " ").replace(".", " ").split():
            if token in KNOWN_MODALITIES:
                return token
        return "unknown"


# --- ResNet placeholder ------------------------------------------------------


class ResnetModalityClassifier:
    """Placeholder for a future ResNet-class modality head.

    Raises at construct time so the runner surfaces a clear "wire this
    up" line. When a real classifier ships, replace the body with the
    adapter; tests pin the ``name`` attribute, so changing it is the
    only intentional API break.
    """

    name = "resnet"

    def __init__(self) -> None:
        raise NotImplementedError(
            "ResnetModalityClassifier is a placeholder — wire it to a "
            "real ResNet head trained on labeled modality data when one "
            "ships."
        )

    async def classify(self, image_bytes: bytes, *, sha256: str) -> str:
        raise NotImplementedError


# --- helpers ----------------------------------------------------------------


def _guess_media_type(image_bytes: bytes) -> str:
    """Return ``"image/png"`` / ``"image/jpeg"`` based on magic bytes."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/jpeg"


def sha256_bytes(data: bytes) -> str:
    """Convenience: hex sha256 of ``data`` — used by run.py for request ids."""
    return hashlib.sha256(data).hexdigest()


def load_image(path: Path) -> bytes:
    """Read an image file into memory.

    Pulled out so the runner doesn't repeat the ``Path.read_bytes``
    call and can swap in a decoder if a dataset ever ships in a
    format that needs conversion before classification (e.g. DICOM).
    """
    return path.read_bytes()


# --- factory ----------------------------------------------------------------


def build_classifiers(
    requested: list[str],
    *,
    medical_clip_url: str = DEFAULT_MEDICAL_CLIP_URL,
    omlx_provider_id: str = "omlx",
) -> dict[str, ModalityClassifier]:
    """Construct every requested classifier that can be built.

    Unknown ids raise. Buildable classifiers that can't reach their
    backing service log a single skip line and are omitted from the
    result — same skip semantics as the eligibility benchmark.
    """
    supported = {"medical_clip", "omlx_llm", "resnet"}
    unknown = [r for r in requested if r not in supported]
    if unknown:
        raise ValueError(
            f"unknown classifier ids: {unknown!r}; supported: {sorted(supported)!r}"
        )

    out: dict[str, ModalityClassifier] = {}
    if "medical_clip" in requested:
        try:
            out["medical_clip"] = MedicalClipClassifier(base_url=medical_clip_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("medical_clip classifier unavailable (%s); skipped", exc)
    if "omlx_llm" in requested:
        try:
            out["omlx_llm"] = OmlxLlmClassifier(provider_id=omlx_provider_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("omlx_llm classifier unavailable (%s); skipped", exc)
    if "resnet" in requested:
        try:
            out["resnet"] = ResnetModalityClassifier()
        except NotImplementedError as exc:
            logger.warning("resnet classifier not wired (%s); skipped", exc)
    return out


__all__ = [
    "DEFAULT_MEDICAL_CLIP_URL",
    "KNOWN_MODALITIES",
    "MedicalClipClassifier",
    "ModalityClassifier",
    "OmlxLlmClassifier",
    "ResnetModalityClassifier",
    "build_classifiers",
    "load_image",
    "sha256_bytes",
]


# --- aclose helper for run.py -----------------------------------------------


async def aclose_classifiers(classifiers: dict[str, ModalityClassifier]) -> None:
    """Close any classifier that owns a transport (medical_clip).

    Pulled out so run.py doesn't isinstance-check every classifier;
    keeps the run.py event loop teardown one line.
    """
    for c in classifiers.values():
        aclose = getattr(c, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                logger.debug("classifier %s aclose raised", c.name, exc_info=True)
