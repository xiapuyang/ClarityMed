"""BiomedCLIP engine wrapper — zero-shot modality classification.

The engine is the boundary between the FastAPI app and the underlying
``open_clip`` runtime. Three responsibilities:

1. **Load the model at a pinned revision.** ``open_clip`` resolves
   ``hf-hub:microsoft/BiomedCLIP-...`` against the HuggingFace Hub at
   load time. If a non-empty revision is configured, the engine
   cross-checks the actual served revision via ``huggingface_hub`` and
   refuses to start on mismatch — silent embedding drift would
   invalidate the modality thresholds tuned against a specific
   revision (Unit 2's gating calibration).

2. **Precompute label-prompt embeddings.** Each modality label carries
   one or more text prompts (``configs/medical_clip.yaml::tasks.modality.candidates``).
   The engine encodes every prompt at lifespan, averages per-label, and
   caches the result. Per-request work is then a single image-encoder
   pass plus a cosine dot-product.

3. **Score one image at a time.** ``classify(image_bytes)`` decodes,
   preprocesses, encodes, computes cosine similarity against the
   cached label embeddings, applies softmax, and returns ranked scores
   in canonical ``Modality`` Literal order so the FastAPI handler can
   build the wire response without re-indexing.

The heavy imports (``open_clip``, ``PIL``) are lazy inside the
classmethod so the rest of the package — including the FastAPI app's
testable surface — does not require the ``medical-clip-server`` extra
just to be imported.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from claritymed.core.medical_clip.schemas import Modality, ModalityScore

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# Softmax temperature used by BiomedCLIP at inference — open_clip's stock
# loader bakes a ``logit_scale`` parameter into the model. We re-read it
# from the model in ``classify`` rather than hardcoding so a future
# revision that recalibrates this value rides through cleanly.

# Image bytes ceiling. BiomedCLIP eats 224×224 crops; anything bigger is
# downsampled by the transform anyway, so capping the upload at 16 MB
# (~ a 4K JPEG) keeps a hostile client from exhausting memory before
# Pillow gets to fail. Bound is generous on purpose — chest CT scans
# saved as PNG occasionally cross 10 MB.
MAX_IMAGE_BYTES = 16 * 1024 * 1024


class ImageDecodeError(ValueError):
    """Raised by ``classify`` when the bytes don't decode to a valid image.

    The FastAPI handler translates this to a 400 response with
    ``code=image_decode_failed`` so the client (and ultimately the LLM)
    sees a stable, machine-readable error code.
    """


class ModelRevisionMismatchError(RuntimeError):
    """Raised at lifespan when HF Hub serves a revision other than the pinned one.

    Tightens the contract: a model swap on the Hub side would silently
    drift the modality classifier's calibration. We surface that as a
    hard startup failure rather than discovering it via slow recall
    degradation in production.
    """


@dataclass(frozen=True)
class ModalityCandidate:
    """One label + its candidate prompts. Lifespan input shape."""

    label: Modality
    prompts: tuple[str, ...]


class BiomedClipEngine:
    """Zero-shot modality classifier wrapping BiomedCLIP.

    Construct via :meth:`load`; do not call the constructor directly
    outside of tests (which inject stub engines under the same
    interface). The engine is stateful — label embeddings cached at
    load time are immutable for the process lifetime.
    """

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        preprocess: object,
        device: str,
        labels: list[Modality],
        label_embeddings: torch.Tensor,
        model_id: str,
        model_revision: str | None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self._device = device
        # Canonical label order — matches the row order of label_embeddings.
        # We never re-shuffle these; the FastAPI handler relies on the
        # order matching what classify() returns.
        self._labels = labels
        self._label_embeddings = label_embeddings
        self._model_id = model_id
        self._model_revision = model_revision

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str | None:
        return self._model_revision

    @property
    def device(self) -> str:
        return self._device

    @property
    def labels(self) -> list[Modality]:
        return list(self._labels)

    # --- lifespan loader -------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        revision: str | None,
        device: str,
        candidates: list[ModalityCandidate],
    ) -> BiomedClipEngine:
        """Load BiomedCLIP + precompute label embeddings.

        Args:
            model_id: HuggingFace model id, e.g.
                ``"microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"``.
            revision: Pinned revision hash. Empty string or ``None`` skips
                the cross-check (early dev only); production deployments
                must pin so a Hub-side revision change cannot silently
                drift the calibration.
            device: ``"cpu"`` / ``"cuda"`` / ``"mps"`` / ``"auto"`` — the
                FastAPI app's lifespan resolves ``"auto"`` via
                ``servers._devices.default_device`` before calling us so
                logging shows the concrete choice.
            candidates: Per-label prompt lists from
                ``configs/medical_clip.yaml::tasks.modality.candidates``.

        Raises:
            ModelRevisionMismatchError: HF Hub serves a different
                revision than ``revision``.
            ImportError: ``open_clip`` is not installed; install with
                ``uv sync --extra medical-clip-server``.
        """
        if revision:
            cls._verify_revision(model_id, revision)

        # Lazy import — keeps the medical-clip-server extra optional for
        # consumers that only need the schemas / client.
        import open_clip  # type: ignore[import-not-found]

        logger.info("loading BiomedCLIP %s on %s", model_id, device)
        hf_hub_url = f"hf-hub:{model_id}"
        model, preprocess = open_clip.create_model_from_pretrained(hf_hub_url)
        tokenizer = open_clip.get_tokenizer(hf_hub_url)
        model = model.to(device)
        model.eval()

        labels, label_embeddings = cls._encode_candidates(
            model=model,
            tokenizer=tokenizer,
            candidates=candidates,
            device=device,
        )
        logger.info(
            "BiomedCLIP ready: %d labels, embedding dim %d",
            len(labels),
            label_embeddings.shape[1],
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            device=device,
            labels=labels,
            label_embeddings=label_embeddings,
            model_id=model_id,
            model_revision=revision or None,
        )

    @staticmethod
    def _verify_revision(model_id: str, pinned_revision: str) -> None:
        """Fail-loud if HF Hub serves a revision different from the pinned one."""
        # Lazy import for the same reason as open_clip — operators
        # without the extra installed shouldn't hit the import either.
        from huggingface_hub import HfApi  # type: ignore[import-not-found]

        try:
            info = HfApi().model_info(model_id)
        except Exception as exc:  # noqa: BLE001 — surface as a clean RuntimeError
            raise ModelRevisionMismatchError(
                f"BiomedCLIP revision verification failed for {model_id!r}: "
                f"could not query HF Hub ({exc!s})"
            ) from exc
        served = (info.sha or "").strip()
        if served != pinned_revision:
            raise ModelRevisionMismatchError(
                f"BiomedCLIP revision drift on {model_id!r}: pinned "
                f"{pinned_revision!r}, HF Hub serves {served!r}. Either "
                f"update configs/medical_clip.yaml::model.revision after "
                f"reviewing the diff, or pin to the previous revision."
            )

    @staticmethod
    def _encode_candidates(
        *,
        model: object,
        tokenizer: object,
        candidates: list[ModalityCandidate],
        device: str,
    ) -> tuple[list[Modality], torch.Tensor]:
        """Encode each label's prompts → averaged per-label embedding.

        Cosine semantics work on L2-normalized vectors, so we normalize
        once per prompt, average the prompt vectors for one label, then
        normalize the average. (Normalize-then-average then normalize
        again is the standard open_clip recipe and slightly more stable
        than averaging then normalizing once.)
        """
        labels: list[Modality] = []
        embeddings: list[torch.Tensor] = []
        with torch.inference_mode():
            for candidate in candidates:
                tokenized = tokenizer(list(candidate.prompts)).to(device)
                # open_clip's BiomedCLIP exposes encode_text returning
                # token-level features; the standard zero-shot recipe
                # uses the [CLS]-equivalent pooled output.
                features = model.encode_text(tokenized)
                features = features / features.norm(dim=-1, keepdim=True)
                pooled = features.mean(dim=0)
                pooled = pooled / pooled.norm()
                labels.append(candidate.label)
                embeddings.append(pooled)
        stacked = torch.stack(embeddings, dim=0).to(device)
        return labels, stacked

    # --- per-request classification --------------------------------------

    def classify(self, image_bytes: bytes) -> list[ModalityScore]:
        """Score one image against every cached label embedding.

        Returns the scoreboard in **descending score order**, ready for
        the FastAPI handler to read ``top1`` off ``scores[0]``.

        Raises:
            ImageDecodeError: The bytes don't decode as a valid image.
        """
        if not image_bytes:
            raise ImageDecodeError("empty image payload")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ImageDecodeError(
                f"image bytes {len(image_bytes)} exceed cap {MAX_IMAGE_BYTES}"
            )
        image = self._decode(image_bytes)
        scores = self._cosine_softmax(image)
        ranked = sorted(
            zip(self._labels, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [ModalityScore(label=label, score=score) for label, score in ranked]

    def _decode(self, image_bytes: bytes) -> Image:
        """Decode + RGB-convert. Pillow handles JPEG / PNG / WEBP / HEIC (with plugin)."""
        from PIL import Image as PILImage, UnidentifiedImageError

        try:
            with PILImage.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB").copy()
        except UnidentifiedImageError as exc:
            raise ImageDecodeError(
                "image bytes could not be decoded as a valid image"
            ) from exc
        except (OSError, ValueError) as exc:
            # Pillow surfaces truncated JPEGs as OSError and corrupted
            # PNGs as ValueError. Either way the right response is the
            # same `image_decode_failed` 400 — don't leak Pillow's
            # internals to the API caller.
            raise ImageDecodeError(
                f"image bytes could not be decoded: {exc!s}"
            ) from exc

    def _cosine_softmax(self, image: Image) -> list[float]:
        """Run the image encoder + cosine sim + temperature-scaled softmax."""
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            features = self._model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            # Logit scale baked into open_clip's BiomedCLIP — pulled
            # off the live model so calibration tracks revision.
            logit_scale = self._model.logit_scale.exp().clamp(max=100.0)
            logits = (features @ self._label_embeddings.T) * logit_scale
            probs = logits.softmax(dim=-1)[0]
        return [float(x) for x in probs.detach().cpu().tolist()]


__all__ = [
    "BiomedClipEngine",
    "ImageDecodeError",
    "MAX_IMAGE_BYTES",
    "ModalityCandidate",
    "ModelRevisionMismatchError",
]
