"""Free-text PII scrubbing service: regex first, privacy-filter model as fallback.

Two-layer pipeline:

1. Regex pass — fast, zero-dependency, covers structured PII (phone, email,
   ID, MRN). Rules declared in ``configs/safety.yaml`` under
   ``phi.free_text_patterns``.

2. Model pass — OpenAI Privacy Filter (1.5B-param bidirectional token
   classifier). Catches unstructured PII the regex layer misses: names,
   addresses, dates, secrets, URLs, account numbers. Two sub-paths:

   a. ONNX path (default, ~809 MB): ``onnx_file`` set in config →
      ``ORTModelForTokenClassification`` via ``optimum[onnxruntime]``.
      CPU-only; avoids MPS/CUDA compatibility issues and loads ~3× faster.
   b. PyTorch path (``onnx_file: null``): ``transformers`` pipeline on
      the device chosen by ``resolve_device``. ~2.8 GB safetensors.

   Opt-in via ``phi.privacy_filter.enabled``.

Both layers run regardless of provider kind (local or cloud) — user privacy
is not solely a cloud-egress concern.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from claritymed import config as _cfg
from claritymed.core.device import resolve_device

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

# HF entity_group label → our [REDACTED:X] placeholder
_LABEL_MAP: dict[str, str] = {
    "private_person": "[REDACTED:PERSON]",
    "private_address": "[REDACTED:ADDRESS]",
    "private_email": "[REDACTED:EMAIL]",
    "private_phone": "[REDACTED:PHONE]",
    "private_url": "[REDACTED:URL]",
    "private_date": "[REDACTED:DATE]",
    "account_number": "[REDACTED:ACCOUNT]",
    "secret": "[REDACTED:SECRET]",
}


def _patch_tqdm_lock() -> None:
    """Replace tqdm's class-level lock with a threading.RLock.

    tqdm's default TqdmDefaultWriteLock creates a multiprocessing.RLock
    which spawns a resource_tracker subprocess (spawnv_passfds). That spawn
    fails on macOS with uv's Python 3.12, crashing any code that uses tqdm —
    including transformers weight-loading progress bars. A threading.RLock is
    sufficient for single-process use and avoids the subprocess entirely.

    Idempotent: calling multiple times is safe.
    """
    import threading

    try:
        import tqdm

        tqdm.tqdm.set_lock(threading.RLock())
    except Exception:  # noqa: BLE001
        pass


class FreeTextRule(BaseModel):
    """One regex rule for free-text PII scrubbing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    regex: str
    replacement: str


class PrivacyFilterConfig(BaseModel):
    """OpenAI Privacy Filter model config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    device: str = "auto"  # "auto" → mps → cuda → cpu; or explicit "cpu"/"mps"/"cuda"
    model_name: str = "openai/privacy-filter"
    # ONNX quantized file path relative to the HF repo root (~809 MB total
    # including the _data companion). Set to null to use PyTorch safetensors.
    onnx_file: str | None = "onnx/model_q4f16.onnx"


class ScrubConfig(BaseModel):
    """Parsed scrub settings from ``configs/safety.yaml`` ``phi`` section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    free_text_patterns: list[FreeTextRule] = Field(default_factory=list)
    privacy_filter: PrivacyFilterConfig = Field(default_factory=PrivacyFilterConfig)


class ScrubReport(BaseModel):
    """Per-call summary of what the scrub pipeline did.

    Counts only — no original spans recorded, so the report is safe to
    emit as audit log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_hits: dict[str, int] = Field(default_factory=dict)
    model_hits: int = 0
    text_len_before: int = 0
    text_len_after: int = 0


class ScrubService:
    """Two-layer free-text PII scrubber: regex + optional privacy-filter model.

    Thread-safe: the transformers pipeline is loaded at most once per instance
    under a lock.
    """

    def __init__(self, config: ScrubConfig) -> None:
        self._config = config
        self._pipeline: Any = None
        self._pipeline_tried = False
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls) -> "ScrubService":
        """Load config from safety.yaml and return a ready service."""
        _cfg.reload_configs()
        phi_raw = _cfg.load_yaml("safety.yaml").get("phi") or {}
        config = ScrubConfig.model_validate(
            {
                "free_text_patterns": phi_raw.get("free_text_patterns", []),
                "privacy_filter": phi_raw.get("privacy_filter", {}),
            }
        )
        return cls(config)

    def scrub(self, text: str) -> tuple[str, ScrubReport]:
        """Run regex then model pass. Returns (scrubbed_text, report)."""
        if not text:
            return text, ScrubReport(text_len_before=0, text_len_after=0)

        original_len = len(text)

        scrubbed, rule_hits = self._layer_regex(text)

        model_hits = 0
        if self._config.privacy_filter.enabled:
            scrubbed, model_hits = self._layer_model(scrubbed)

        return scrubbed, ScrubReport(
            rule_hits=rule_hits,
            model_hits=model_hits,
            text_len_before=original_len,
            text_len_after=len(scrubbed),
        )

    def ensure_downloaded(self) -> bool:
        """Pre-download model weights to the HuggingFace cache.

        ONNX path (``onnx_file`` set): fetches only the specified ONNX file
        + its ``_data`` companion + tokenizer files — roughly 809 MB total
        for ``model_q4f16``.

        PyTorch path (``onnx_file`` unset): fetches the safetensors weights,
        skipping TF/Flax/ONNX variants — roughly 2.8 GB.

        The model is NOT loaded into memory here; that still happens lazily
        on the first ``scrub()`` call. Returns True if the cache is ready,
        False on failure or missing extras.
        """
        if not self._config.privacy_filter.enabled:
            return True
        try:
            from huggingface_hub import snapshot_download

            onnx_file = self._config.privacy_filter.onnx_file
            repo_id = self._config.privacy_filter.model_name
            if onnx_file:
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[
                        "config.json",
                        "tokenizer*.json",
                        "special_tokens_map.json",
                        "vocab.txt",
                        onnx_file,
                        f"{onnx_file}_data",
                    ],
                )
            else:
                snapshot_download(
                    repo_id=repo_id,
                    ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*", "onnx/*"],
                )
            return True
        except ImportError:
            logger.warning("huggingface_hub not installed; skipping pre-download")
            return False
        except Exception:
            logger.exception(
                "could not pre-download %s", self._config.privacy_filter.model_name
            )
            return False

    def _layer_regex(self, text: str) -> tuple[str, dict[str, int]]:
        rule_hits: dict[str, int] = {}
        for rule in self._config.free_text_patterns:
            pattern = re.compile(rule.regex)
            new_text, count = pattern.subn(rule.replacement, text)
            if count > 0:
                rule_hits[rule.name] = count
                text = new_text
        return text, rule_hits

    def _layer_model(self, text: str) -> tuple[str, int]:
        """Run privacy-filter pipeline. Returns input unchanged on failure."""
        pipe = self._get_pipeline()
        if pipe is None:
            return text, 0
        try:
            spans = pipe(text)
            scrubbed = self._apply_spans(text, spans)
            return scrubbed, len(spans)
        except Exception:
            logger.exception("privacy-filter scrub failed; using regex-only output")
            return text, 0

    def _get_pipeline(self) -> Any:
        """Return the cached pipeline (ONNX or PyTorch), or None if unavailable."""
        if self._pipeline_tried:
            return self._pipeline
        with self._lock:
            if self._pipeline_tried:
                return self._pipeline
            self._pipeline_tried = True
            onnx_file = self._config.privacy_filter.onnx_file
            if onnx_file:
                self._pipeline = self._load_onnx_pipeline(onnx_file)
            else:
                self._pipeline = self._load_torch_pipeline()
        return self._pipeline

    def _load_onnx_pipeline(self, onnx_file: str) -> Any:
        """Load pipeline via ONNX Runtime (CPU; avoids MPS/CUDA op gaps)."""
        try:
            from optimum.onnxruntime import ORTModelForTokenClassification
            from transformers import AutoTokenizer, pipeline

            model_name = self._config.privacy_filter.model_name
            model = ORTModelForTokenClassification.from_pretrained(
                model_name, file_name=onnx_file
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe = pipeline(
                task="token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
            logger.info("privacy-filter ONNX pipeline loaded (%s)", onnx_file)
            return pipe
        except ImportError:
            logger.warning(
                "optimum[onnxruntime] not installed; privacy-filter layer disabled. "
                "Install with: uv sync --extra privacy-filter"
            )
            return None
        except Exception:
            logger.exception("privacy-filter ONNX pipeline load failed; layer disabled")
            return None

    def _load_torch_pipeline(self) -> Any:
        """Load pipeline via PyTorch transformers (GPU-capable, larger footprint)."""
        try:
            from transformers import pipeline

            # tqdm's default lock is a multiprocessing.RLock whose init
            # spawns a resource_tracker subprocess via spawnv_passfds —
            # that spawn fails with uv's Python 3.12 on macOS. Pre-set
            # tqdm's class-level lock to a threading.RLock so the mp path
            # is never taken, regardless of what transformers does internally.
            _patch_tqdm_lock()

            device = resolve_device(self._config.privacy_filter.device)
            try:
                pipe = pipeline(
                    task="token-classification",
                    model=self._config.privacy_filter.model_name,
                    aggregation_strategy="simple",
                    device=device,
                )
                logger.info("privacy-filter pipeline loaded on %s", device)
                return pipe
            except Exception:
                if device == "cpu":
                    raise
                # Non-CPU backends (MPS, CUDA) may not support all ops in
                # this model. Fall back to CPU rather than disabling the layer.
                logger.warning("privacy-filter failed on %s, retrying on cpu", device)
                pipe = pipeline(
                    task="token-classification",
                    model=self._config.privacy_filter.model_name,
                    aggregation_strategy="simple",
                    device="cpu",
                )
                logger.info("privacy-filter pipeline loaded on cpu (fallback)")
                return pipe
        except ImportError:
            logger.warning(
                "transformers not installed; privacy-filter layer disabled. "
                "Install with: uv sync --extra privacy-filter"
            )
            return None
        except Exception:
            logger.exception("privacy-filter pipeline load failed; layer disabled")
            return None

    @staticmethod
    def _apply_spans(text: str, spans: list[dict]) -> str:
        """Replace detected spans in reverse offset order to preserve indices."""
        if not spans:
            return text
        for span in sorted(spans, key=lambda s: s["start"], reverse=True):
            label = span["entity_group"]
            placeholder = _LABEL_MAP.get(label, REDACTED)
            text = text[: span["start"]] + placeholder + text[span["end"] :]
        return text
