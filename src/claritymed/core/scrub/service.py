"""Free-text PII scrubbing service: regex first, privacy-filter model as fallback.

Two-layer pipeline:

1. Regex pass — fast, zero-dependency, covers structured PII (phone, email,
   ID, MRN). Rules declared in ``configs/safety.yaml`` under
   ``phi.free_text_patterns``.

2. Model pass — OpenAI Privacy Filter (1.5B-param bidirectional token
   classifier). Catches unstructured PII the regex layer misses: names,
   addresses, dates, secrets, URLs, account numbers. Two sub-paths:

   a. ONNX path (default, ~809 MB): ``onnx_file`` set in config →
      raw ``onnxruntime.InferenceSession``. Bypasses ``AutoConfig`` entirely
      (which fails for the non-standard ``openai_privacy_filter`` model type).
      Label mapping is read from ``config.json`` as plain JSON.
      CoreMLExecutionProvider used on Apple Silicon when available; falls
      back to CPUExecutionProvider.
   b. PyTorch path (``onnx_file: null``): ``transformers`` pipeline on
      the device chosen by ``resolve_device``. ~2.8 GB safetensors.

   Opt-in via ``phi.privacy_filter.enabled``.

Both layers run regardless of provider kind (local or cloud) — user privacy
is not solely a cloud-egress concern.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from claritymed import config as _cfg
from claritymed.core.device import resolve_device

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"


@contextlib.contextmanager
def _silence_fd2():
    """Redirect C-level stderr (fd 2) to /dev/null.

    onnxruntime's CoreML execution provider writes diagnostic messages
    directly to fd 2, bypassing Python's sys.stderr entirely. In a TUI
    (Textual) session sys.stderr is replaced with a pseudo-file that has
    no real fileno(), so using sys.stderr.fileno() fails silently and
    leaves fd 2 open. We use the literal fd number 2 instead, which is
    always the OS-level stderr regardless of Python's sys.stderr state.
    """
    try:
        saved = os.dup(2)
    except OSError:
        # fd 2 not open (unusual test environments) — nothing to redirect
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def _emit_scrub_audit(payload: dict) -> None:
    """Emit scrub.privacy_filter audit event; silently skips if context is unset."""
    try:
        from claritymed.core.observability.audit import audit_event

        audit_event("scrub.privacy_filter", payload)
    except Exception:  # MissingContextError or anything else  # noqa: BLE001
        logger.debug("scrub.privacy_filter audit skipped (no request context)")


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


class _OnnxNerPipeline:
    """Minimal NER pipeline backed by an ``onnxruntime.InferenceSession``.

    Replicates the output contract of ``transformers.pipeline(
    "token-classification", aggregation_strategy="simple")``:
    a list of dicts with ``entity_group``, ``start``, ``end``.

    Aggregates consecutive subword tokens with the same label (BIO or flat)
    into a single span, skipping special tokens (offset start == end).
    """

    def __init__(self, session: Any, tokenizer: Any, id2label: dict[int, str]) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._id2label = id2label
        self._input_names: frozenset[str] = frozenset(
            i.name for i in session.get_inputs()
        )

    def __call__(self, text: str) -> list[dict]:
        enc = self._tokenizer(
            text,
            return_tensors="np",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping = enc.pop("offset_mapping")[0]  # (seq_len, 2)
        feed = {k: v for k, v in enc.items() if k in self._input_names}
        # Silence CoreML per-inference diagnostics written directly to fd 2.
        with _silence_fd2():
            logits = self._session.run(None, feed)[0][0]  # (seq_len, num_labels)
        predictions = logits.argmax(axis=-1)
        return self._aggregate(predictions, offset_mapping)

    def _aggregate(self, predictions: Any, offset_mapping: Any) -> list[dict]:
        """Merge subword tokens into char-offset spans using BIOES decoding.

        The model uses BIOES tagging (Begin/Inside/Outside/End/Single):
        - B-X: opens a new span of type X
        - I-X: extends the current B-X span
        - E-X: extends and closes the current span
        - S-X: single-token span (open + close immediately)
        - O:   closes any open span
        """
        spans: list[dict] = []
        current: dict | None = None

        for pred, (char_start, char_end) in zip(predictions, offset_mapping):
            # Special tokens ([CLS], [SEP], padding) have zero-length offsets
            if int(char_start) == int(char_end):
                if current:
                    spans.append(current)
                    current = None
                continue

            raw_label = self._id2label.get(int(pred), "O")
            prefix = raw_label[:2]

            if prefix == "B-":
                label = raw_label[2:]
                if current:
                    spans.append(current)
                current = {
                    "entity_group": label,
                    "start": int(char_start),
                    "end": int(char_end),
                }

            elif prefix == "I-":
                label = raw_label[2:]
                if current and current["entity_group"] == label:
                    current["end"] = int(char_end)
                else:
                    # I- without a matching open span — treat as span start
                    if current:
                        spans.append(current)
                    current = {
                        "entity_group": label,
                        "start": int(char_start),
                        "end": int(char_end),
                    }

            elif prefix == "E-":
                label = raw_label[2:]
                if current and current["entity_group"] == label:
                    current["end"] = int(char_end)
                    spans.append(current)
                    current = None
                else:
                    # E- without matching open span — close anything open, emit this token
                    if current:
                        spans.append(current)
                    spans.append(
                        {
                            "entity_group": label,
                            "start": int(char_start),
                            "end": int(char_end),
                        }
                    )
                    current = None

            elif prefix == "S-":
                label = raw_label[2:]
                if current:
                    spans.append(current)
                    current = None
                spans.append(
                    {
                        "entity_group": label,
                        "start": int(char_start),
                        "end": int(char_end),
                    }
                )

            else:  # "O" or anything unrecognised
                if current:
                    spans.append(current)
                    current = None

        if current:
            spans.append(current)
        return spans


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

    Thread-safe: the pipeline is loaded at most once per instance under a lock.
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

    def check_runtime_deps(self) -> None:
        """Raise ImportError if required packages for the model layer are missing.

        Call at startup to fast-fail with a clear remediation hint rather
        than silently degrading to regex-only at the first scrub call.
        Does nothing when ``privacy_filter.enabled`` is False.
        """
        if not self._config.privacy_filter.enabled:
            return
        if self._config.privacy_filter.onnx_file:
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                raise ImportError(
                    "privacy_filter with onnx_file requires onnxruntime. "
                    "Install with: uv sync --extra privacy-filter"
                ) from None
        else:
            try:
                import torch  # noqa: F401
            except ImportError:
                raise ImportError(
                    "privacy_filter with onnx_file=null requires torch. "
                    "Install with: uv sync --extra privacy-filter"
                ) from None
        try:
            # transformers._configure_library_root_logger() resets the Python
            # log level on import, overwriting any pre-set we do.  The only
            # reliable hook is TRANSFORMERS_VERBOSITY, which it reads as its
            # initial level.  We use setdefault so a user-set env var wins.
            import os

            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            import transformers  # noqa: F401

            transformers.logging.set_verbosity_error()
            logger.debug(
                "transformers imported (ONNX path); PyTorch/TF/Flax "
                "backend warning suppressed via TRANSFORMERS_VERBOSITY"
            )
        except ImportError:
            raise ImportError(
                "privacy_filter.enabled=true requires transformers. "
                "Install with: uv sync --extra privacy-filter"
            ) from None

    def ensure_downloaded(self) -> bool:
        """Pre-download model weights to the HuggingFace cache.

        ONNX path (``onnx_file`` set): fetches only the specified ONNX file
        + its ``_data`` companion + tokenizer + ``config.json`` (~809 MB total
        for ``model_q4f16``). No custom Python code needed — we read
        ``config.json`` directly as JSON.

        PyTorch path (``onnx_file`` unset): fetches the safetensors weights,
        skipping TF/Flax/ONNX variants (~2.8 GB).

        The model is NOT loaded into memory here; that still happens lazily
        on the first ``scrub()`` call. Returns True if the cache is ready,
        False on failure or missing extras.
        """
        if not self._config.privacy_filter.enabled:
            return True
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError

            onnx_file = self._config.privacy_filter.onnx_file
            repo_id = self._config.privacy_filter.model_name
            kwargs: dict = (
                {
                    "allow_patterns": [
                        "config.json",
                        "tokenizer*.json",
                        "special_tokens_map.json",
                        onnx_file,
                        f"{onnx_file}_data",
                    ]
                }
                if onnx_file
                else {
                    "ignore_patterns": [
                        "*.msgpack",
                        "*.h5",
                        "flax_*",
                        "tf_*",
                        "onnx/*",
                    ]
                }
            )
            try:
                # Fast path: all files already in local cache — skip ETag check.
                snapshot_download(repo_id=repo_id, local_files_only=True, **kwargs)
                logger.debug("privacy-filter cache hit; skipping network check")
            except LocalEntryNotFoundError:
                # First run or cache evicted: download from HuggingFace Hub.
                logger.info("downloading privacy-filter model weights: %s", repo_id)
                snapshot_download(repo_id=repo_id, **kwargs)
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
        backend = "onnx" if self._config.privacy_filter.onnx_file else "torch"
        if pipe is None:
            _emit_scrub_audit({"status": "skipped", "backend": backend})
            return text, 0
        t0 = time.perf_counter()
        try:
            spans = pipe(text)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            scrubbed = self._apply_spans(text, spans)
            _emit_scrub_audit(
                {
                    "status": "ok",
                    "backend": backend,
                    "duration_ms": duration_ms,
                    "hits": len(spans),
                    "chars_in": len(text),
                    "chars_out": len(scrubbed),
                }
            )
            return scrubbed, len(spans)
        except Exception:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            _emit_scrub_audit(
                {
                    "status": "error",
                    "backend": backend,
                    "duration_ms": duration_ms,
                }
            )
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
        """Load pipeline via raw onnxruntime, bypassing AutoConfig entirely.

        ``openai/privacy-filter`` has a non-standard model type
        (``openai_privacy_filter``) that is not registered in the transformers
        library. Calling ``AutoConfig.from_pretrained`` (which both the
        transformers pipeline and ``optimum.ORTModelForTokenClassification``
        do internally) raises ``ValueError`` / ``KeyError``. This method
        avoids that by:

        - Using ``ort.InferenceSession`` directly on the local ONNX file.
        - Reading ``config.json`` as plain JSON to get ``id2label``.
        - Using ``AutoTokenizer`` only (tokenizers are not model-type-gated).

        Providers: tries CoreMLExecutionProvider first (Apple Silicon), then
        falls back to CPUExecutionProvider.
        """
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import PreTrainedTokenizerFast

            model_name = self._config.privacy_filter.model_name

            # Resolve cached file paths (downloads only if not already cached)
            onnx_path = hf_hub_download(repo_id=model_name, filename=onnx_file)
            config_path = hf_hub_download(repo_id=model_name, filename="config.json")

            # Read id2label from plain JSON — no AutoConfig needed
            with open(config_path) as fh:
                id2label = {
                    int(k): v for k, v in json.load(fh).get("id2label", {}).items()
                }

            # tokenizer_config.json specifies "tokenizer_class": "TokenizersBackend"
            # which is not registered in transformers and has no .py in the repo.
            # Load tokenizer.json directly via the fast tokenizer constructor —
            # this bypasses the class dispatch entirely and avoids the warning.
            tok_path = hf_hub_download(repo_id=model_name, filename="tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)

            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            # Suppress CoreML's C-level stderr diagnostics — they write directly
            # to fd 2 and corrupt TUI display if not redirected.
            with _silence_fd2():
                session = ort.InferenceSession(onnx_path, providers=providers)
            used = session.get_providers()
            logger.info("privacy-filter ONNX session ready (providers: %s)", used)

            return _OnnxNerPipeline(session, tokenizer, id2label)

        except ImportError:
            logger.warning(
                "onnxruntime not installed; privacy-filter layer disabled. "
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
