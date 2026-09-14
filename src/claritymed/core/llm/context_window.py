"""Static best-guess mapping of model name → context window in tokens.

The catalog (``configs/models.yaml``) does not carry a context-window field —
adding one would put the value at risk of drifting from the vendor's real
support as model versions update. Instead the web layer keeps a small
pattern table here so the model picker can render a ratio
(``12k / 128k``) without burdening the YAML.

Unknown models fall back to :data:`DEFAULT_WINDOW`. The frontend treats
zero/None as "context window unknown" and hides the ratio.
"""

from __future__ import annotations

# Patterns are checked in order, first substring hit wins. Lowercase-only.
# When adding entries, put longer/more-specific patterns first.
_WINDOW_PATTERNS: tuple[tuple[str, int], ...] = (
    ("claude-opus-4-7", 1_000_000),
    ("claude-opus", 200_000),
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-sonnet", 200_000),
    ("claude-haiku", 200_000),
    ("claude", 200_000),
    ("gpt-4o", 128_000),
    ("gpt-4", 128_000),
    ("o1-", 128_000),
    ("deepseek-v4", 128_000),
    ("deepseek", 64_000),
    ("gemini-3", 1_000_000),
    ("gemini-2.5", 1_000_000),
    ("gemini", 32_000),
    ("kimi-k2", 128_000),
    ("kimi", 128_000),
    ("qwen-max", 32_000),
    ("qwen3.6", 128_000),
    ("qwen3", 32_000),
    ("qwen", 32_000),
    ("llama", 32_000),
    ("mistral", 32_000),
)

DEFAULT_WINDOW = 32_000


def estimate_context_window(model_name: str) -> int:
    """Return a best-guess context window for ``model_name``.

    ``model_name`` may be the pydantic-ai ``<prefix>:<name>`` form or a
    bare self-hosted name — both are lower-cased and substring-matched.
    """
    name = model_name.lower()
    for pat, n in _WINDOW_PATTERNS:
        if pat in name:
            return n
    return DEFAULT_WINDOW


def model_family(model_name: str) -> str:
    """Coarse family bucket inferred from the model string.

    Used by the UI to group providers in the picker (Claude / GPT /
    Gemini / DeepSeek / Qwen / Local). Defaults to ``"other"`` so a new
    model lands as a flat row and gets bucketed when somebody updates
    this map.
    """
    name = model_name.lower()
    if "claude" in name:
        return "claude"
    if "gpt" in name or "o1" in name:
        return "openai"
    if "gemini" in name:
        return "gemini"
    if "deepseek" in name:
        return "deepseek"
    if "kimi" in name or "moonshot" in name:
        return "kimi"
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    if "mistral" in name:
        return "mistral"
    return "other"
