"""Fill in empty per-language strings in a symptoms_<dataset>.yaml.

Reads ``configs/i18n/<source>/symptoms_<dataset>.yaml`` as the truth
side, walks the matching ``configs/i18n/<target>/symptoms_<dataset>.yaml``,
and asks the configured LLM to translate every leaf whose target value
is blank. Idempotent — already-populated target entries are skipped, so
re-running after a partial run is safe.

Why a separate script (not the generator):
``scripts/generate_symptoms_i18n.py`` rebuilds the YAML *structure* from
the dataset's release_evidences/release_conditions JSONs and emits empty
strings for languages the corpus doesn't carry. This script fills those
empties with LLM translations. Keeping the two steps separate means a
schema-only refresh (re-run the generator) never accidentally overwrites
a translator's hand edits, and a translation refresh (re-run this) never
needs to touch the dataset's JSON.

Usage (provider API keys are loaded from ``~/.claritymed/.env`` via
:func:`claritymed.config.load_env_file`)::

    uv run claritymed-symptoms-translate-i18n \\
        --dataset-id ddxplus \\
        --source en \\
        --target zh \\
        --provider omlx \\
        --batch-size 25

Add ``--dry-run`` to preview the diff without writing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from claritymed.config import I18N_DIR, PROJECT_ROOT, load_env_file
from claritymed.core.llm.model import build_model, build_model_settings
from claritymed.stores.models import resolve_provider

logger = logging.getLogger("translate_symptoms_i18n")

# Hard ceiling — too large a batch and small models drop entries from the
# JSON dict; too small and we waste a model call's fixed overhead.
_DEFAULT_BATCH_SIZE = 25

# Human-readable target language names for the system prompt. The model
# does better with "Chinese" than "zh".
_LANG_NAME = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "fr": "French",
}


def _system_prompt(target_lang: str) -> str:
    """Tight system prompt — JSON in, JSON out, no prose."""
    target_name = _LANG_NAME.get(target_lang, target_lang)
    return (
        f"You translate short medical terms into {target_name}. "
        f"Input is a JSON object mapping ids to English strings. "
        f"Output the same object with every value replaced by its "
        f"{target_name} translation.\n\n"
        f"Rules:\n"
        f"1. Output ONLY the JSON object. No preamble, no markdown fence, "
        f"no commentary.\n"
        f"2. Preserve every input key exactly. Do not add, remove, or "
        f"rename keys.\n"
        f"3. Translate clinical terms with their {target_name} medical "
        f"counterparts where they exist (chest pain → 胸痛, dyspnea → "
        f"呼吸困难, nausea → 恶心). Lay phrasing stays lay.\n"
        f"4. Keep anatomical-region suffixes intact: "
        f"'iliac wing(R)' → '髂翼(右)', not '右髂翼'.\n"
        f"5. Yes/No question phrasing stays as a question in "
        f"{target_name}.\n"
        f"6. Do not add periods, ellipses, or other punctuation that the "
        f"source string does not contain.\n"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file as a nested dict; missing file → empty dict."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"unexpected shape for {path}: expected dict")
    return data


def _walk_leaves(node: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Yield ``(dotted_key, value)`` for every string leaf in ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            full = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_leaves(value, full)
    elif isinstance(node, str):
        yield prefix, node


def _set_dotted(tree: dict, dotted: str, value: str) -> None:
    """Set ``tree[a][b][c] = value`` from ``"a.b.c"``."""
    parts = dotted.split(".")
    cursor: dict = tree
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _collect_pending(source_tree: dict, target_tree: dict) -> dict[str, str]:
    """Return ``{key: source_value}`` for every leaf where target is blank.

    Skips keys where the source itself is blank (no translatable text)
    and keys where the target already has a non-empty value (idempotent
    re-runs).
    """
    target_map = dict(_walk_leaves(target_tree))
    pending: dict[str, str] = {}
    for key, source_value in _walk_leaves(source_tree):
        if not source_value.strip():
            continue
        current = target_map.get(key, "")
        if current.strip():
            continue
        pending[key] = source_value
    return pending


def _chunks(items: list[tuple[str, str]], size: int) -> Iterable[list[tuple[str, str]]]:
    """Split a list into fixed-size batches."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def _translate_batch(
    agent: Any,
    batch: list[tuple[str, str]],
) -> dict[str, str]:
    """Send one batch to the agent; parse + validate the JSON response.

    Returns a dict that always covers every key in ``batch``. Missing
    keys in the model's response are left as empty strings so the
    caller can re-batch them.
    """
    payload = {f"k{i}": v for i, (_, v) in enumerate(batch)}
    user_msg = json.dumps(payload, ensure_ascii=False)
    result = await agent.run(user_msg)
    raw = getattr(result, "output", str(result)).strip()
    # Strip optional ```json fences — small models sometimes ignore the
    # "no markdown" rule. The first '{' to the matching last '}' is the
    # safest extraction without re-implementing JSON tokenizing.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.warning("model returned non-JSON: %r", raw[:200])
        return {key: "" for key, _ in batch}
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.warning("JSON decode failed: %s — payload: %r", exc, raw[:200])
        return {key: "" for key, _ in batch}
    if not isinstance(parsed, dict):
        logger.warning("model JSON not a dict: %r", parsed)
        return {key: "" for key, _ in batch}
    out: dict[str, str] = {}
    for i, (key, _) in enumerate(batch):
        translated = parsed.get(f"k{i}", "")
        if not isinstance(translated, str):
            translated = str(translated)
        out[key] = translated.strip()
    return out


def _build_agent(provider_id: str, system_prompt: str) -> Any:
    """Construct a pydantic-ai Agent against the resolved provider.

    Local import of ``pydantic_ai`` keeps the module-level import graph
    light when the user is only inspecting the script with ``--help``.
    """
    from pydantic_ai import Agent

    provider = resolve_provider(override=provider_id)
    model = build_model(provider)
    settings = build_model_settings(provider)
    if settings is None:
        return Agent(model=model, system_prompt=system_prompt, output_type=str)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=str,
        model_settings=settings,
    )


def _yaml_dump(payload: dict) -> str:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )


def _preserve_header(path: Path) -> str:
    """Return the leading comment block of an existing YAML, or default header."""
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header: list[str] = []
    for line in lines:
        if line.startswith("#"):
            header.append(line)
        elif line.strip() == "":
            header.append(line)
        else:
            break
    return "".join(header)


async def _run(args: argparse.Namespace) -> int:
    source_path = I18N_DIR / args.source / f"symptoms_{args.dataset_id}.yaml"
    target_path = I18N_DIR / args.target / f"symptoms_{args.dataset_id}.yaml"
    source_tree = _load_yaml(source_path)
    target_tree = _load_yaml(target_path)
    if not source_tree:
        raise SystemExit(f"source YAML is empty: {source_path}")

    pending = _collect_pending(source_tree, target_tree)
    if not pending:
        logger.info("nothing to translate — every target entry is already filled")
        return 0
    logger.info(
        "translating %d entries (%s → %s) via provider %s",
        len(pending),
        args.source,
        args.target,
        args.provider,
    )
    if args.dry_run:
        for key, value in list(pending.items())[:10]:
            print(f"would translate {key} = {value!r}")
        print(f"... ({len(pending)} total)")
        return 0

    agent = _build_agent(args.provider, _system_prompt(args.target))
    items = list(pending.items())
    translated_total = 0
    for batch in _chunks(items, args.batch_size):
        batch_result = await _translate_batch(agent, batch)
        for key, value in batch_result.items():
            if not value:
                continue
            _set_dotted(target_tree, key, value)
            translated_total += 1
        logger.info(
            "batch done — %d/%d entries translated so far",
            translated_total,
            len(pending),
        )

    header = _preserve_header(target_path)
    body = _yaml_dump(target_tree)
    target_path.write_text(header + body, encoding="utf-8")
    logger.info(
        "wrote %d translated entries to %s",
        translated_total,
        target_path.relative_to(PROJECT_ROOT),
    )
    if translated_total < len(pending):
        logger.warning(
            "%d entries were not translated; re-run the script to retry",
            len(pending) - translated_total,
        )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-id", required=True, help="e.g. 'ddxplus'")
    ap.add_argument("--source", default="en", help="Source language code (default: en)")
    ap.add_argument(
        "--target",
        required=True,
        help="Target language code to fill (e.g. 'zh').",
    )
    ap.add_argument(
        "--provider",
        required=True,
        help="ModelsConfig provider id (e.g. 'omlx', 'ollama').",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Entries per LLM call (default: {_DEFAULT_BATCH_SIZE}).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be translated without calling the LLM or writing.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging.",
    )
    return ap


def main() -> None:
    load_env_file()
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose or args.dry_run else logging.WARNING,
        format="%(message)s",
    )
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
