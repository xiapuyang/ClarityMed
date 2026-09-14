"""Programmatic edits to ``configs/retrieval.yaml`` for the admin RAG flow.

The CLI script (``scripts/init_system_rag.py``) prints a snippet and
asks the operator to paste it manually — that posture preserves
reviewable config changes from a shell. The admin SPA endpoint takes
the opposite stance: after a successful ingest, the new collection
becomes routable immediately by auto-appending the snippet here. The
SPA caller can audit the diff in the Jobs UI; that is the review trail.

Implementation: text splice rather than YAML round-trip. ``retrieval.yaml``
has hand-written comments that document operational invariants (e.g.
"FROZEN: collection names are Qdrant index names on disk"). PyYAML
loses comments on dump; ``ruamel.yaml`` round-trips them but isn't in
the dependency set. Splicing the rendered snippet onto the end of the
existing ``system_rag.collections:`` block preserves every comment.

The helper is idempotent for the "already present" case — if ``name``
already appears under ``system_rag.collections``, it returns False and
leaves the file untouched. That matches the resolver's expectation that
append-mode ingest never duplicates an entry.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from claritymed import config as _cfg

logger = logging.getLogger(__name__)


def _retrieval_path() -> Path:
    return _cfg.CONFIGS_DIR / "retrieval.yaml"


def _find_collections_block(text: str) -> tuple[int, int] | None:
    """Locate the ``system_rag.collections:`` block in ``text``.

    Returns ``(block_start, block_end)`` — character offsets bounding
    the *contents* of the collections list (i.e. lines indented under
    ``  collections:``, excluding the ``collections:`` line itself and
    excluding the next sibling key under ``system_rag:``).

    ``None`` when the block isn't found — the caller fails loud rather
    than silently appending to nowhere.
    """
    # Find the system_rag: top-level key at column 0.
    sysrag_re = re.compile(r"^system_rag:\s*$", re.MULTILINE)
    m_sysrag = sysrag_re.search(text)
    if not m_sysrag:
        return None
    sysrag_end = m_sysrag.end()

    # Find collections: indented under system_rag (2 spaces).
    collections_re = re.compile(r"^  collections:\s*$", re.MULTILINE)
    m_coll = collections_re.search(text, sysrag_end)
    if not m_coll:
        return None

    # Block starts on the line after `  collections:`.
    block_start = (
        m_coll.end() + 1
    )  # skip the trailing newline of the `collections:` line
    if block_start > len(text):
        block_start = len(text)

    # Block ends at the next line that starts a sibling-or-shallower
    # YAML key. Anything under collections is indented >= 4 spaces (the
    # `- name:` entries) or blank or a `# comment` at deeper indent.
    # The first line indented <= 2 spaces (or 0 spaces) terminates it.
    block_end = len(text)
    line_start = block_start
    while line_start < len(text):
        nl = text.find("\n", line_start)
        if nl == -1:
            nl = len(text)
        line = text[line_start:nl]
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        # Blank or comment line — part of the block (don't terminate).
        if not stripped or stripped.startswith("#"):
            line_start = nl + 1
            continue
        # A line indented <= 2 spaces is a sibling under system_rag or
        # a new top-level key — terminate before it.
        if indent <= 2:
            block_end = line_start
            break
        line_start = nl + 1
    return block_start, block_end


def _collection_already_present(block_text: str, name: str) -> bool:
    """True when ``- name: <name>`` already appears in the block."""
    pattern = re.compile(rf"^\s*-\s+name:\s+{re.escape(name)}\s*(#.*)?$", re.MULTILINE)
    return bool(pattern.search(block_text))


def append_system_rag_collection(
    snippet: str,
    *,
    name: str,
    path: Path | None = None,
) -> bool:
    """Append ``snippet`` under ``system_rag.collections`` in retrieval.yaml.

    Returns True when the file was modified, False when ``name`` was
    already present (idempotent no-op). Raises ``RuntimeError`` when
    the ``system_rag.collections:`` block can't be located — better to
    fail loud than splice into the wrong section.

    ``snippet`` should be the rendered entry from
    :func:`claritymed.ingest.system_rag.build_yaml_snippet` (starts with
    ``"    - name: …"``, four-space indented).

    Writes atomically via ``os.replace`` from a temp file in the same
    directory so a crashed write never leaves a half-written config.
    """
    target = path or _retrieval_path()
    original = target.read_text(encoding="utf-8")

    block = _find_collections_block(original)
    if block is None:
        raise RuntimeError(
            f"could not locate `system_rag.collections:` block in {target}"
        )
    block_start, block_end = block

    if _collection_already_present(original[block_start:block_end], name):
        logger.info(
            "retrieval.yaml: collection %r already present; skipping append", name
        )
        return False

    # Trim trailing whitespace from the existing block so we always
    # insert with exactly one newline separator. Snippet has no trailing
    # newline; we add one when inserting.
    existing = original[block_start:block_end].rstrip("\n")
    new_block = f"{existing}\n{snippet}\n" if existing else f"{snippet}\n"
    new_text = original[:block_start] + new_block + original[block_end:]

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

    # Drop the load_yaml cache so the next read sees the appended entry.
    _cfg.load_yaml.cache_clear()
    logger.info("retrieval.yaml: appended collection %r", name)
    return True
