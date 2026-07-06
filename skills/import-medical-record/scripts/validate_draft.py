#!/usr/bin/env python3
"""Validate a draft template directory before invoking the CLI.

Wraps ``claritymed.ingest.records.template_loader.load_template`` so the
skill can pre-flight a draft without round-tripping the full template
through the model. Prints one JSON object on stdout:

* Success: ``{"ok": true, "import_id": "..."}``
* Failure: ``{"ok": false, "error": "..."}`` + non-zero exit code.

Run via ``uv run`` so the project's Python (and the package) are on
PATH:

    uv run python skills/import-medical-record/scripts/validate_draft.py <draft_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "usage: validate_draft.py <template_dir>",
                }
            )
        )
        return 2
    target = Path(sys.argv[1])
    try:
        from claritymed.ingest.records.template_loader import load_template

        loaded = load_template(target)
    except Exception as exc:  # noqa: BLE001 — top-level error funnel
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    payload = {
        "ok": True,
        "import_id": loaded.import_id,
        "user_ids": list(loaded.bundles.keys()),
        "case_counts": {uid: len(b.cases) for uid, b in loaded.bundles.items()},
        "warnings": list(loaded.warnings),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
