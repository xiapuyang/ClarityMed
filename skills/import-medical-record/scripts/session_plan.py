#!/usr/bin/env python3
"""Per-source multi-session plan: read / init / complete.

Stores a non-PHI plan at ``data/_skill_sessions/<source_id>/plan.yaml``
recording: source identifier hash, split strategy, discovered users,
completed sessions, and remaining scopes. The skill reads this at
step 2 (look up prior plan) and writes via init/complete after each
session.

Subcommands:

* ``read <source_id>``                            — emit JSON; if no plan,
                                                    ``{"exists": false}``.
* ``init <source_id> --description ... --split ...
       --users uid:role,... --remaining-scopes s1,s2,... [--force]``
                                                  — create the plan (refuse
                                                    if exists unless --force).
* ``complete <source_id> --scope <s> --import-id <id>
       --case-count <n> --fact-count <n>``         — move scope from
                                                    remaining_scopes to
                                                    sessions[]; atomic
                                                    rewrite.

All file ops are atomic via tmp + rename; file mode 0600; dir 0700.

Run via ``uv run python ...`` so the project's path helpers + DATA_DIR
resolution work the same as the rest of the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


SOURCE_ID_RE = r"^[a-f0-9]{8,32}$"


def _plan_dir(source_id: str) -> Path:
    from claritymed import config as _cfg

    return _cfg.DATA_DIR / "_skill_sessions" / source_id


def _plan_path(source_id: str) -> Path:
    return _plan_dir(source_id) / "plan.yaml"


def _validate_source_id(source_id: str) -> None:
    import re

    if not re.match(SOURCE_ID_RE, source_id):
        raise ValueError(f"invalid source_id: {source_id!r}")


def _atomic_write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _guard_source_id(source_id: str) -> int:
    try:
        _validate_source_id(source_id)
    except ValueError as exc:
        print(json.dumps({"error": "invalid_source_id", "detail": str(exc)}))
        return 2
    return 0


def cmd_read(args) -> int:
    code = _guard_source_id(args.source_id)
    if code:
        return code
    path = _plan_path(args.source_id)
    if not path.exists():
        print(json.dumps({"exists": False}))
        return 0
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["exists"] = True
    print(json.dumps(data, ensure_ascii=False))
    return 0


def cmd_init(args) -> int:
    code = _guard_source_id(args.source_id)
    if code:
        return code
    path = _plan_path(args.source_id)
    if path.exists() and not args.force:
        print(json.dumps({"error": "plan_exists", "source_id": args.source_id}))
        return 1
    users = {}
    for entry in (args.users or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            print(
                json.dumps(
                    {"error": "bad_users_format", "expected": "uid:role,uid:role"}
                )
            )
            return 2
        uid, role = entry.split(":", 1)
        users[uid.strip()] = role.strip()
    remaining = [
        s.strip() for s in (args.remaining_scopes or "").split(",") if s.strip()
    ]
    plan = {
        "source_id": args.source_id,
        "description": args.description,
        "split_strategy": args.split,
        "users": users,
        "remaining_scopes": remaining,
        "sessions": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_yaml(path, plan)
    print(json.dumps({"ok": True, "path": str(path)}))
    return 0


def cmd_complete(args) -> int:
    code = _guard_source_id(args.source_id)
    if code:
        return code
    path = _plan_path(args.source_id)
    if not path.exists():
        print(json.dumps({"error": "no_plan"}))
        return 1
    plan = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    remaining = list(plan.get("remaining_scopes", []))
    sessions = list(plan.get("sessions", []))
    if any(s.get("scope") == args.scope for s in sessions):
        print(json.dumps({"error": "scope_already_complete"}))
        return 1
    if args.scope not in remaining:
        print(json.dumps({"error": "scope_unknown"}))
        return 1
    remaining.remove(args.scope)
    sessions.append(
        {
            "scope": args.scope,
            "import_id": args.import_id,
            "case_count": args.case_count,
            "fact_count": args.fact_count,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    plan["remaining_scopes"] = remaining
    plan["sessions"] = sessions
    _atomic_write_yaml(path, plan)
    print(json.dumps({"ok": True, "remaining": remaining}))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="session_plan.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_read = sub.add_parser("read")
    p_read.add_argument("source_id")
    p_read.set_defaults(func=cmd_read)

    p_init = sub.add_parser("init")
    p_init.add_argument("source_id")
    p_init.add_argument("--description", default="")
    p_init.add_argument("--split", required=True)
    p_init.add_argument("--users", default="")
    p_init.add_argument("--remaining-scopes", default="")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_complete = sub.add_parser("complete")
    p_complete.add_argument("source_id")
    p_complete.add_argument("--scope", required=True)
    p_complete.add_argument("--import-id", required=True)
    p_complete.add_argument("--case-count", type=int, required=True)
    p_complete.add_argument("--fact-count", type=int, required=True)
    p_complete.set_defaults(func=cmd_complete)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
