"""``claritymed init-user`` — create an account + data directory.

The first user created on a fresh install is auto-promoted to ``admin``;
later users default to ``user``. Idempotent — re-running with the same
``user_id`` returns the existing account unchanged.
"""

from __future__ import annotations

import json

import typer

from claritymed.cli.common import bootstrap_once, console


def init_user_cmd(
    user_id: str = typer.Argument(
        ..., help="User ID to create (alphanumeric, hyphens, underscores)."
    ),
    display_name: str | None = typer.Option(
        None, "--name", "-n", help="Display name (defaults to user_id)."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Emit JSON on stdout (machine-readable)."
    ),
) -> None:
    """Create a user account and initialise their data directory.

    The first user created on a fresh install is automatically promoted to
    admin.  Subsequent users receive the ``user`` role.

    Idempotent — safe to run more than once for the same user_id.
    """
    from claritymed.stores.account import init_user

    bootstrap_once()
    account = init_user(user_id, display_name=display_name)
    if json_out:
        print(
            json.dumps(
                {
                    "user_id": account.user_id,
                    "role": account.role,
                    "display_name": account.display_name,
                },
                ensure_ascii=False,
            )
        )
        return
    console.print(
        f"[green]✓[/green] user=[bold]{account.user_id}[/bold]  "
        f"role={account.role}  display_name={account.display_name!r}"
    )
