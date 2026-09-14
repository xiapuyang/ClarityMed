"""``claritymed user <subcommand>`` — out-of-band user-state management.

Currently exposes one subcommand:

* ``set-password <user_id>`` — interactive bcrypt hash write to
  ``data/users/<user_id>/auth.yaml``. The **only** sanctioned path
  for setting or rotating a password; the web layer never exposes a
  password endpoint (origin R13).

Refuses to run if the user has not been initialized (no
``settings.yaml``) so a typo in ``<user_id>`` does not create a
hash file under a sibling directory.
"""

from __future__ import annotations

import typer

from claritymed.cli.common import bootstrap_once, console
from claritymed.cli.entry import inject_context
from claritymed.core.observability.audit import audit_event
from claritymed.errors import InvalidUserIdError
from claritymed.stores.account import AccountStore
from claritymed.stores.auth import PasswordStore
from claritymed.stores.paths import validate_user_id

user_app = typer.Typer(
    name="user",
    help="Per-user state management (out-of-band password rotation).",
    no_args_is_help=True,
)

# Minimum password length — keeps "test" / "1234" out of production
# without imposing a complexity policy the project doesn't intend to
# enforce (no upper/symbol/digit rules).
MIN_PASSWORD_LENGTH = 8


@user_app.command("set-password")
def set_password_cmd(
    user_id: str = typer.Argument(
        ..., help="Existing user_id (must have settings.yaml on disk)."
    ),
) -> None:
    """Set or rotate the bcrypt hash for ``<user_id>``.

    Prompts twice for the password (confirmation). Refuses with a
    pointer to ``init-user`` if the user has not been initialised.
    Audits ``cli.user.password_set`` on success. The plaintext is
    NEVER echoed and never logged.
    """
    bootstrap_once()

    # Validate the user_id at the CLI boundary too; ``init_user`` and
    # PasswordStore both validate again but failing here gives a
    # better error message than a stack trace mid-store.
    try:
        validate_user_id(user_id)
    except InvalidUserIdError as exc:
        raise typer.BadParameter(str(exc)) from None

    if not AccountStore(user_id).exists():
        console.print(
            f"[red]✗[/red] user {user_id!r} not initialised — "
            f"run [bold]claritymed init-user {user_id}[/bold] first."
        )
        raise typer.Exit(code=2)

    password = typer.prompt(
        f"Password for {user_id!r}",
        hide_input=True,
        confirmation_prompt=True,
    )

    if len(password) < MIN_PASSWORD_LENGTH:
        # Surface as BadParameter so the exit code is non-zero AND the
        # user sees a sensible message; no audit event because no
        # business-meaningful action occurred yet.
        raise typer.BadParameter(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )

    # ``inject_context`` populates request_id / user_id / language so
    # audit_event() inside set_password's call chain (and the explicit
    # one below) never trips MissingContextError. Acts the user_id we
    # are setting the password FOR — that's the entity the audit row
    # is about.
    with inject_context(user_id=user_id, command="user.set-password"):
        PasswordStore.set_password(user_id, password)
        audit_event("cli.user.password_set", payload={"user_id": user_id})
    console.print(f"[green]✓[/green] password set for [bold]{user_id}[/bold]")
