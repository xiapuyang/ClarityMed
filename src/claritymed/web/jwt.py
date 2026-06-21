"""JWT issue / decode + secret-startup gate + IP HMAC helper.

The decode result is a discriminated union of three variants so the
middleware can dispatch fail-loud on tamper, silent on lifecycle:

* :class:`Valid` — signature OK, not expired, ``lang`` in the allowlist.
* :class:`Tamper` — signature OK, not expired, but ``lang`` (or another
  schema field) violates the allowlist. Implies secret compromise or a
  buggy issuer. Middleware clears the cookie and audits
  ``web.jwt.tamper_suspected``.
* :class:`Invalid` — signature mismatch / expired / malformed. Normal
  lifecycle. Middleware downgrades to anonymous and audits
  ``web.jwt.invalid`` only when a cookie was actually presented.

Secret comes from ``CLARITYMED_JWT_SECRET``. ``validate_secret_or_raise``
enforces at lifespan startup that the env is set AND not a known-bad
literal (empty string, ``changeme``, ``secret``) — the alternative is a
prod deployment that silently signs with a guessable secret.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Union

from jose import jwt as _jose_jwt
from jose.exceptions import ExpiredSignatureError, JWTError

ENV_JWT_SECRET = "CLARITYMED_JWT_SECRET"
JWT_ALGORITHM = "HS256"
DEFAULT_EXP_DAYS = 7

# Languages that may appear in the JWT `lang` claim. Mirrors
# ``Account.language`` (Literal["en", "zh"]). A token signed with a valid
# HMAC carrying any other value is treated as Tamper, not Invalid.
LANG_ALLOWLIST: frozenset[str] = frozenset({"en", "zh"})

# Refuse to boot if the operator left a placeholder in the env. The
# empty string lands here because ``os.environ.get(..., "")`` is the
# common shape. ``changeme`` / ``secret`` are the canonical "I forgot
# to rotate this" markers.
_BAD_SECRETS: frozenset[str] = frozenset({"", "changeme", "secret"})


# --- discriminated decode result ----------------------------------------


@dataclass(frozen=True)
class Valid:
    """Signature OK, not expired, all schema constraints satisfied."""

    claims: dict[str, Any]


@dataclass(frozen=True)
class Tamper:
    """Signature OK but a schema constraint (e.g. lang allowlist) failed.

    This is **not** a normal lifecycle event — it implies the signing
    secret was compromised or the issuer has a bug producing tokens
    outside the documented schema. The middleware MUST fail loud.
    """

    reason: str


@dataclass(frozen=True)
class Invalid:
    """Signature mismatch / expired / malformed JWT. Normal lifecycle."""

    reason: str


DecodeResult = Union[Valid, Tamper, Invalid]


# --- secret access ------------------------------------------------------


def _read_secret() -> str:
    """Return the configured JWT secret. Does not validate."""
    return os.environ.get(ENV_JWT_SECRET, "")


def validate_secret_or_raise() -> str:
    """Lifespan-side check: refuse to boot with no / placeholder secret.

    Returns the secret on success so callers can cache it. The check
    runs once at startup; per-request decode reads the env directly
    (cheap, lets tests monkeypatch between requests).
    """
    secret = _read_secret()
    if secret in _BAD_SECRETS:
        raise RuntimeError(
            f"{ENV_JWT_SECRET} must be set to a non-trivial value "
            f"(not one of {sorted(_BAD_SECRETS)!r}). "
            "Generate one with: openssl rand -hex 32"
        )
    return secret


# --- issue / decode -----------------------------------------------------


def create_token(sub: str, lang: str, exp_days: int = DEFAULT_EXP_DAYS) -> str:
    """Issue a JWT with the documented two-claim schema.

    Raises ``ValueError`` if ``lang`` is outside ``LANG_ALLOWLIST`` so
    a caller cannot accidentally mint a token that decode would have to
    classify as Tamper. Issue-time validation is the right place to
    catch this; Tamper at decode time should only ever come from
    outside the issuer.
    """
    if lang not in LANG_ALLOWLIST:
        raise ValueError(f"lang={lang!r} not in allowlist {sorted(LANG_ALLOWLIST)!r}")
    now = datetime.now(timezone.utc)
    claims: dict[str, Any] = {
        "sub": sub,
        "lang": lang,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=exp_days)).timestamp()),
    }
    return _jose_jwt.encode(claims, _read_secret(), algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> DecodeResult:
    """Return Valid / Tamper / Invalid for a presented cookie value.

    The dispatch order is deliberate: signature + expiry first (any
    failure is normal lifecycle → Invalid), then schema constraints on
    the claims (failure with a valid signature → Tamper). A buggy
    issuer that emits unknown-lang tokens with the real secret would
    otherwise look like a "logged out" event in the audit log.
    """
    try:
        claims = _jose_jwt.decode(
            token,
            _read_secret(),
            algorithms=[JWT_ALGORITHM],
        )
    except ExpiredSignatureError:
        return Invalid("expired")
    except JWTError as exc:
        return Invalid(f"jwt_error:{type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001
        # Catch-all for non-JOSE decode failures (e.g. corrupt header).
        # Bucketed as Invalid because the caller cannot tell whether
        # this token was ever-valid or just garbage.
        return Invalid(f"decode_error:{type(exc).__name__}")

    sub = claims.get("sub")
    lang = claims.get("lang")
    if not isinstance(sub, str) or not sub:
        return Tamper("missing_sub")
    if not isinstance(lang, str):
        return Tamper("missing_lang_claim")
    if lang not in LANG_ALLOWLIST:
        return Tamper("invalid_lang_claim")
    return Valid(claims)


# --- IP HMAC for login-spray forensics ---------------------------------


def hmac_ip(host: str) -> str:
    """Return a stable HMAC-SHA256 hex of ``host`` keyed by the JWT secret.

    Lets post-hoc clustering of login-spray attempts identify "same IP"
    without storing raw IP addresses in the audit log. Same IP across
    attempts produces the same hash; different IPs produce different
    hashes. The HMAC key is derived from the JWT secret so no new env
    var is introduced.
    """
    key = _read_secret().encode("utf-8")
    return hmac.new(key, host.encode("utf-8"), sha256).hexdigest()
