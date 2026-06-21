"""JWT issue / decode / discriminated result tests."""

from __future__ import annotations

import time

import pytest
from jose import jwt as _jose_jwt

from claritymed.web.jwt import (
    ENV_JWT_SECRET,
    JWT_ALGORITHM,
    Invalid,
    Tamper,
    Valid,
    create_token,
    decode_token,
    hmac_ip,
    validate_secret_or_raise,
)


def test_create_decode_round_trip(jwt_secret):  # noqa: ARG001
    token = create_token("user-1", "en")
    result = decode_token(token)
    assert isinstance(result, Valid)
    assert result.claims["sub"] == "user-1"
    assert result.claims["lang"] == "en"
    assert "exp" in result.claims


def test_create_token_rejects_disallowed_lang_at_issue_time(jwt_secret):  # noqa: ARG001
    with pytest.raises(ValueError, match="ja"):
        create_token("user-1", "ja")


def test_decode_tamper_lang_outside_allowlist(jwt_secret):
    # Manually forge a token with a valid HMAC but a disallowed lang.
    # Skip ``create_token`` (which would refuse) and use jose directly.
    bad = _jose_jwt.encode(
        {"sub": "user-1", "lang": "ja", "iat": 0, "exp": int(time.time()) + 60},
        jwt_secret,
        algorithm=JWT_ALGORITHM,
    )
    result = decode_token(bad)
    assert isinstance(result, Tamper)
    assert result.reason == "invalid_lang_claim"


def test_decode_tamper_missing_sub(jwt_secret):
    bad = _jose_jwt.encode(
        {"lang": "en", "iat": 0, "exp": int(time.time()) + 60},
        jwt_secret,
        algorithm=JWT_ALGORITHM,
    )
    result = decode_token(bad)
    assert isinstance(result, Tamper)
    assert result.reason == "missing_sub"


def test_decode_tamper_missing_lang(jwt_secret):
    bad = _jose_jwt.encode(
        {"sub": "user-1", "iat": 0, "exp": int(time.time()) + 60},
        jwt_secret,
        algorithm=JWT_ALGORITHM,
    )
    result = decode_token(bad)
    assert isinstance(result, Tamper)
    assert result.reason == "missing_lang_claim"


def test_decode_invalid_signature(jwt_secret):  # noqa: ARG001
    token = create_token("user-1", "en")
    tampered = token[:-4] + "AAAA"
    result = decode_token(tampered)
    assert isinstance(result, Invalid)


def test_decode_invalid_expired(jwt_secret):
    expired = _jose_jwt.encode(
        {"sub": "user-1", "lang": "en", "iat": 0, "exp": 1},
        jwt_secret,
        algorithm=JWT_ALGORITHM,
    )
    result = decode_token(expired)
    assert isinstance(result, Invalid)
    assert result.reason == "expired"


def test_decode_invalid_malformed(jwt_secret):  # noqa: ARG001
    result = decode_token("not-a-jwt")
    assert isinstance(result, Invalid)


def test_validate_secret_or_raise_passes(monkeypatch):
    monkeypatch.setenv(ENV_JWT_SECRET, "x" * 32)
    assert validate_secret_or_raise() == "x" * 32


@pytest.mark.parametrize("bad", ["", "changeme", "secret"])
def test_validate_secret_or_raise_rejects_placeholders(monkeypatch, bad: str):
    monkeypatch.setenv(ENV_JWT_SECRET, bad)
    with pytest.raises(RuntimeError, match=ENV_JWT_SECRET):
        validate_secret_or_raise()


def test_validate_secret_or_raise_rejects_unset(monkeypatch):
    monkeypatch.delenv(ENV_JWT_SECRET, raising=False)
    with pytest.raises(RuntimeError, match=ENV_JWT_SECRET):
        validate_secret_or_raise()


def test_hmac_ip_deterministic(jwt_secret):  # noqa: ARG001
    a = hmac_ip("1.2.3.4")
    b = hmac_ip("1.2.3.4")
    assert a == b
    assert len(a) == 64  # SHA-256 hex


def test_hmac_ip_different_inputs_differ(jwt_secret):  # noqa: ARG001
    assert hmac_ip("1.2.3.4") != hmac_ip("5.6.7.8")
