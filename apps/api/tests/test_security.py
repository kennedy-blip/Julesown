"""Access control: JULES_ACCESS_KEY, locked mode, public access, provider keys."""
from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from apps.api.security import (
    MAX_KEY_LENGTH,
    LOCKED_MESSAGE,
    provider_keys,
    require_access,
    require_json_body,
)


# --------------------------------------------------------------------------- #
# require_access
# --------------------------------------------------------------------------- #


def test_require_access_open_mode(monkeypatch):
    monkeypatch.delenv("JULES_ACCESS_KEY", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("JULES_ALLOW_PUBLIC_ACCESS", raising=False)
    require_access(authorization=None, x_jules_key=None)
    require_access(authorization="Bearer anything", x_jules_key=None)


def test_require_access_locked_without_key(monkeypatch):
    monkeypatch.delenv("JULES_ACCESS_KEY", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("JULES_ALLOW_PUBLIC_ACCESS", raising=False)
    with pytest.raises(HTTPException) as caught:
        require_access(authorization=None, x_jules_key=None)
    assert caught.value.status_code == status.HTTP_403_FORBIDDEN
    assert caught.value.detail["code"] == "access_locked"


def test_require_access_locked_with_right_key(monkeypatch):
    monkeypatch.setenv("JULES_ACCESS_KEY", "secret-123")
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("JULES_ALLOW_PUBLIC_ACCESS", raising=False)
    require_access(authorization="Bearer secret-123", x_jules_key=None)
    require_access(authorization="Bearer secret-123", x_jules_key="secret-123")


def test_require_access_locked_with_wrong_key(monkeypatch):
    monkeypatch.setenv("JULES_ACCESS_KEY", "secret-123")
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("JULES_ALLOW_PUBLIC_ACCESS", raising=False)
    with pytest.raises(HTTPException) as caught:
        require_access(authorization="Bearer wrong", x_jules_key=None)
    assert caught.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert caught.value.detail["code"] == "access_key_required"

    with pytest.raises(HTTPException) as caught:
        require_access(authorization=None, x_jules_key=None)
    assert caught.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_require_access_constant_time_compare(monkeypatch):
    """A wrong key must not short-circuit early: use hmac.compare_digest."""
    monkeypatch.setenv("JULES_ACCESS_KEY", "a" * 32)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("JULES_ALLOW_PUBLIC_ACCESS", raising=False)
    with pytest.raises(HTTPException) as caught:
        require_access(authorization="Bearer " + "b" * 32, x_jules_key=None)
    assert caught.value.status_code == status.HTTP_401_UNAUTHORIZED


# --------------------------------------------------------------------------- #
# require_json_body
# --------------------------------------------------------------------------- #


def test_require_json_body_accepts_json():
    require_json_body("application/json")
    require_json_body("application/json; charset=utf-8")
    require_json_body("application/vnd.api+json")


def test_require_json_body_rejects_other():
    with pytest.raises(HTTPException) as caught:
        require_json_body("text/plain")
    assert caught.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert caught.value.detail["code"] == "unsupported_media_type"

    with pytest.raises(HTTPException) as caught:
        require_json_body(None)
    assert caught.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


# --------------------------------------------------------------------------- #
# provider_keys
# --------------------------------------------------------------------------- #


def _request_with_headers(headers: dict):
    class FakeRequest:
        def __init__(self, headers: dict) -> None:
            self.headers = headers

    return FakeRequest(headers)


def test_provider_keys_parses_all_providers():
    headers = {
        "x-groq-key": "gsk_abc",
        "x-gemini-key": "gemini_key",
        "x-nvidia-key": "nvapi_xyz",
        "x-zai-key": "zai_key",
        "x-cloudflare-key": "account_id:token",
        "x-sambanova-key": "sn_key",
        "x-openrouter-key": "or_key",
        "x-pollinations-key": "sk_poll",
        "x-deepseek-key": "ds_key",
        "x-openai-key": "sk_openai",
        "x-anthropic-key": "sk_ant",
    }
    found = provider_keys(_request_with_headers(headers))
    assert len(found) == 11
    assert found["groq"] == "gsk_abc"
    assert found["cloudflare"] == "account_id:token"


def test_provider_keys_ignores_blank_and_invalid():
    headers = {
        "x-groq-key": "",
        "x-gemini-key": "   ",
        "x-openai-key": "sk_valid",
        "x-anthropic-key": "key with spaces",
        "x-zai-key": "x" * (MAX_KEY_LENGTH + 1),
    }
    found = provider_keys(_request_with_headers(headers))
    assert found == {"openai": "sk_valid"}


# --------------------------------------------------------------------------- #
# End-to-end: access key through the API
# --------------------------------------------------------------------------- #


def test_api_requires_access_key_when_locked(client, monkeypatch):
    monkeypatch.setenv("JULES_ACCESS_KEY", "locked-key")
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("JULES_ALLOW_PUBLIC_ACCESS", raising=False)
    response = client.post(
        "/api/v1/tasks/stream",
        json={"repository": "acme/demo", "prompt": "hello"},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "access_key_required"

    response = client.post(
        "/api/v1/tasks/stream",
        json={"repository": "acme/demo", "prompt": "hello"},
        headers={"Authorization": "Bearer locked-key"},
    )
    assert response.status_code != 401, response.text


def test_api_open_without_key(client, monkeypatch):
    monkeypatch.delenv("JULES_ACCESS_KEY", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("JULES_ALLOW_PUBLIC_ACCESS", raising=False)
    response = client.post(
        "/api/v1/tasks/stream",
        json={"repository": "acme/demo", "prompt": "hello"},
    )
    # Not 401: open mode. It may fail later (missing config), but not for access.
    assert response.status_code != 401, response.text


def test_locked_message_mentions_access_key():
    assert "JULES_ACCESS_KEY" in LOCKED_MESSAGE
