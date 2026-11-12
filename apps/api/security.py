"""Access control for the endpoints that spend the owner's credentials.

Running a task uses the deployment's LLM key, sandbox quota and GitHub write
access, and its logs can contain repository source code. On an internet-facing
deployment that must not be available to anyone who finds the URL, so:

* ``JULES_ACCESS_KEY`` set       -> callers must send it (``Authorization: Bearer <key>``)
* not set, local development     -> open
* not set, public deployment     -> locked, unless ``JULES_ALLOW_PUBLIC_ACCESS=true``
"""
from __future__ import annotations

import hmac
from typing import Dict, Optional

from fastapi import Header, HTTPException, Request, status

from .config import LLM_PROVIDERS, get_settings

# Headers in which a caller can send its own model-provider keys along with a
# request ("bring your own key"). The dashboard keeps such keys in the browser
# and sends them only with the requests that need them.
PROVIDER_KEY_HEADERS = {name: f"x-{name}-key" for name in LLM_PROVIDERS}
MAX_KEY_LENGTH = 512

LOCKED_MESSAGE = (
    "Task execution is locked on this deployment. Set the JULES_ACCESS_KEY environment variable "
    "(recommended) and redeploy, then enter that key in the dashboard. To let anyone with the URL "
    "run tasks with your credentials instead, set JULES_ALLOW_PUBLIC_ACCESS=true."
)


def _presented_key(authorization: Optional[str], header_key: Optional[str]) -> str:
    if header_key:
        return header_key.strip()
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return ""


def require_access(
    authorization: Optional[str] = Header(None),
    x_jules_key: Optional[str] = Header(None),
) -> None:
    settings = get_settings()
    mode = settings.access_mode
    if mode == "open":
        return
    if mode == "locked":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "access_locked", "message": LOCKED_MESSAGE},
        )
    presented = _presented_key(authorization, x_jules_key)
    expected = settings.access_key or ""
    if not presented or not hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "access_key_required",
                "message": "This deployment requires an access key. Enter the JULES_ACCESS_KEY value.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_json_body(content_type: Optional[str] = Header(None)) -> None:
    """Tasks are only accepted as JSON.

    A web page cannot make a visitor's browser send ``Content-Type:
    application/json`` to another origin without a CORS pre-flight, and this
    API refuses that pre-flight unless the origin is listed in
    ``JULES_CORS_ORIGINS``. Checking it here - instead of relying on the
    framework's body parsing, which differs between FastAPI versions - keeps
    other websites from starting tasks on a server that runs without an access
    key, such as a local development server.
    """
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    is_json = media_type == "application/json" or (
        media_type.startswith("application/") and media_type.endswith("+json")
    )
    if not is_json:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "unsupported_media_type",
                "message": "Send the task as JSON with the header Content-Type: application/json.",
            },
        )


def provider_keys(request: Request) -> Dict[str, str]:
    """Model-provider API keys sent with this request, by provider.

    They are used for this one request, take precedence over the server's own
    keys, and are neither stored nor logged. A value that cannot be a key
    (blank, absurdly long, with spaces or control characters) is ignored.
    """
    found: Dict[str, str] = {}
    for provider, header in PROVIDER_KEY_HEADERS.items():
        value = (request.headers.get(header) or "").strip()
        if value and len(value) <= MAX_KEY_LENGTH and value.isprintable() and " " not in value:
            found[provider] = value
    return found
