"""FastAPI entrypoint for the Jules API.

Deployed as a Vercel Python function (``vercel.json`` routes ``/api/*`` here,
where it is imported as ``apps.api.main``) and run locally with
``uvicorn apps.api.main:app`` from the repository root.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, List

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from .config import SERVICE_VERSION, env_bool, env_str, load_local_env

load_local_env()

from .routers import github, health, llm, tasks, webhooks  # noqa: E402  (after the environment is loaded)

logger = logging.getLogger("jules.api")


class _JsonFormatter(logging.Formatter):
    """One JSON object per log line: easy to ingest by log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        import json as _json

        payload: Dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status_code", "duration_ms", "task_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return _json.dumps(payload)


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.getLogger("jules").handlers = [_handler]
logging.getLogger("jules").setLevel(logging.INFO)
logging.getLogger("jules").propagate = False


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Every request gets an ``X-Request-ID`` (generated or echoed from the
    client) so that log lines and errors can be traced end to end across the
    API, the sandbox and GitHub."""

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.time()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        duration_ms = int((time.time() - start) * 1000)
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter for task-creating endpoints.

    ``JULES_RATE_LIMIT_REQUESTS`` (default 60) requests per
    ``JULES_RATE_LIMIT_WINDOW_SECONDS`` (default 60) per client IP.
    Only the task-creating endpoints are counted; health and GitHub webhook
    endpoints are exempt.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._calls: Dict[str, List[float]] = {}

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        client = forwarded.split(",")[0].strip() if forwarded else ""
        return client or (request.client.host if request.client else "unknown")

    def _allowed(self, key: str) -> bool:
        # Rate limiting is opt-in for production deployments; local
        # development and tests are unaffected.
        if not env_bool("JULES_RATE_LIMIT_ENABLED", False):
            return True
        now = time.time()
        window = int(env_str("JULES_RATE_LIMIT_WINDOW_SECONDS", "60"))
        limit = int(env_str("JULES_RATE_LIMIT_REQUESTS", "60"))
        calls = [t for t in self._calls.get(key, []) if now - t < window]
        if len(calls) >= limit:
            self._calls[key] = calls
            return False
        calls.append(now)
        self._calls[key] = calls
        return True

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        path = request.url.path
        if path.startswith("/api/v1/tasks") or path.startswith("/api/tasks"):
            key = self._client_key(request)
            if not self._allowed(key):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": {
                            "code": "rate_limited",
                            "message": "Too many task requests. Slow down and try again later.",
                        }
                    },
                )
        return await call_next(request)


app = FastAPI(
    title="Jules Agent Platform API",
    description=(
        "Autonomous coding agent API: runs an LLM tool-calling agent inside an E2B sandbox on a "
        "clone of a GitHub repository and opens a real pull request with the result."
    ),
    version=SERVICE_VERSION,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

# The dashboard calls this API from the same origin (on Vercel, and through the
# Next.js dev proxy locally), so no cross-origin access is needed by default.
# Leaving it off also means a web page open in a developer's browser cannot
# start tasks on a local server. Other browser clients can be allowed
# explicitly with JULES_CORS_ORIGINS (comma-separated origins).
_cors_origins = [
    origin.strip().rstrip("/")
    for origin in (env_str("JULES_CORS_ORIGINS") or "").split(",")
    if origin.strip() and origin.strip() != "*"
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Jules-Key", "X-Request-ID"],
    )

app.add_middleware(RequestIdMiddleware)
app.add_middleware(RateLimitMiddleware)

app.include_router(health.router)
app.include_router(webhooks.router)
app.include_router(github.router)
app.include_router(llm.router)
app.include_router(tasks.router, prefix="/api/v1/tasks")
# Unversioned alias kept for older clients.
app.include_router(tasks.router, prefix="/api/tasks", include_in_schema=False)


@app.get("/api/health", tags=["health"])
async def legacy_health_check():
    """Alias of ``/api/v1/health``."""
    return health.health_payload()


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Jules Autonomous Agent API is live.", "docs": "/api/docs"}


@app.exception_handler(StarletteHTTPException)
async def not_found_aware_handler(request: Request, exc: StarletteHTTPException):
    """Send GitHub App redirects that hit an unknown API path to the dashboard.

    After a GitHub App is installed or authorised, GitHub redirects the browser
    to whatever Setup/Callback URL the App was given. If that URL is not a
    route of this API the visitor would be left on a bare 404, so recognisable
    GitHub redirects are forwarded to the dashboard instead.
    """
    # 405 covers an App whose Setup/Callback URL was pointed at the (POST-only) webhook URL.
    if (
        exc.status_code in (404, 405)
        and request.method == "GET"
        and github.is_github_app_redirect(request.query_params, request.url.path)
    ):
        return RedirectResponse(github.dashboard_landing_url(request.query_params), status_code=307)
    if exc.status_code == 404 and exc.detail == "Not Found":
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Not Found",
                "path": request.url.path,
                "hint": "This API route does not exist. See /api/docs for the available endpoints.",
            },
        )
    return await http_exception_handler(request, exc)


def _readable(error: Any) -> str:
    message = str(error.get("msg") or "invalid value")
    if message.lower().startswith("value error, "):
        message = message[len("value error, ") :]
    location = [str(part) for part in error.get("loc", ()) if part not in ("body", "query", "path")]
    return f"{'.'.join(location)}: {message}" if location else message


@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, exc: RequestValidationError):
    """Turn pydantic's error list into one message the dashboard can show."""
    errors: List[Any] = exc.errors()
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_request",
                "message": "; ".join(_readable(error) for error in errors) or "Invalid request.",
                "errors": [
                    {"loc": list(error.get("loc", ())), "msg": str(error.get("msg")), "type": error.get("type")}
                    for error in errors
                ],
            }
        },
    )
