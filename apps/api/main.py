"""FastAPI entrypoint for the Jules API.

Deployed as a Vercel Python function (``vercel.json`` routes ``/api/*`` here,
where it is imported as ``apps.api.main``) and run locally with
``uvicorn apps.api.main:app`` from the repository root.
"""
from __future__ import annotations

from typing import Any, List

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import SERVICE_VERSION, env_str, load_local_env

load_local_env()

from .routers import github, health, llm, tasks, webhooks  # noqa: E402  (after the environment is loaded)

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
        allow_headers=["Authorization", "Content-Type", "X-Jules-Key"],
    )

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
