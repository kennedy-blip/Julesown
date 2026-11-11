"""Health and configuration diagnostics. These endpoints never reveal secret
values - only whether something is configured."""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter

from ..config import SERVICE_NAME, SERVICE_VERSION, env_str, get_settings

router = APIRouter(prefix="/api/v1/health", tags=["health"])

REPORTED_VARIABLES = (
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "E2B_API_KEY",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_TOKEN",
    "GITHUB_WEBHOOK_SECRET",
    "JULES_ACCESS_KEY",
)


def health_payload() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "environment": get_settings().environment,
    }


@router.get("")
async def health_check():
    """Liveness check: the backend function is deployed and answering."""
    return health_payload()


@router.get("/env")
async def health_env_check():
    """Readiness check: what is configured and what is still missing to run a task.

    ``status`` is ``HEALTHY`` only when an LLM key, the sandbox key and GitHub
    write credentials are all present. This checks presence, not validity -
    wrong credentials surface as a clear error when a task runs.
    """
    settings = get_settings()
    variables = {name: bool(env_str(name)) for name in REPORTED_VARIABLES}
    missing = settings.missing_requirements()
    provider = settings.llm_provider
    return {
        "status": "HEALTHY" if not missing else "INCOMPLETE",
        "ready": not missing,
        "missing": missing,
        "variables": variables,
        "llm": {"provider": provider, "model": settings.llm_model} if provider else None,
        "github": {"auth_mode": settings.github_auth_mode},
        "sandbox": {"configured": bool(settings.e2b_api_key), "provider": "e2b"},
        "webhooks": {"secret_configured": bool(settings.github_webhook_secret)},
        "access": {"mode": settings.access_mode},
        "limits": {
            "task_budget_seconds": settings.task_budget_seconds,
            "max_agent_turns": settings.max_agent_turns,
        },
        "environment": settings.environment,
        "region": os.getenv("VERCEL_REGION"),
    }
