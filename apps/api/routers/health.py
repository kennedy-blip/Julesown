"""Health and configuration diagnostics. These endpoints never reveal secret
values - only whether something is configured."""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter

from ..config import (
    LLM_PROVIDERS,
    PROVIDER_KEY_VARIABLES,
    PROVIDER_LABELS,
    PROVIDER_MODEL_VARIABLES,
    SERVICE_NAME,
    SERVICE_VERSION,
    env_str,
    get_settings,
)

router = APIRouter(prefix="/api/v1/health", tags=["health"])

REPORTED_VARIABLES = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
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
    variables["GEMINI_API_KEY"] = bool(settings.gemini_api_key)  # GOOGLE_API_KEY counts too
    missing = settings.missing_requirements()
    provider = settings.llm_provider
    identity = settings.commit_identity
    return {
        "status": "HEALTHY" if not missing else "INCOMPLETE",
        "ready": not missing,
        "missing": missing,
        # The same by kind (llm, sandbox, github): what is missing, or null.
        # A model key is the one thing a caller can bring along with a task
        # (see the X-...-Key headers), so the dashboard treats it separately.
        "needs": settings.needs(),
        "variables": variables,
        "llm": {
            "provider": provider,
            "model": settings.llm_model,
            # Every model a task can use, in the order they are tried: with more
            # than one, a task carries on with the next when one reaches its
            # limit or fails. A provider appears once per model it is given.
            "chain": [{"provider": name, "model": model} for name, model in settings.llm_slots],
            "fallback": settings.llm_fallback,
        }
        if provider
        else None,
        # How providers are chosen, whether or not the server has a key for
        # any: the dashboard works out from it what a task will use once the
        # keys kept in the browser are counted in.
        "llm_preferences": {
            "order": settings.llm_preference,
            "first_choice": settings.llm_provider_override,
            "fallback": settings.llm_fallback,
        },
        # What the dashboard needs to offer the providers: their names, the
        # server's default models and whether the server has a key for them.
        "providers": [
            {
                "id": name,
                "label": PROVIDER_LABELS[name],
                "configured": bool(settings.api_key_for(name)),
                "model": settings.model_for(name),
                # Every model the provider is given, in the order they are tried.
                "models": settings.models_for(name),
                "key_variable": PROVIDER_KEY_VARIABLES[name],
                "model_variable": PROVIDER_MODEL_VARIABLES[name],
            }
            for name in LLM_PROVIDERS
        ],
        "commits": {
            "authorship": identity["mode"],
            "author_configured": bool(identity["name"] and identity["email"]),
            "author_name": identity["name"],
        },
        "automation": {
            "ci_autofix": settings.ci_autofix,
            "ci_autofix_attempts": settings.ci_autofix_attempts,
            "keep_unfinished_work": settings.keep_unfinished_work,
            # Whether the server adds instructions of its own to every task (JULES_TASK_GUIDELINES).
            "task_guidelines": bool(settings.task_guidelines),
        },
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
