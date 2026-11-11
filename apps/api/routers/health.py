import os
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/health", tags=["health"])

@router.get("")
async def health_check():
    """Health check endpoint to verify backend serverless function is live."""
    return {
        "status": "ok",
        "service": "jules-clone-api",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@router.get("/env")
async def health_env_check():
    """
    Diagnostic endpoint that safely validates the presence of required environment variables
    without leaking secret keys.
    """
    env_vars = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "E2B_API_KEY": bool(os.getenv("E2B_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "GITHUB_TOKEN": bool(os.getenv("GITHUB_TOKEN")),
        "GITHUB_WEBHOOK_SECRET": bool(os.getenv("GITHUB_WEBHOOK_SECRET")),
        "GITHUB_APP_ID": bool(os.getenv("GITHUB_APP_ID")),
        "GITHUB_APP_PRIVATE_KEY": bool(os.getenv("GITHUB_APP_PRIVATE_KEY")),
    }

    # Healthy if at least one LLM key is configured along with sandbox / webhook
    has_llm = (
        env_vars["GROQ_API_KEY"] or
        env_vars["OPENAI_API_KEY"] or
        env_vars["ANTHROPIC_API_KEY"]
    )
    overall_status = "HEALTHY" if has_llm else "CONFIGURED"

    return {
        "status": overall_status,
        "variables": env_vars
    }
