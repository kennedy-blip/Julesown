import os
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/health", tags=["health"])

@router.get("")
@router.get("/")
async def health_check():
    """Health check endpoint for Vercel deployment verification."""
    return {
        "status": "ok",
        "service": "jules-clone-api",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@router.get("/env")
async def health_env_check():
    """
    Validates presence of environment variables without leaking secrets.
    """
    keys = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "E2B_API_KEY": bool(os.getenv("E2B_API_KEY")),
        "GITHUB_APP_ID": bool(os.getenv("GITHUB_APP_ID")),
        "GITHUB_APP_PRIVATE_KEY": bool(os.getenv("GITHUB_APP_PRIVATE_KEY")),
        "GITHUB_WEBHOOK_SECRET": bool(os.getenv("GITHUB_WEBHOOK_SECRET")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
    }
    is_healthy = any([keys["GROQ_API_KEY"], keys["ANTHROPIC_API_KEY"], keys["OPENAI_API_KEY"]])
    return {
        "status": "HEALTHY" if is_healthy else "DEGRADED",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "variables": keys
    }
