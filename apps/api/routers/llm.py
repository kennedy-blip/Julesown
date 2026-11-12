"""Model providers: which ones a task can use, and trying a key out."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..config import LLM_PROVIDERS, get_settings
from ..security import provider_keys, require_access, require_json_body
from ..services.llm import probe_provider

router = APIRouter(prefix="/api/v1/llm", tags=["llm"], dependencies=[Depends(require_access)])


class ProviderTestRequest(BaseModel):
    provider: str = Field(..., description="groq, gemini, openrouter, openai or anthropic")
    model: Optional[str] = Field(
        None,
        max_length=400,
        description="Model to try, or several separated by commas. Defaults to the configured ones.",
    )


@router.post("/test", dependencies=[Depends(require_json_body)])
async def test_provider(req: ProviderTestRequest, keys: Dict[str, str] = Depends(provider_keys)) -> Dict[str, Any]:
    """Try a provider out with one small request that contains a tool.

    The key is taken from the request's ``X-<Provider>-Key`` header if it is
    there, else from the server's environment. The answer says whether the key
    was accepted, whether the model exists and whether it called the tool -
    the three things a task needs. With several models each one is tried and
    listed under ``models``. This costs one request of the provider's quota
    per model.
    """
    provider = req.provider.strip().lower()
    settings = get_settings().with_overrides(
        api_keys={provider: keys.get(provider)} if provider in LLM_PROVIDERS else None,
        models={provider: req.model} if req.model else None,
    )
    result = await run_in_threadpool(probe_provider, settings, provider)
    result["key_source"] = (
        "request" if provider in keys else "server" if settings.api_key_for(provider) else None
    )
    return result
