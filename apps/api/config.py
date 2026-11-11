"""Runtime configuration for the Jules API.

Everything is read from environment variables *on demand* (``get_settings()``
builds a fresh snapshot every call) so that tests can monkeypatch the
environment and so that nothing secret is captured at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Model defaults
#
# Hosted model catalogues change often: Groq retired ``llama-3.3-70b-versatile``
# (the model this project used to hardcode) on 2026-08-16. Every default below
# can be overridden with an environment variable, and the LLM client falls back
# to the provider's ``/models`` listing when the configured model is gone.
# --------------------------------------------------------------------------- #
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL_PREFERENCES = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
)
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
OPENAI_MODEL_PREFERENCES = ("gpt-6-luna", "gpt-6.1-terra")
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
ANTHROPIC_MODEL_PREFERENCES = ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5")

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_GITHUB_API_URL = "https://api.github.com"

SERVICE_NAME = "jules-clone-api"
SERVICE_VERSION = "2.0.0"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def load_local_env() -> None:
    """Load ``.env`` / ``.env.local`` from the repository root for local runs.

    ``python-dotenv`` has always been a dependency but nothing ever called it,
    so the documented ``cp .env.example .env`` step had no effect. Vercel
    injects its own environment, so this is skipped there. Existing variables
    are never overridden.
    """
    if os.getenv("VERCEL") or os.getenv("JULES_SKIP_DOTENV"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency is in requirements.txt
        return
    root = Path(__file__).resolve().parents[2]
    for name in (".env", ".env.local"):
        candidate = root / name
        if candidate.is_file():
            load_dotenv(candidate, override=False)


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a trimmed environment value, treating blanks and the
    ``your_..._here`` placeholders from ``.env.example`` as unset."""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    lowered = value.lower()
    if lowered.startswith("your_") and lowered.endswith("_here"):
        return default
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = env_str(name)
    if value is None:
        return default
    return value.lower() in _TRUE_VALUES


def env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = env_str(name)
    if value is None:
        return default
    try:
        parsed = int(float(value))
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


@dataclass(frozen=True)
class Settings:
    environment: str
    on_vercel: bool

    # LLM providers
    llm_provider_override: Optional[str]
    groq_api_key: Optional[str]
    groq_model: str
    groq_base_url: str
    openai_api_key: Optional[str]
    openai_model: str
    openai_base_url: str
    anthropic_api_key: Optional[str]
    anthropic_model: str
    anthropic_base_url: str

    # Sandbox
    e2b_api_key: Optional[str]
    e2b_template: Optional[str]

    # GitHub
    github_token: Optional[str]
    github_app_id: Optional[str]
    github_app_private_key: Optional[str]
    github_api_url: str
    github_webhook_secret: Optional[str]

    # Access control
    access_key: Optional[str]
    allow_public_access: bool

    # Limits
    task_budget_seconds: int
    max_agent_turns: int
    command_timeout_seconds: int
    max_output_tokens: int

    # ------------------------------------------------------------------ #
    # Derived state
    # ------------------------------------------------------------------ #
    @property
    def llm_provider(self) -> Optional[str]:
        """The provider that will be used: explicit override, else the first
        configured one in the order Groq -> OpenAI -> Anthropic."""
        available = {
            "groq": bool(self.groq_api_key),
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
        }
        if self.llm_provider_override:
            choice = self.llm_provider_override.lower()
            return choice if available.get(choice) else None
        for name in ("groq", "openai", "anthropic"):
            if available[name]:
                return name
        return None

    @property
    def llm_model(self) -> Optional[str]:
        return {
            "groq": self.groq_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
        }.get(self.llm_provider or "")

    @property
    def github_app_configured(self) -> bool:
        return bool(self.github_app_id and self.github_app_private_key)

    @property
    def github_auth_mode(self) -> str:
        """``app`` (GitHub App installation tokens), ``token`` (GITHUB_TOKEN)
        or ``none``. An App is preferred when both are configured."""
        if self.github_app_configured:
            return "app"
        if self.github_token:
            return "token"
        return "none"

    @property
    def is_public_deployment(self) -> bool:
        return self.on_vercel or self.environment == "production"

    @property
    def access_mode(self) -> str:
        """``key``: callers must present JULES_ACCESS_KEY.
        ``open``: no key needed (local development, or explicitly opted in).
        ``locked``: an internet-facing deployment without a key - task
        execution is refused, because it would let any visitor spend the
        owner's credentials and open pull requests on their repositories."""
        if self.access_key:
            return "key"
        if self.allow_public_access or not self.is_public_deployment:
            return "open"
        return "locked"

    def missing_requirements(self) -> List[str]:
        """Human-readable list of what is still needed to run a task."""
        missing: List[str] = []
        if not self.llm_provider:
            if self.llm_provider_override:
                missing.append(
                    f"an API key for the selected LLM provider (JULES_LLM_PROVIDER={self.llm_provider_override})"
                )
            else:
                missing.append("an LLM API key (GROQ_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        if not self.e2b_api_key:
            missing.append("E2B_API_KEY (sandbox used to clone, edit and test the repository)")
        if self.github_auth_mode == "none":
            missing.append(
                "GitHub write credentials (GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY, or GITHUB_TOKEN)"
            )
        return missing


def get_settings() -> Settings:
    on_vercel = bool(os.getenv("VERCEL"))
    environment = (
        env_str("ENVIRONMENT") or env_str("VERCEL_ENV") or "development"
    ).lower()
    return Settings(
        environment=environment,
        on_vercel=on_vercel,
        llm_provider_override=env_str("JULES_LLM_PROVIDER"),
        groq_api_key=env_str("GROQ_API_KEY"),
        groq_model=env_str("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL,
        groq_base_url=(env_str("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL) or DEFAULT_GROQ_BASE_URL).rstrip("/"),
        openai_api_key=env_str("OPENAI_API_KEY"),
        openai_model=env_str("OPENAI_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL,
        openai_base_url=(env_str("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL) or DEFAULT_OPENAI_BASE_URL).rstrip("/"),
        anthropic_api_key=env_str("ANTHROPIC_API_KEY"),
        anthropic_model=env_str("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL) or DEFAULT_ANTHROPIC_MODEL,
        anthropic_base_url=(
            env_str("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL) or DEFAULT_ANTHROPIC_BASE_URL
        ).rstrip("/"),
        e2b_api_key=env_str("E2B_API_KEY"),
        e2b_template=env_str("E2B_TEMPLATE"),
        github_token=env_str("GITHUB_TOKEN"),
        github_app_id=env_str("GITHUB_APP_ID"),
        github_app_private_key=env_str("GITHUB_APP_PRIVATE_KEY"),
        github_api_url=(env_str("GITHUB_API_URL", DEFAULT_GITHUB_API_URL) or DEFAULT_GITHUB_API_URL).rstrip("/"),
        github_webhook_secret=env_str("GITHUB_WEBHOOK_SECRET"),
        access_key=env_str("JULES_ACCESS_KEY"),
        allow_public_access=env_bool("JULES_ALLOW_PUBLIC_ACCESS"),
        task_budget_seconds=env_int("JULES_TASK_BUDGET_SECONDS", 270, minimum=30, maximum=1700),
        max_agent_turns=env_int("JULES_MAX_AGENT_TURNS", 30, minimum=1, maximum=200),
        command_timeout_seconds=env_int("JULES_COMMAND_TIMEOUT_SECONDS", 120, minimum=5, maximum=900),
        max_output_tokens=env_int("JULES_MAX_OUTPUT_TOKENS", 8192, minimum=256, maximum=64000),
    )


def platform_seconds_remaining() -> Optional[float]:
    """Seconds left before the hosting platform kills this invocation.

    On Vercel the Python runtime exposes the invocation deadline (max duration)
    through ``vercel_runtime.get_deadline()``. Returns ``None`` when unknown,
    e.g. when running under plain uvicorn.
    """
    try:
        from vercel_runtime import get_deadline  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        deadline = get_deadline()
    except Exception:
        return None
    if deadline is None:
        return None
    try:
        return (deadline - datetime.now(timezone.utc)).total_seconds()
    except Exception:
        return None
