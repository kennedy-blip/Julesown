"""Runtime configuration for the Jules API.

Everything is read from environment variables *on demand* (``get_settings()``
builds a fresh snapshot every call) so that tests can monkeypatch the
environment and so that nothing secret is captured at import time.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
# Google's alias for its newest Flash model: it moves with the catalogue, so it
# does not go stale the way a pinned version does.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
GEMINI_MODEL_PREFERENCES = ("gemini-flash-latest", "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite")
# OpenRouter's free-models router: for each request it picks a free model that
# is available and can do what the request needs (tool calls). A single free
# model is throttled upstream whenever it is popular, and free "preview" models
# come and go; the router has neither problem. A specific model is a setting
# away (OPENROUTER_MODEL, or the model field in the dashboard's settings).
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
# NVIDIA's hosted catalogue (build.nvidia.com, "NIM"): open models behind one
# key, free within a requests-per-minute limit and without a daily one. Models
# are added and withdrawn every few weeks, so more names are known than are
# used: when a configured one is gone, the first of these that NVIDIA still
# lists takes its place. They are models NVIDIA offers on a free endpoint and
# describes as made for agents, coding and tool calls; NVIDIA's own come
# early, because those stay in its catalogue the longest.
DEFAULT_NVIDIA_MODEL = "deepseek-ai/deepseek-v4.1-flash"
NVIDIA_MODEL_PREFERENCES = (
    "deepseek-ai/deepseek-v4.1-flash",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "minimaxai/minimax-m3.1-flash",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "moonshotai/kimi-k2.6",
    "openai/gpt-oss-20b",
)
# Z.AI's GLM models. The Flash models named here cost nothing with a standard
# API key (one request at a time); the larger ones (glm-5.3, glm-5.3-flash)
# are paid from the account's balance and are a setting away (ZAI_MODEL).
DEFAULT_ZAI_MODEL = "glm-4.7-flash"

# A provider can be given several models (GROQ_MODEL="a,b"): the first one is
# used, and when it is rate limited, overloaded or gone the next one takes
# over - limits are counted per model, so a second model is a second allowance.
# These are the lists used when nothing is configured. Where a provider has a
# name that moves with its catalogue, that is used: Gemini's first entry is
# Google's alias for its current Flash model, followed by the lighter model
# that has capacity when the alias is busy; OpenRouter's is its router, which
# is a pool already. NVIDIA and Z.AI have no such name: a model of theirs that
# is gone is replaced (NVIDIA) or passed over (Z.AI) - see services/llm.py.
DEFAULT_MODELS: Dict[str, Tuple[str, ...]] = {
    "groq": (DEFAULT_GROQ_MODEL, "openai/gpt-oss-20b"),
    "gemini": (DEFAULT_GEMINI_MODEL, "gemini-3.5-flash-lite"),
    "nvidia": (DEFAULT_NVIDIA_MODEL, "nvidia/nemotron-3-super-120b-a12b"),
    "zai": (DEFAULT_ZAI_MODEL, "glm-4.5-flash"),
    "openrouter": (DEFAULT_OPENROUTER_MODEL,),
    "openai": (DEFAULT_OPENAI_MODEL,),
    "anthropic": (DEFAULT_ANTHROPIC_MODEL,),
}
MAX_MODELS_PER_PROVIDER = 4
# Standing instructions (JULES_TASK_GUIDELINES, or sent along with a task): how long they may be.
MAX_GUIDELINES_CHARS = 4000

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Z.AI's standard API. (Its "coding" endpoint belongs to the GLM Coding Plan,
# which Z.AI restricts to the coding tools it lists; this is not one of them.)
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
DEFAULT_GITHUB_API_URL = "https://api.github.com"

# Every provider the agent can use, in the order they are tried when nothing
# else is said: the free tiers first, the pay-per-token ones after them. Among
# the free ones, those with a model of their own choosing come before
# OpenRouter's router, which answers with whichever free model it picks and has
# the smallest daily allowance.
LLM_PROVIDERS: Tuple[str, ...] = ("groq", "gemini", "nvidia", "zai", "openrouter", "openai", "anthropic")
PROVIDER_LABELS = {
    "groq": "Groq",
    "gemini": "Google Gemini",
    "nvidia": "NVIDIA NIM",
    "zai": "Z.AI (GLM)",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}
PROVIDER_KEY_VARIABLES = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "zai": "ZAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
PROVIDER_MODEL_VARIABLES = {
    "groq": "GROQ_MODEL",
    "gemini": "GEMINI_MODEL",
    "nvidia": "NVIDIA_MODEL",
    "zai": "ZAI_MODEL",
    "openrouter": "OPENROUTER_MODEL",
    "openai": "OPENAI_MODEL",
    "anthropic": "ANTHROPIC_MODEL",
}

# Who a commit made by the agent is attributed to.
#   bot      - the GitHub App (or the owner of GITHUB_TOKEN): GitHub's default
#   coauthor - the same, plus a "Co-authored-by" line naming a person
#   user     - a person is the author; the App is only the committer
COMMIT_AUTHORSHIP_MODES: Tuple[str, ...] = ("bot", "coauthor", "user")
_EMAIL_RE = re.compile(r"^[^\s<>@,;]+@[^\s<>@,;]+\.[^\s<>@,;]+$")
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")

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


def clean_person_name(value: Any) -> Optional[str]:
    """A name usable in a commit's author field, or None."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.replace("<", " ").replace(">", " ").split())
    return cleaned[:100] or None


def clean_email(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if len(cleaned) > 200 or not _EMAIL_RE.match(cleaned):
        return None
    return cleaned


def clean_model_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if _MODEL_NAME_RE.match(cleaned) else None


def split_models(value: Any) -> List[str]:
    """The model names in a setting such as ``"a, b"`` (or a list): valid ones
    only, without repeats, in the order given, at most MAX_MODELS_PER_PROVIDER."""
    if isinstance(value, str):
        value = re.split(r"[,;\n]+", value)
    models: List[str] = []
    for item in value or []:
        name = clean_model_name(item)
        if name and name not in models:
            models.append(name)
    return models[:MAX_MODELS_PER_PROVIDER]


def default_models(provider: str) -> str:
    """The models of a provider when none are configured, as one setting."""
    return ",".join(DEFAULT_MODELS.get(provider, ()))


def clean_provider_order(values: Any) -> List[str]:
    """Provider names from a list or a comma-separated string: known ones only,
    lower-case, without repeats, in the order given."""
    if isinstance(values, str):
        values = values.replace(";", ",").split(",")
    order: List[str] = []
    for value in values or []:
        name = str(value).strip().lower()
        if name in LLM_PROVIDERS and name not in order:
            order.append(name)
    return order


@dataclass(frozen=True)
class Settings:
    environment: str
    on_vercel: bool

    # LLM providers. A ``*_model`` setting holds one model or several,
    # separated by commas, in the order they are to be tried (see models_for).
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

    # More LLM providers, and how they work together
    gemini_api_key: Optional[str] = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_base_url: str = DEFAULT_GEMINI_BASE_URL
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    nvidia_api_key: Optional[str] = None
    nvidia_model: str = ",".join(DEFAULT_MODELS["nvidia"])
    nvidia_base_url: str = DEFAULT_NVIDIA_BASE_URL
    zai_api_key: Optional[str] = None
    zai_model: str = ",".join(DEFAULT_MODELS["zai"])
    zai_base_url: str = DEFAULT_ZAI_BASE_URL
    # Instructions added to every task of this installation (conventions of the
    # owner's projects: the stack, what a deployment needs, what to leave alone).
    task_guidelines: str = ""
    # When the provider in use hits a limit or fails, carry on with the next one.
    llm_fallback: bool = True
    # Order in which the configured providers are tried (after the override).
    llm_order: Tuple[str, ...] = ()

    # Commit attribution
    commit_authorship: str = "bot"
    commit_author_name: Optional[str] = None
    commit_author_email: Optional[str] = None

    # Fixing failed checks on the agent's own pull requests when GitHub reports them
    ci_autofix: bool = True
    ci_autofix_attempts: int = 2
    # When the agent is cut short (time, model provider) after changing files,
    # deliver what it has - marked as unfinished - instead of discarding it.
    keep_unfinished_work: bool = True

    # ------------------------------------------------------------------ #
    # Derived state
    # ------------------------------------------------------------------ #
    def api_key_for(self, provider: str) -> Optional[str]:
        return getattr(self, f"{provider}_api_key", None) if provider in LLM_PROVIDERS else None

    def models_for(self, provider: str) -> List[str]:
        """The models configured for a provider, in the order they are tried."""
        if provider not in LLM_PROVIDERS:
            return []
        return split_models(getattr(self, f"{provider}_model", "")) or list(DEFAULT_MODELS[provider])

    def model_for(self, provider: str) -> Optional[str]:
        """The model a provider is used with first."""
        models = self.models_for(provider)
        return models[0] if models else None

    @property
    def llm_preference(self) -> List[str]:
        """Every provider, in the order they are preferred (``JULES_LLM_ORDER``
        first, the rest in their usual order)."""
        order = [name for name in self.llm_order if name in LLM_PROVIDERS]
        return order + [name for name in LLM_PROVIDERS if name not in order]

    @property
    def configured_llm_providers(self) -> List[str]:
        """Providers that have an API key, in the order they are preferred."""
        return [name for name in self.llm_preference if self.api_key_for(name)]

    @property
    def llm_chain(self) -> List[str]:
        """The providers a task uses, first choice first.

        ``JULES_LLM_PROVIDER`` names the first choice and must have a key.
        The others follow as fallbacks unless ``JULES_LLM_FALLBACK`` is off.
        """
        configured = self.configured_llm_providers
        if self.llm_provider_override:
            choice = self.llm_provider_override.lower()
            if choice not in configured:
                return []
            configured = [choice] + [name for name in configured if name != choice]
        return configured if self.llm_fallback else configured[:1]

    @property
    def llm_slots(self) -> List[Tuple[str, str]]:
        """Every model a task can use, as ``(provider, model)``, in the order
        they are tried when the one in use is rate limited, overloaded or fails.

        Each provider of the chain with its first model, then each provider's
        second model, and so on: every provider's best model is tried before
        anybody's stand-in. With fallback switched off there is exactly one.
        """
        chain = self.llm_chain
        if not chain:
            return []
        if not self.llm_fallback:
            return [(chain[0], self.model_for(chain[0]) or "")]
        lists = [(name, self.models_for(name)) for name in chain]
        depth = max(len(models) for _name, models in lists)
        return [(name, models[level]) for level in range(depth) for name, models in lists if level < len(models)]

    @property
    def llm_provider(self) -> Optional[str]:
        """The provider a task starts with: the explicit override, else the
        first configured one in the order of ``llm_preference``."""
        chain = self.llm_chain
        return chain[0] if chain else None

    @property
    def llm_model(self) -> Optional[str]:
        return self.model_for(self.llm_provider or "")

    @property
    def commit_identity(self) -> Dict[str, Optional[str]]:
        """How commits are attributed. ``mode`` falls back to ``bot`` when a
        person is asked for but not named."""
        mode = self.commit_authorship if self.commit_authorship in COMMIT_AUTHORSHIP_MODES else "bot"
        return {"mode": mode, "name": self.commit_author_name, "email": self.commit_author_email}

    def secret_values(self) -> List[str]:
        """Every secret this configuration holds, for redacting logs."""
        values = [self.api_key_for(name) for name in LLM_PROVIDERS]
        values += [self.github_token, self.e2b_api_key, self.access_key, self.github_webhook_secret]
        return [value for value in values if value]

    def with_overrides(
        self,
        *,
        api_keys: Optional[Mapping[str, Optional[str]]] = None,
        models: Optional[Mapping[str, Optional[str]]] = None,
        primary: Optional[str] = None,
        fallback: Optional[bool] = None,
        order: Optional[List[str]] = None,
        commit: Optional[Mapping[str, Any]] = None,
    ) -> "Settings":
        """This configuration with the choices of one request applied.

        Used for what the dashboard sends along with a task: API keys kept in
        the browser, the preferred provider, the commit identity. Unknown or
        malformed values are ignored rather than trusted.
        """
        changes: Dict[str, Any] = {}
        brings_key = False
        for provider, key in (api_keys or {}).items():
            if provider in LLM_PROVIDERS and isinstance(key, str) and key.strip():
                changes[f"{provider}_api_key"] = key.strip()
                brings_key = True
        for provider, model in (models or {}).items():
            # One model or several ("a, b"): this request's list replaces the server's.
            cleaned = split_models(model)
            if provider in LLM_PROVIDERS and cleaned:
                changes[f"{provider}_model"] = ",".join(cleaned)
        cleaned_order = clean_provider_order(order)
        if cleaned_order:
            changes["llm_order"] = tuple(cleaned_order)
        if primary is not None:
            choice = str(primary).strip().lower()
            if choice in LLM_PROVIDERS:
                # A preference, not a requirement: without a key for that
                # provider the order stays as it was.
                if changes.get(f"{choice}_api_key") or self.api_key_for(choice):
                    changes["llm_provider_override"] = choice
            elif choice in ("", "auto"):
                changes["llm_provider_override"] = None
        if fallback is not None:
            changes["llm_fallback"] = bool(fallback)
        if commit:
            mode = str(commit.get("mode") or "").strip().lower()
            if mode in COMMIT_AUTHORSHIP_MODES:
                changes["commit_authorship"] = mode
            name, email = clean_person_name(commit.get("name")), clean_email(commit.get("email"))
            if name and email:
                changes["commit_author_name"], changes["commit_author_email"] = name, email
        if not changes:
            return self
        result = replace(self, **changes)
        if brings_key and not result.llm_chain:
            # The server names a first choice it has no key for (JULES_LLM_PROVIDER),
            # but this request brought a key of its own: that one serves it.
            result = replace(result, llm_provider_override=None)
        return result

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

    def needs(self) -> Dict[str, Optional[str]]:
        """What is still needed to run a task, by kind: ``llm``, ``sandbox``
        and ``github`` each map to a description of what is missing, or None."""
        llm: Optional[str] = None
        if not self.llm_provider:
            if self.llm_provider_override:
                llm = f"an API key for the selected LLM provider (JULES_LLM_PROVIDER={self.llm_provider_override})"
            else:
                names = [PROVIDER_KEY_VARIABLES[name] for name in LLM_PROVIDERS]
                llm = f"an LLM API key ({', '.join(names[:-1])} or {names[-1]})"
        return {
            "llm": llm,
            "sandbox": None
            if self.e2b_api_key
            else "E2B_API_KEY (sandbox used to clone, edit and test the repository)",
            "github": "GitHub write credentials (GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY, or GITHUB_TOKEN)"
            if self.github_auth_mode == "none"
            else None,
        }

    def missing_requirements(self) -> List[str]:
        """Human-readable list of what is still needed to run a task."""
        return [message for message in self.needs().values() if message]


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
        groq_model=env_str("GROQ_MODEL") or default_models("groq"),
        groq_base_url=(env_str("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL) or DEFAULT_GROQ_BASE_URL).rstrip("/"),
        openai_api_key=env_str("OPENAI_API_KEY"),
        openai_model=env_str("OPENAI_MODEL") or default_models("openai"),
        openai_base_url=(env_str("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL) or DEFAULT_OPENAI_BASE_URL).rstrip("/"),
        anthropic_api_key=env_str("ANTHROPIC_API_KEY"),
        anthropic_model=env_str("ANTHROPIC_MODEL") or default_models("anthropic"),
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
        # Google's own tools read GOOGLE_API_KEY as well; either name works here.
        gemini_api_key=env_str("GEMINI_API_KEY") or env_str("GOOGLE_API_KEY"),
        gemini_model=env_str("GEMINI_MODEL") or default_models("gemini"),
        gemini_base_url=(env_str("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL) or DEFAULT_GEMINI_BASE_URL).rstrip("/"),
        openrouter_api_key=env_str("OPENROUTER_API_KEY"),
        openrouter_model=env_str("OPENROUTER_MODEL") or default_models("openrouter"),
        openrouter_base_url=(
            env_str("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL) or DEFAULT_OPENROUTER_BASE_URL
        ).rstrip("/"),
        # Other tools know these keys under a second name; either one works here.
        nvidia_api_key=env_str("NVIDIA_API_KEY") or env_str("NVIDIA_NIM_API_KEY"),
        nvidia_model=env_str("NVIDIA_MODEL") or default_models("nvidia"),
        nvidia_base_url=(env_str("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL) or DEFAULT_NVIDIA_BASE_URL).rstrip("/"),
        zai_api_key=env_str("ZAI_API_KEY") or env_str("GLM_API_KEY"),
        zai_model=env_str("ZAI_MODEL") or default_models("zai"),
        zai_base_url=(env_str("ZAI_BASE_URL", DEFAULT_ZAI_BASE_URL) or DEFAULT_ZAI_BASE_URL).rstrip("/"),
        llm_fallback=env_bool("JULES_LLM_FALLBACK", True),
        llm_order=tuple(clean_provider_order(env_str("JULES_LLM_ORDER") or "")),
        commit_authorship=_commit_authorship(env_str("JULES_COMMIT_AUTHORSHIP")),
        commit_author_name=clean_person_name(env_str("JULES_COMMIT_AUTHOR_NAME")),
        commit_author_email=clean_email(env_str("JULES_COMMIT_AUTHOR_EMAIL")),
        ci_autofix=env_bool("JULES_CI_AUTOFIX", True),
        ci_autofix_attempts=env_int("JULES_CI_AUTOFIX_ATTEMPTS", 2, minimum=0, maximum=5),
        keep_unfinished_work=env_bool("JULES_KEEP_UNFINISHED_WORK", True),
        # "\n" typed into a one-line field of a hosting dashboard is meant as a line break.
        task_guidelines=(env_str("JULES_TASK_GUIDELINES") or "").replace("\\n", "\n").strip()[:MAX_GUIDELINES_CHARS],
    )


def _commit_authorship(raw: Optional[str]) -> str:
    value = (raw or "bot").strip().lower().replace("-", "").replace("_", "")
    aliases = {"bot": "bot", "jules": "bot", "app": "bot", "coauthor": "coauthor", "coauthored": "coauthor",
               "user": "user", "you": "user", "me": "user"}
    return aliases.get(value, "bot")


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
