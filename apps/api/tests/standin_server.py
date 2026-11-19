"""The real API on local stand-ins, served over HTTP.

For the dashboard's browser tests (``apps/web/tests/e2e``) and for trying the
dashboard out without an account or a key::

    python -m apps.api.tests.standin_server                  # http://127.0.0.1:8011
    API_BASE_URL=http://127.0.0.1:8011 npm run dev:web

What is real: this is ``apps.api.main:app`` - every route, the task runner,
the agent loop, the provider clients and the fallback between them.

What is not:

* GitHub is ``FakeGitHub`` over bare repositories in a temporary directory;
* the sandbox is a temporary directory and bash on this machine;
* the model providers are a scripted "model" served by this same process.
  Every provider's base URL is pointed at it, so a key that somebody types
  into the dashboard goes no further than this machine either.

It runs commands on the machine it is started on and asks for no key of its
own, so it listens on 127.0.0.1 only, answers only requests that were
addressed to this machine by that name, and is never to be deployed.

Tests steer it over ``/__standin``:

* ``POST /__standin/reset`` starts from scratch: the repositories as they
  were, no task, no pull request. Its JSON body may carry ``env`` (settings
  to add or, with ``null``, to take away), ``providers`` (how a provider
  behaves, see ``BEHAVIOURS``) and ``script`` (what the model does, turn by
  turn).
* ``POST /__standin/update`` takes the same body and changes only what it
  names: what exists stays, for the next run of a test in several steps.
* ``GET /__standin/state`` returns the requests the providers received and
  the pull requests that exist.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Before the app is imported: a developer's .env, with real keys in it, is not read.
os.environ["JULES_SKIP_DOTENV"] = "1"

from fastapi import APIRouter, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from apps.api.config import LLM_PROVIDERS, PROVIDER_KEY_VARIABLES  # noqa: E402
from apps.api.dependencies import get_runner_deps  # noqa: E402
from apps.api.main import app  # noqa: E402
from apps.api.routers import webhooks  # noqa: E402
from apps.api.services.llm import create_llm_client  # noqa: E402
from apps.api.services.runner import RunnerDeps  # noqa: E402
from apps.api.services.store import task_store  # noqa: E402

from .support import FakeGitHub, GitRemote, LocalWorkspace  # noqa: E402

DEFAULT_PORT = 8011
PREFIX = "/__standin"
# The names this server answers to. A request that reached it under another
# name came through a tunnel or a port forwarder - or from a web page that
# pointed a name of its own at 127.0.0.1 to get past the browser's same-origin
# rule. Neither has any business with a server that runs commands.
LOCAL_NAMES = ("127.0.0.1", "localhost", "::1")

# Providers whose API is the OpenAI one. Gemini and Anthropic have their own;
# the stand-in does not speak them and says so (HTTP 501) when asked.
OTHER_APIS = ("gemini", "anthropic")
ACCOUNT_IN_ADDRESS = ("cloudflare",)

# How a provider can be made to behave (``providers`` of a reset):
#   ok         answers
#   bad_key    refuses every key
#   day_used   its daily allowance is used up
#   no_credit  the account has no balance
#   down       the service is failing
# With ``after: n`` it answers n requests first.
BEHAVIOURS = ("ok", "bad_key", "day_used", "no_credit", "down")

DEMO_FILES = {
    "README.md": "# Demo\n\nA small demo project: a calculator with one bug.\n",
    "calc.py": "def add(a, b):\n    return a - b\n",
    "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    ".gitignore": "*.log\n__pycache__/\n",
}

# A test command that needs nothing but the interpreter this server runs on.
CHECK_ADD = "{python} -c 'from calc import add; assert add(2, 3) == 5, add(2, 3)'"

# What the model does when a test does not say otherwise: it reads the bug,
# fixes it, checks the fix and finishes. A step is a list of [tool, arguments]
# pairs - or a string, for a reply in words.
FIX_ADD: List[Any] = [
    [["read_file", {"path": "calc.py"}]],
    [["replace_in_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}]],
    [["run_tests", {"command": CHECK_ADD}]],
    [
        [
            "finish",
            {
                "summary": "add() subtracted instead of adding; fixed the operator.",
                "pr_title": "Fix add()",
            },
        ]
    ],
]
# When a script has run out the model wraps up rather than leave a run hanging.
WRAP_UP = [["finish", {"summary": "Nothing more to do.", "pr_title": "Update"}]]


class StandIn:
    """Everything the stand-in keeps between requests."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._lock = threading.RLock()
        self._root: Optional[Path] = None
        self._baseline: Dict[str, str] = {}
        self.github: FakeGitHub
        self.script: List[Any] = list(FIX_ADD)
        self.behaviour: Dict[str, Dict[str, Any]] = {}
        self.answered: Dict[str, int] = {}
        self.requests: List[Dict[str, Any]] = []

    # -- configuration ---------------------------------------------------
    def base_environment(self) -> Dict[str, str]:
        """A server that lacks nothing: one provider key, the sandbox, GitHub."""
        values = {
            "JULES_SKIP_DOTENV": "1",
            "E2B_API_KEY": "e2b_" + "s" * 40,
            "GITHUB_TOKEN": FakeGitHub.PAT,
            "GROQ_API_KEY": "standin-groq-key",
        }
        for provider in LLM_PROVIDERS:
            address = f"{self.base_url}{PREFIX}/llm/{provider}"
            if provider in ACCOUNT_IN_ADDRESS:
                address += "/accounts/{account_id}/ai/v1"
            values[f"{provider.upper()}_BASE_URL"] = address
        return values

    def remember_environment(self) -> None:
        """Called once at start: what a reset goes back to."""
        managed = set(PROVIDER_KEY_VARIABLES.values()) | {"GH_TOKEN", "ENVIRONMENT"}
        for name in list(os.environ):
            # Whatever this shell had configured for a real deployment is not for here.
            if name in managed or name.startswith(("JULES_", "GITHUB_", "E2B_", "VERCEL")) or name.endswith(
                ("_API_KEY", "_API_TOKEN", "_MODEL", "_BASE_URL", "_ACCOUNT_ID")
            ):
                del os.environ[name]
        os.environ.update(self.base_environment())
        self._baseline = dict(os.environ)

    def update(
        self,
        *,
        env: Optional[Dict[str, Optional[str]]] = None,
        providers: Optional[Dict[str, Any]] = None,
        script: Optional[Sequence[Any]] = None,
    ) -> None:
        """Change what is named; everything else stays as it is."""
        with self._lock:
            for name, value in (env or {}).items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = str(value)
            for name, value in (providers or {}).items():
                self.behaviour[name] = {"mode": value} if isinstance(value, str) else dict(value)
                self.answered.pop(name, None)  # "after n" counts from now
            if script is not None:
                self.script = list(script)

    def reset(
        self,
        *,
        env: Optional[Dict[str, Optional[str]]] = None,
        providers: Optional[Dict[str, Any]] = None,
        script: Optional[Sequence[Any]] = None,
    ) -> None:
        with self._lock:
            os.environ.clear()
            os.environ.update(self._baseline)
            task_store.clear()
            webhooks._seen_deliveries.clear()
            webhooks._claimed_commits.clear()
            if self._root is not None:
                shutil.rmtree(self._root, ignore_errors=True)
            self._root = Path(tempfile.mkdtemp(prefix="jules-standin-"))
            self.github = FakeGitHub(GitRemote(self._root / "remotes"))
            self.github.add_repo("acme/demo", DEMO_FILES, description="A calculator with one bug", language="Python")
            self.github.add_repo("acme/empty", None, empty=True, description="Nothing in it yet")
            self.github.add_listed_repo("acme/website", description="The marketing site", language="TypeScript")
            self.github.add_listed_repo("acme/api-server", private=True, language="Go")
            self.github.add_listed_repo("octo-org/handbook", private=True, language="Markdown")
            self.script = list(FIX_ADD)
            self.behaviour = {}
            self.answered = {}
            self.requests = []
            self.update(env=env, providers=providers, script=script)

    def close(self) -> None:
        if self._root is not None:
            shutil.rmtree(self._root, ignore_errors=True)

    # -- the systems a task works with -------------------------------------
    def deps(self) -> RunnerDeps:
        root = self._root
        assert root is not None
        return RunnerDeps(
            github_factory=lambda settings: self.github.client(settings),
            workspace_factory=lambda _settings, task_id, _lifetime: LocalWorkspace(root / "sandboxes" / task_id),
            # The real clients: they reach the scripted model through the base URLs above.
            llm_factory=lambda settings, notice: create_llm_client(settings, on_notice=notice),
            clone_url_for=self.github.remote.clone_url,
        )

    # -- the model ---------------------------------------------------------
    def answer(self, provider: str, key: str, account: Optional[str], body: Dict[str, Any]) -> JSONResponse:
        model = str(body.get("model") or "")
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        tools = [
            str((tool.get("function") or {}).get("name") or "")
            for tool in body.get("tools") or []
            if isinstance(tool, dict)
        ]
        turn = sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "assistant")
        with self._lock:
            behaviour = self.behaviour.get(provider) or {}
            mode = str(behaviour.get("mode") or "ok")
            answered = self.answered.get(provider, 0)
            if answered < int(behaviour.get("after") or 0):
                mode = "ok"
            if "bad" in key.lower():
                mode = "bad_key"
            entry = {
                "provider": provider,
                "model": model,
                "key": key,
                "account": account,
                "turn": turn,
                "tools": tools,
                "outcome": mode,
            }
            self.requests.append(entry)
            if mode == "ok":
                self.answered[provider] = answered + 1
            script = list(self.script)
        if mode != "ok":
            return _refusal(mode, model)
        if "confirm" in tools:  # the connection test of Settings
            step: Any = [["confirm", {"ready": True}]]
        else:
            step = script[turn] if turn < len(script) else WRAP_UP
        return JSONResponse(_completion(model, step, turn, messages))


def _refusal(mode: str, model: str) -> JSONResponse:
    if mode == "bad_key":
        return JSONResponse(
            {"error": {"message": "Invalid API Key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status_code=401,
        )
    if mode == "day_used":
        return JSONResponse(
            {
                "error": {
                    "message": f"Rate limit reached for model `{model}` on requests per day (RPD): "
                    "Limit 1000, Used 1000, Requested 1. Please try again in 9h12m.",
                    "type": "requests",
                    "code": "rate_limit_exceeded",
                }
            },
            status_code=429,
        )
    if mode == "no_credit":
        return JSONResponse({"error": {"message": "Insufficient Balance"}}, status_code=402)
    return JSONResponse({"error": {"message": "The service is unavailable."}}, status_code=503)


def _fill(value: Any) -> Any:
    """``{python}`` in a script stands for the interpreter this server runs on."""
    if isinstance(value, str):
        return value.replace("{python}", sys.executable)
    if isinstance(value, dict):
        return {key: _fill(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fill(item) for item in value]
    return value


def _completion(model: str, step: Any, turn: int, messages: List[Any]) -> Dict[str, Any]:
    if isinstance(step, str):
        message: Dict[str, Any] = {"role": "assistant", "content": step}
        finish_reason = "stop"
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{turn}_{position}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(_fill(arguments))},
                }
                for position, (name, arguments) in enumerate(step)
            ],
        }
        finish_reason = "tool_calls"
    prompt_tokens = sum(len(json.dumps(item)) for item in messages) // 4
    completion_tokens = max(1, len(json.dumps(message)) // 4)
    return {
        "id": f"chatcmpl-standin-{turn}",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _bearer(request: Request) -> str:
    scheme, _, value = (request.headers.get("authorization") or "").partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


def install(standin: StandIn) -> None:
    """Point the app at the stand-ins and add the routes that steer them."""
    router = APIRouter(prefix=PREFIX, include_in_schema=False)

    async def steer(request: Request, apply: Any) -> JSONResponse:
        raw = await request.body()
        body = json.loads(raw) if raw.strip() else {}
        for name, value in (body.get("providers") or {}).items():
            mode = value.get("mode", "ok") if isinstance(value, dict) else value
            if name not in LLM_PROVIDERS or mode not in BEHAVIOURS:
                # A typing mistake in a test must not pass for "behaves normally".
                return JSONResponse({"error": f"unknown provider or behaviour: {name}={mode}"}, status_code=400)
        apply(env=body.get("env"), providers=body.get("providers"), script=body.get("script"))
        return JSONResponse({"ok": True})

    @router.post("/reset")
    async def reset(request: Request) -> JSONResponse:
        return await steer(request, standin.reset)

    @router.post("/update")
    async def update(request: Request) -> JSONResponse:
        return await steer(request, standin.update)

    @router.get("/state")
    def state() -> Dict[str, Any]:
        with standin._lock:
            return {
                "requests": list(standin.requests),
                "answered": dict(standin.answered),
                "pulls": {
                    slug: [
                        {
                            "number": pull["number"],
                            "title": pull["title"],
                            "state": pull["state"],
                            "head": pull["head"]["ref"],
                            "base": pull["base"]["ref"],
                        }
                        for pull in pulls
                    ]
                    for slug, pulls in standin.github.pulls.items()
                    if pulls
                },
                "comments": len(standin.github.comments),
            }

    @router.post("/llm/cloudflare/accounts/{account}/ai/v1/chat/completions")
    async def cloudflare_chat(account: str, request: Request) -> JSONResponse:
        return standin.answer("cloudflare", _bearer(request), account, json.loads(await request.body()))

    @router.post("/llm/{provider}/chat/completions")
    async def chat(provider: str, request: Request) -> JSONResponse:
        if provider not in LLM_PROVIDERS or provider in OTHER_APIS or provider in ACCOUNT_IN_ADDRESS:
            return _no_stand_in(provider)
        return standin.answer(provider, _bearer(request), None, json.loads(await request.body()))

    @router.api_route("/llm/{provider}/{rest:path}", methods=["GET", "POST"])
    async def anything_else(provider: str, rest: str) -> JSONResponse:
        if provider in OTHER_APIS or provider not in LLM_PROVIDERS:
            return _no_stand_in(provider)
        if rest.endswith("models"):
            return JSONResponse({"object": "list", "data": []})
        return JSONResponse({"error": {"message": f"The stand-in has no route {rest!r}."}}, status_code=404)

    @app.middleware("http")
    async def local_requests_only(request: Request, call_next: Any) -> Any:
        if (request.url.hostname or "") not in LOCAL_NAMES:
            return JSONResponse(
                {"detail": "The stand-in server only answers requests addressed to 127.0.0.1 or localhost."},
                status_code=403,
            )
        return await call_next(request)

    app.include_router(router)
    app.dependency_overrides[get_runner_deps] = standin.deps


def _no_stand_in(provider: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": f"The stand-in does not speak the API of '{provider}'. Use one of the OpenAI-compatible providers."}},
        status_code=501,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="The API on local stand-ins (see the module's docstring).")
    parser.add_argument("--port", type=int, default=int(os.environ.get("STANDIN_PORT") or DEFAULT_PORT))
    options = parser.parse_args(argv)

    import uvicorn

    # 127.0.0.1 and nothing else: this server runs commands and has no access key.
    host = "127.0.0.1"
    standin = StandIn(f"http://{host}:{options.port}")
    standin.remember_environment()
    standin.reset()
    install(standin)
    try:
        uvicorn.run(app, host=host, port=options.port, log_level="warning")
    finally:
        standin.close()


if __name__ == "__main__":
    main()
