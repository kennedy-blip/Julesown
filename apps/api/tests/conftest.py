"""Shared fixtures. The suite is hermetic: no network, no real credentials."""
from __future__ import annotations

import os
import sys

# Must be set before the app is imported so a developer's .env is not loaded.
os.environ["JULES_SKIP_DOTENV"] = "1"

import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies import get_runner_deps
from apps.api.main import app
from apps.api.routers import webhooks
from apps.api.services.store import task_store

from .support import FakeGitHub, GitRemote, ScriptedLLM, make_deps

CONFIG_VARIABLES = (
    "ENVIRONMENT",
    "VERCEL",
    "VERCEL_ENV",
    "VERCEL_REGION",
    "JULES_LLM_PROVIDER",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_BASE_URL",
    "GROQ_REASONING_EFFORT",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_REASONING_EFFORT",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_BASE_URL",
    "E2B_API_KEY",
    "E2B_TEMPLATE",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_API_URL",
    "GITHUB_WEBHOOK_SECRET",
    "JULES_ACCESS_KEY",
    "JULES_ALLOW_PUBLIC_ACCESS",
    "JULES_TASK_BUDGET_SECONDS",
    "JULES_MAX_AGENT_TURNS",
    "JULES_COMMAND_TIMEOUT_SECONDS",
    "JULES_MAX_OUTPUT_TOKENS",
)

DEMO_FILES = {
    "README.md": "# Demo\n\nA small demo projct.\n",
    "calc.py": "def add(a, b):\n    return a - b\n",
    "tests/test_calc.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    ".gitignore": "*.log\n",
}

# A test command that needs nothing but the interpreter running the suite.
CHECK_ADD = f"{sys.executable} -c 'from calc import add; assert add(2, 3) == 5, add(2, 3)'"


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    for name in CONFIG_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    task_store.clear()
    webhooks._seen_deliveries.clear()
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()
    task_store.clear()


@pytest.fixture(scope="session")
def rsa_private_key_pem() -> str:
    """An RSA key in the PKCS#1 PEM format GitHub generates for Apps."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode("ascii")


@pytest.fixture
def remote(tmp_path) -> GitRemote:
    return GitRemote(tmp_path / "remotes")


@pytest.fixture
def github(remote) -> FakeGitHub:
    fake = FakeGitHub(remote)
    fake.add_repo("acme/demo", DEMO_FILES)
    return fake


@pytest.fixture
def token_env(monkeypatch):
    """A fully configured server authenticating to GitHub with a personal token."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "g" * 40)
    monkeypatch.setenv("E2B_API_KEY", "e2b_" + "e" * 40)
    monkeypatch.setenv("GITHUB_TOKEN", FakeGitHub.PAT)


@pytest.fixture
def app_env(monkeypatch, rsa_private_key_pem):
    """A fully configured server authenticating to GitHub as a GitHub App."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "g" * 40)
    monkeypatch.setenv("E2B_API_KEY", "e2b_" + "e" * 40)
    monkeypatch.setenv("GITHUB_APP_ID", "123456")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", rsa_private_key_pem)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def install_deps(github, tmp_path):
    """Returns a function that points the API at the local stand-ins."""

    def install(llm, workspaces=None):
        deps = make_deps(
            github=github, llm=llm, workspace_root=tmp_path / "sandboxes", workspaces=workspaces
        )
        app.dependency_overrides[get_runner_deps] = lambda: deps
        return deps

    return install


def fix_add_script() -> ScriptedLLM:
    """A model run that reads the bug, fixes it, verifies it and finishes."""
    return ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"})],
            [("replace_in_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"})],
            [("run_tests", {"command": CHECK_ADD})],
            [("finish", {"summary": "add() subtracted instead of adding; fixed the operator.", "pr_title": "Fix add()"})],
        ]
    )
