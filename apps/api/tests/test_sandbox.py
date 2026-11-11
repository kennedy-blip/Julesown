"""The E2B adapter, checked against a stand-in that mimics the SDK's contract.

The real ``e2b`` package is installed, so the genuine exception classes are
used and the keyword arguments are validated against the genuine signatures -
only the network calls are replaced.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

import pytest

import e2b
from e2b import CommandExitException, TimeoutException
from e2b.exceptions import (
    AuthenticationException,
    FileNotFoundException,
    RateLimitException,
    SandboxNotFoundException,
    SandboxNotRunningException,
)

from apps.api.services.sandbox import E2BWorkspace, SandboxError


class FakeResult:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout, self.stderr, self.exit_code, self.error = stdout, stderr, exit_code, None


class FakeCommands:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.behaviours: Dict[str, Any] = {}

    def run(self, cmd: str, **kwargs: Any) -> FakeResult:
        # Reject keyword arguments the real SDK method does not accept.
        inspect.signature(_REAL_RUN).bind(None, cmd, **kwargs)
        self.calls.append({"cmd": cmd, **kwargs})
        behaviour = self.behaviours.get(cmd)
        if isinstance(behaviour, BaseException):
            raise behaviour
        if behaviour is not None:
            return behaviour
        return FakeResult(stdout="/home/user\n" if cmd == "pwd" else "")


class FakeFiles:
    def __init__(self) -> None:
        self.store: Dict[str, str] = {}
        self.error: Optional[BaseException] = None

    def read(self, path: str) -> str:
        if self.error:
            raise self.error
        if path not in self.store:
            raise FileNotFoundException(f"path '{path}' does not exist")
        return self.store[path]

    def write(self, path: str, data: str) -> None:
        if self.error:
            raise self.error
        self.store[path] = data


class FakeSandbox:
    created: List[Dict[str, Any]] = []
    create_error: Optional[BaseException] = None

    def __init__(self) -> None:
        self.sandbox_id = "sbx_fake123"
        self.commands = FakeCommands()
        self.files = FakeFiles()
        self.killed = False

    @classmethod
    def create(cls, **kwargs: Any) -> "FakeSandbox":
        inspect.signature(_REAL_CREATE).bind(**{k: v for k, v in kwargs.items() if k not in ("api_key", "request_timeout")})
        cls.created.append(kwargs)
        if cls.create_error:
            raise cls.create_error
        return cls()

    def kill(self) -> bool:
        self.killed = True
        return True


from e2b.sandbox_sync.commands.command import Commands as _RealCommands  # noqa: E402

_REAL_RUN = _RealCommands.run
_REAL_CREATE = e2b.Sandbox.create


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeSandbox.created = []
    FakeSandbox.create_error = None
    monkeypatch.setattr(e2b, "Sandbox", FakeSandbox)
    return FakeSandbox


def test_api_params_accepted_by_the_real_sdk():
    """`api_key` and `request_timeout` travel through **opts (ApiParams)."""
    from e2b.connection_config import ApiParams

    assert {"api_key", "request_timeout"} <= set(ApiParams.__annotations__)


def test_create_passes_key_lifetime_template_and_metadata(fake_sdk):
    workspace = E2BWorkspace.create(
        api_key="e2b_key", template="my-template", lifetime_seconds=330, metadata={"task": "abc"}
    )
    assert fake_sdk.created == [
        {"timeout": 330, "api_key": "e2b_key", "request_timeout": 30.0, "template": "my-template", "metadata": {"task": "abc"}}
    ]
    assert workspace.sandbox_id == "sbx_fake123" and workspace.description == "E2B sandbox sbx_fake123"
    assert workspace.workdir == "/home/user"

    E2BWorkspace.create(api_key="e2b_key")
    assert "template" not in fake_sdk.created[-1]  # default template unless configured


def test_create_failures_are_explained(fake_sdk):
    with pytest.raises(SandboxError) as caught:
        E2BWorkspace.create(api_key="")
    assert "E2B_API_KEY is not configured" in str(caught.value)

    fake_sdk.create_error = AuthenticationException("401: Invalid API key")
    with pytest.raises(SandboxError) as caught:
        E2BWorkspace.create(api_key="bad")
    assert "check E2B_API_KEY" in str(caught.value)

    fake_sdk.create_error = RateLimitException("429: too many sandboxes")
    with pytest.raises(SandboxError) as caught:
        E2BWorkspace.create(api_key="k")
    assert "limit reached" in str(caught.value)


def test_run_maps_results_exits_and_timeouts(fake_sdk):
    workspace = E2BWorkspace.create(api_key="k")
    commands = workspace._sandbox.commands

    commands.behaviours["ok"] = FakeResult(stdout="out\n", stderr="warn\n")
    result = workspace.run("ok", cwd="/home/user/repo", env={"CI": "true"}, timeout=42)
    assert (result.exit_code, result.stdout, result.stderr, result.timed_out, result.ok) == (0, "out\n", "warn\n", False, True)
    assert commands.calls[-1] == {"cmd": "ok", "cwd": "/home/user/repo", "envs": {"CI": "true"}, "timeout": 42.0}

    # The SDK raises on a non-zero exit; the adapter returns it as a result.
    commands.behaviours["fail"] = CommandExitException(stderr="boom\n", stdout="partial\n", exit_code=3, error=None)
    result = workspace.run("fail")
    assert (result.exit_code, result.stdout, result.stderr, result.ok) == (3, "partial\n", "boom\n", False)
    assert commands.calls[-1]["cwd"] == "/home/user" and commands.calls[-1]["envs"] is None

    commands.behaviours["slow"] = TimeoutException("context deadline exceeded")
    result = workspace.run("slow", timeout=7)
    assert result.timed_out and result.exit_code == 124 and "timed out after 7 seconds" in result.stderr

    # A sandbox that has gone away is fatal, not a command timeout.
    commands.behaviours["gone"] = SandboxNotRunningException("sandbox was killed")
    with pytest.raises(SandboxError) as caught:
        workspace.run("gone")
    assert "no longer reachable" in str(caught.value)

    commands.behaviours["weird"] = RuntimeError("connection reset")
    with pytest.raises(SandboxError):
        workspace.run("weird")


def test_files_and_close(fake_sdk):
    workspace = E2BWorkspace.create(api_key="k")
    workspace.write_file("/home/user/repo/a.txt", "héllo\n")
    assert workspace.read_file("/home/user/repo/a.txt") == "héllo\n"
    with pytest.raises(FileNotFoundError):
        workspace.read_file("/home/user/repo/missing.txt")

    workspace._sandbox.files.error = SandboxNotFoundException("sandbox not found")
    with pytest.raises(SandboxError):  # the sandbox is gone - not "file missing"
        workspace.read_file("/home/user/repo/a.txt")
    with pytest.raises(SandboxError):
        workspace.write_file("/home/user/repo/a.txt", "x")

    workspace.close()
    assert workspace._sandbox.killed

    def broken_kill():
        raise RuntimeError("already gone")

    workspace._sandbox.kill = broken_kill
    workspace.close()  # best effort, never raises


def test_sandbox_that_dies_during_startup_is_cleaned_up(fake_sdk, monkeypatch):
    killed: List[bool] = []
    original_init = FakeSandbox.__init__

    def init(self):
        original_init(self)
        self.commands.behaviours["pwd"] = SandboxNotRunningException("gone")
        self.kill = lambda: killed.append(True) or True

    monkeypatch.setattr(FakeSandbox, "__init__", init)
    with pytest.raises(SandboxError):
        E2BWorkspace.create(api_key="k")
    assert killed == [True]
