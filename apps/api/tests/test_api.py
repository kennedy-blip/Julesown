"""End-to-end API tests.

The whole pipeline runs for real against local stand-ins (see ``support.py``):
a real git clone, real file edits, real commands, real git objects pushed to a
real (bare) remote and a pull request created through the GitHub REST calls.
Nothing in the pipeline is simulated, which is the point - the previous
implementation reported success and handed out pull request links without
doing any of it.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from apps.api.services.llm import LLMError
from apps.api.services.sandbox import SandboxError
from apps.api.services.store import branch_for_task, new_task_id, task_store

from apps.api.routers.webhooks import build_result_comment

from .conftest import CHECK_ADD, fix_add_script
from .support import FakeGitHub, LocalWorkspace, ScriptedLLM, git

REAL_PR_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/pull/\d+$")
TASK = {"repository": "acme/demo", "branch": "main", "prompt": "add() returns the wrong result, fix it"}


def run_stream(
    client: TestClient, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    response = client.post("/api/v1/tasks/stream", json=payload, headers=headers or {})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0]["type"] == "task" and events[-1]["type"] == "done"
    return events, events[-1]["task"]


def messages(task: Dict[str, Any], level: Optional[str] = None) -> List[str]:
    return [entry["message"] for entry in task["log_entries"] if level is None or entry["level"] == level]


# --------------------------------------------------------------------------- #
# Health and configuration
# --------------------------------------------------------------------------- #
def test_health_endpoints(client: TestClient):
    for path in ("/api/health", "/api/v1/health"):
        body = client.get(path).json()
        assert body["status"] == "ok" and body["service"] == "jules-clone-api"


def test_env_health_reports_what_is_missing(client: TestClient):
    body = client.get("/api/v1/health/env").json()
    assert body["status"] == "INCOMPLETE" and body["ready"] is False
    assert len(body["missing"]) == 3
    # The same by kind, so that a caller that brings its own model key knows what else is missing.
    assert list(body["needs"]) == ["llm", "sandbox", "github"]
    assert [message for message in body["needs"].values()] == body["missing"]
    assert body["llm"] is None and body["github"] == {"auth_mode": "none"}
    assert body["access"] == {"mode": "open"}
    assert set(body["variables"].values()) == {False}


def test_env_health_ready_and_never_leaks_values(client: TestClient, app_env, monkeypatch):
    monkeypatch.setenv("JULES_ACCESS_KEY", "open-sesame-123")
    response = client.get("/api/v1/health/env")
    body = response.json()
    assert body["status"] == "HEALTHY" and body["ready"] is True and body["missing"] == []
    assert body["needs"] == {"llm": None, "sandbox": None, "github": None}
    assert body["llm_preferences"] == {
        "order": ["groq", "gemini", "openrouter", "openai", "anthropic"],
        "first_choice": None,
        "fallback": True,
    }
    assert body["llm"] == {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "chain": [{"provider": "groq", "model": "openai/gpt-oss-120b"}],
        "fallback": True,
    }
    assert [(item["id"], item["configured"]) for item in body["providers"]] == [
        ("groq", True), ("gemini", False), ("openrouter", False), ("openai", False), ("anthropic", False)
    ]
    assert body["providers"][1] == {
        "id": "gemini",
        "label": "Google Gemini",
        "configured": False,
        "model": "gemini-flash-latest",
        "key_variable": "GEMINI_API_KEY",
        "model_variable": "GEMINI_MODEL",
    }
    assert body["commits"] == {"authorship": "bot", "author_configured": False, "author_name": None}
    assert body["automation"] == {"ci_autofix": True, "ci_autofix_attempts": 2, "keep_unfinished_work": True}
    assert body["github"] == {"auth_mode": "app"} and body["access"] == {"mode": "key"}
    assert body["variables"]["GITHUB_APP_PRIVATE_KEY"] is True and body["variables"]["GITHUB_TOKEN"] is False
    for secret in ("open-sesame-123", "gsk_", "e2b_", "PRIVATE KEY"):
        assert secret not in response.text


# --------------------------------------------------------------------------- #
# Access control
# --------------------------------------------------------------------------- #
def test_open_locally_locked_on_public_deployments(client: TestClient, monkeypatch):
    assert client.get("/api/v1/tasks").status_code == 200  # local development

    monkeypatch.setenv("VERCEL", "1")
    for method, path in (("GET", "/api/v1/tasks"), ("POST", "/api/v1/tasks"), ("POST", "/api/v1/tasks/stream"), ("GET", "/api/v1/github/repos")):
        response = client.request(method, path, json=TASK if method == "POST" else None)
        assert response.status_code == 403, path
        assert response.json()["detail"]["code"] == "access_locked"
        assert "JULES_ACCESS_KEY" in response.json()["detail"]["message"]
    assert task_store.list() == []  # nothing was started
    # Diagnostics and GitHub redirects stay reachable so the dashboard can explain the lock.
    assert client.get("/api/v1/health/env").json()["access"] == {"mode": "locked"}
    assert client.get("/api/v1/github/setup?installation_id=1", follow_redirects=False).status_code == 307

    monkeypatch.setenv("JULES_ALLOW_PUBLIC_ACCESS", "true")
    assert client.get("/api/v1/tasks").status_code == 200


def test_access_key_is_required_when_configured(client: TestClient, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("JULES_ACCESS_KEY", "open-sesame-123")
    denied = client.get("/api/v1/tasks")
    assert denied.status_code == 401 and denied.json()["detail"]["code"] == "access_key_required"
    assert client.get("/api/v1/tasks", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/v1/tasks/stream", json=TASK, headers={"X-Jules-Key": "nope"}).status_code == 401
    assert client.get("/api/v1/tasks", headers={"Authorization": "Bearer open-sesame-123"}).status_code == 200
    assert client.get("/api/v1/tasks", headers={"X-Jules-Key": "open-sesame-123"}).status_code == 200
    assert client.get("/api/tasks", headers={"X-Jules-Key": "open-sesame-123"}).status_code == 200


def test_other_websites_cannot_start_tasks_from_a_visitors_browser(client: TestClient, token_env, install_deps):
    """No cross-origin access is granted unless JULES_CORS_ORIGINS says so, and a
    task is only accepted as JSON - which a browser will not send cross-origin
    without the pre-flight that is refused here."""
    install_deps(fix_add_script())
    preflight = client.options(
        "/api/v1/tasks/stream",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
    )
    assert preflight.status_code >= 400 and "access-control-allow-origin" not in preflight.headers
    assert "access-control-allow-origin" not in client.get("/api/v1/health", headers={"Origin": "https://evil.example"}).headers
    # A cross-origin POST that needs no pre-flight cannot carry a JSON content
    # type, so the body is refused whatever it contains - for both task routes.
    body = json.dumps(TASK).encode()
    for path in ("/api/v1/tasks/stream", "/api/v1/tasks"):
        for content_type in ("text/plain;charset=UTF-8", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x", None):
            request = client.build_request("POST", path, content=body, headers={"Origin": "https://evil.example"})
            request.headers.pop("content-type", None)
            if content_type:
                request.headers["content-type"] = content_type
            response = client.send(request)
            assert response.status_code == 415, (path, content_type, response.text)
            assert response.json()["detail"]["code"] == "unsupported_media_type"
    assert task_store.list() == []
    # The dashboard's own request (same origin, JSON) is unaffected.
    accepted = client.post("/api/v1/tasks", content=body, headers={"Content-Type": "application/json; charset=utf-8"})
    assert accepted.status_code == 201


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({"repository": "not a repo", "prompt": "x"}, "repository"),
        ({"repository": "https://gitlab.com/a/b", "prompt": "x"}, "repository"),
        ({"repository": "acme/demo", "prompt": "   "}, "Describe the task"),
        ({"repository": "acme/demo", "prompt": "x", "branch": "bad branch"}, "branch"),
        ({"repository": "acme/demo"}, "Describe the task"),
        ({"repository": "acme/demo", "prompt": "x", "pull_request": 0}, "pull_request"),
        ({"repository": "acme/demo", "prompt": "x", "pull_request": 3, "branch": "main"}, "leave branch empty"),
    ],
)
def test_invalid_requests_are_rejected_with_a_readable_message(client: TestClient, payload, fragment):
    for path in ("/api/v1/tasks", "/api/v1/tasks/stream"):
        response = client.post(path, json=payload)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["code"] == "invalid_request" and fragment in detail["message"]
    assert task_store.list() == []


def test_repository_urls_are_normalised(client: TestClient, token_env, install_deps):
    install_deps(ScriptedLLM([[("finish", {"summary": "Nothing to change."})]]))
    _events, task = run_stream(
        client, {"repository": "https://github.com/acme/demo/tree/main", "prompt": "check"}
    )
    assert task["repository"] == "acme/demo" and task["status"] == "completed"


# --------------------------------------------------------------------------- #
# The real pipeline
# --------------------------------------------------------------------------- #
def test_task_opens_a_real_pull_request(client: TestClient, token_env, install_deps, github: FakeGitHub):
    workspaces: List[LocalWorkspace] = []
    llm = fix_add_script()
    install_deps(llm, workspaces)
    main_before = github.remote.rev("acme/demo", "refs/heads/main")

    events, task = run_stream(client, TASK)

    # --- what the API reports -------------------------------------------
    assert task["status"] == "completed" and task["stage"] == "done" and task["error"] is None
    assert task["outcome"] == "pull_request"
    assert task["pr_url"] == "https://github.com/acme/demo/pull/1" and task["pr_number"] == 1
    assert task["head_branch"] == branch_for_task(task["task_id"]) and task["base_branch"] == "main"
    assert task["tests_status"] == "passed" and task["tests_passed"] is True
    assert task["verification"] == [
        {
            "command": CHECK_ADD,
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "duration": task["verification"][0]["duration"],
            "outcome": "passed",
            "note": "",
        }
    ]
    assert task["changed_files"] == [{"path": "calc.py", "status": "M", "additions": 1, "deletions": 1}]
    assert task["docs_only"] is False and task["files_read"] == ["README.md", "calc.py"]
    assert task["summary"].startswith("add() subtracted")
    assert (task["llm_provider"], task["llm_model"]) == ("scripted", "scripted-model")
    assert task["sandbox"] == "local test workspace"
    assert task["started_at"] <= task["completed_at"]

    # --- what actually exists on the remote -----------------------------
    head = github.remote.rev("acme/demo", f"refs/heads/{task['head_branch']}")
    assert head == task["commit_sha"]
    assert github.remote.rev("acme/demo", "refs/heads/main") == main_before  # base untouched
    assert github.remote.changed("acme/demo", "main", task["head_branch"]) == ["M\tcalc.py"]
    assert "return a + b" in github.remote.show("acme/demo", task["head_branch"], "calc.py")
    parents = github.remote.rev("acme/demo", f"{head}^")
    assert parents == main_before  # exactly one commit on top of the base

    # --- the pull request GitHub was asked to create --------------------
    (pull,) = github.pulls["acme/demo"]
    assert (pull["head"]["ref"], pull["base"]["ref"], pull["title"]) == (task["head_branch"], "main", "Fix add()")
    assert "add() subtracted instead of adding" in pull["body"]
    assert "> add() returns the wrong result, fix it" in pull["body"]
    assert "`calc.py` (modified, +1 / -1)" in pull["body"]
    assert "Files the agent looked at: `README.md`, `calc.py`" in pull["body"]
    assert "✅ passed" in pull["body"] and task["task_id"] in pull["body"]

    # --- progress stream --------------------------------------------------
    kinds = {event["type"] for event in events}
    assert {"task", "log", "update", "done"} <= kinds
    stages = [event["fields"]["stage"] for event in events if event["type"] == "update" and "stage" in event["fields"]]
    assert stages == ["configuration", "github", "sandbox", "clone", "agent", "commit", "pull_request", "done"]
    # Applying the streamed updates to the first event reproduces the final record.
    folded = dict(events[0]["task"])
    for event in events:
        if event["type"] == "update":
            folded.update(event["fields"])
    assert folded == {key: value for key, value in task.items() if key not in ("logs", "log_entries")}
    assert [event["entry"] for event in events if event["type"] == "log"] == task["log_entries"]

    assert messages(task, "success")[-1] == "Pull request #1 opened: https://github.com/acme/demo/pull/1"
    assert f"$ {CHECK_ADD}" in messages(task, "command")
    assert all(re.match(r"^\[\d\d:\d\d:\d\d\] ", line) for line in task["logs"])
    assert len(task["logs"]) == len(task["log_entries"])

    # --- clean up ---------------------------------------------------------
    assert len(workspaces) == 1 and workspaces[0].closed and llm.closed


def test_task_with_github_app_credentials(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["pr_url"] == "https://github.com/acme/demo/pull/1"
    assert "GitHub access confirmed for acme/demo via GitHub App installation" in messages(task)
    assert github.token_requests and all(request["repositories"] == ["demo"] for request in github.token_requests)


@pytest.mark.parametrize("branch", ["", "   ", None])
def test_blank_branch_uses_the_repository_default(
    client: TestClient, token_env, install_deps, github: FakeGitHub, branch
):
    """The form used to send "main" for every repository, including ones without it."""
    github.add_repo("acme/trunky", {"calc.py": "def add(a, b):\n    return a - b\n"}, default_branch="trunk")
    install_deps(fix_add_script())
    payload = {"repository": "acme/trunky", "prompt": "fix add"}
    if branch is not None:
        payload["branch"] = branch
    _events, task = run_stream(client, payload)
    assert task["status"] == "completed", task["error"]
    assert task["base_branch"] == "trunk" and task["branch"] == "trunk"
    assert github.pulls["acme/trunky"][0]["base"]["ref"] == "trunk"


def test_no_changes_means_no_pull_request(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("finish", {"summary": "The code already does what was asked."})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["outcome"] == "no_changes"
    assert task["pr_url"] is None and task["pr_number"] is None and task["commit_sha"] is None
    assert task["changed_files"] == [] and task["tests_status"] == "not_run" and task["tests_passed"] is None
    assert messages(task, "success")[-1] == "Finished without changing any files, so no pull request was opened."
    assert github.pulls["acme/demo"] == []
    assert github.calls("POST", r"/git/|/pulls") == []
    assert github.remote.rev("acme/demo", f"refs/heads/{task['head_branch']}") is None


def test_failing_tests_are_reported_as_failing(client: TestClient, token_env, install_deps, github: FakeGitHub):
    """The old implementation always claimed "tests passed"."""
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("write_file", {"path": "NOTES.md", "content": "work in progress\n"})],
                [("run_tests", {"command": CHECK_ADD})],  # still failing: add() was not fixed
                [("finish", {"summary": "Added notes; the add() test still fails."})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and REAL_PR_URL.match(task["pr_url"])
    assert task["tests_status"] == "failed" and task["tests_passed"] is False
    assert task["verification"][0]["passed"] is False and task["verification"][0]["exit_code"] == 1
    assert any("last test run failed" in message for message in messages(task, "warn"))
    body = github.pulls["acme/demo"][0]["body"]
    assert "❌ failed (exit 1)" in body and "did not pass. Review before merging." in body


def test_untested_changes_are_labelled_untested(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("write_file", {"path": "helpers.py", "content": "def double(x):\n    return 2 * x\n"})],
                [("finish", {"summary": "Added a helper."})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["tests_status"] == "not_run" and task["tests_passed"] is None
    assert task["changed_files"] == [{"path": "helpers.py", "status": "A", "additions": 2, "deletions": 0}]
    assert task["docs_only"] is False
    assert "did not run any tests" in github.pulls["acme/demo"][0]["body"]
    assert any("did not run tests" in message for message in messages(task, "warn"))
    assert github.pulls["acme/demo"][0]["title"] == "Added a helper."  # falls back to the summary


# npm's placeholder `test` script (what `npm init` writes), without needing npm.
NPM_PLACEHOLDER_TEST = """sh -c 'echo "Error: no test specified" && exit 1'"""


def refused_reply(tool: str) -> LLMError:
    return LLMError(
        f"groq refused the model's reply because it was not a valid tool call (it tried to call a tool named '{tool}', which does not exist).",
        kind="tool_format",
        status_code=400,
        rejected_tool=tool,
    )


def test_first_production_run_replayed(client: TestClient, token_env, install_deps, github: FakeGitHub):
    """The model does what it did in the first run on the deployed app
    (kennedy-blip/Kensheets, "add spec.md and details on the project
    dependencies"): it writes the file without reading anything, answers in
    plain text, runs `npm test` on a project whose test script is npm's
    placeholder, and twice sends a tool call the provider refuses.

    That run ended with "Tests: Last run failed", a red warning in the pull
    request, and a spec whose dependency list was invented. This is what the
    pipeline makes of the same behaviour now."""
    invented = "# Spec\n\nDepends on react, chart.js and webpack. Licence: MIT.\n"
    grounded = "# Spec\n\nA calculator module (`calc.py`) with one function, `add`.\n"
    install_deps(
        ScriptedLLM(
            [
                [("write_file", {"path": "spec.md", "content": invented})],
                "Added a new **spec.md** file detailing the project overview and dependencies.",
                [("read_file", {"path": "calc.py"})],
                [("write_file", {"path": "spec.md", "content": grounded})],
                [("run_tests", {"command": NPM_PLACEHOLDER_TEST})],
                refused_reply("commentary"),
                refused_reply("json"),
                LLMError("groq returned HTTP 429: slow down", kind="rate_limit", status_code=429, retry_after=0.01),
                [("finish", {"summary": "Added spec.md describing the module.", "pr_title": "Add spec.md"})],
            ]
        )
    )
    _events, task = run_stream(client, {**TASK, "prompt": "add spec.md and details on the project dependencies"})

    assert task["status"] == "completed" and REAL_PR_URL.match(task["pr_url"])
    # The invented version never reached the repository.
    assert github.remote.show("acme/demo", task["head_branch"], "spec.md") == grounded.strip()
    assert task["files_read"] == ["README.md", "calc.py"]
    # Nothing failed: there were no tests to run, and the change is prose.
    assert task["tests_status"] == "no_tests" and task["tests_passed"] is None and task["docs_only"] is True
    assert task["verification"][0]["outcome"] == "no_tests" and task["verification"][0]["exit_code"] == 1
    assert messages(task, "error") == []
    assert messages(task, "warn") == [
        "The model's reply was not a valid tool call (it called 'commentary', which is not one of its tools); asking it again (1/3)",
        "The model's reply was not a valid tool call (it called 'json', which is not one of its tools); asking it again (2/3)",
    ]
    assert "write_file spec.md held back: nothing has been read yet - the agent was asked to look at the repository first" in messages(task, "info")
    assert messages(task, "agent") == ["Added a new spec.md file detailing the project overview and dependencies."]
    assert messages(task, "success")[-1].startswith("Pull request #1 opened: ")
    assert task["log_entries"][-1]["level"] == "success"  # no warning after the pull request line

    body = github.pulls["acme/demo"][0]["body"]
    assert "| ➖ no tests to run |" in body and "❌" not in body and "did not pass" not in body
    assert (
        "This repository has no automated tests to run (the \"test\" script in package.json is npm's "
        "placeholder), so the change is untested. It only touches documentation."
    ) in body
    assert "Review before merging" not in body
    assert "Files the agent looked at: `README.md`, `calc.py`" in body


def test_documentation_change_without_a_test_run_is_not_a_warning(
    client: TestClient, token_env, install_deps, github: FakeGitHub
):
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "README.md"})],
                [("replace_in_file", {"path": "README.md", "old_text": "projct", "new_text": "project"})],
                [("write_file", {"path": "docs/guide.md", "content": "# Guide\n"})],
                [("finish", {"summary": "Fixed a typo and added a guide."})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["tests_status"] == "not_run" and task["docs_only"] is True
    assert messages(task, "warn") == [] and task["log_entries"][-1]["level"] == "success"
    body = github.pulls["acme/demo"][0]["body"]
    assert "No tests were run: this pull request only changes documentation." in body
    assert "⚠️" not in body and "did not run any tests" not in body
    assert "tests were needed" in build_result_comment(task) and "⚠️" not in build_result_comment(task)


def test_code_change_in_a_repository_without_tests_says_so(
    client: TestClient, token_env, install_deps, github: FakeGitHub
):
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("replace_in_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"})],
                [("run_tests", {"command": NPM_PLACEHOLDER_TEST})],
                [("finish", {"summary": "Fixed add()."})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["tests_status"] == "no_tests" and task["docs_only"] is False
    assert task["tests_passed"] is None  # untested is not "passed"...
    assert messages(task, "error") == []  # ...and not a failure either
    assert messages(task, "warn") == [
        "This repository has no automated tests, so the change is untested - the pull request says so. "
        "Review before merging."
    ]
    body = github.pulls["acme/demo"][0]["body"]
    assert "so the change is untested. Review before merging." in body and "| ➖ no tests to run |" in body
    assert "Output of the last run" not in body  # there is no test output worth showing
    assert "has no automated tests, so the change is untested." in build_result_comment(task)


def test_pull_request_says_when_nothing_was_read(client: TestClient, token_env, install_deps, github: FakeGitHub):
    """The model may insist on writing without looking - the reviewer is told."""
    github.add_repo("acme/blank", {"main.c": "int main(void) { return 0; }\n"})  # no manifest, no README
    note = "Notes\n"
    install_deps(
        ScriptedLLM(
            [
                [("write_file", {"path": "NOTES.md", "content": note})],
                [("write_file", {"path": "NOTES.md", "content": note})],
                [("finish", {"summary": "Added notes."})],
            ]
        )
    )
    _events, task = run_stream(client, {**TASK, "repository": "acme/blank"})
    assert task["status"] == "completed" and task["files_read"] == []
    body = github.pulls["acme/blank"][0]["body"]
    assert "⚠️ The agent did not open any file of the repository before making this change." in body
    assert "Files the agent looked at" not in body


def test_model_stuck_on_refused_replies_fails_the_task_with_advice(
    client: TestClient, token_env, install_deps, github: FakeGitHub
):
    install_deps(ScriptedLLM([refused_reply("json")] * 4, model="openai/gpt-oss-120b"))
    _events, task = run_stream(client, TASK)
    assert_honest_failure(task, "agent", "answered 4 times in a row with an invalid tool call", "run it again")
    assert github.calls("POST", r"/git/|/pulls") == []


def test_private_repository_is_cloned_with_a_read_only_token(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    github.add_repo("acme/secret", {"calc.py": "def add(a, b):\n    return a - b\n"}, private=True)
    workspaces: List[LocalWorkspace] = []
    install_deps(fix_add_script(), workspaces)
    response = client.post("/api/v1/tasks/stream", json={"repository": "acme/secret", "prompt": "fix add"})
    task = json.loads(response.text.splitlines()[-1])["task"]
    assert task["status"] == "completed", task["error"]

    by_permission = {json.dumps(info["permissions"], sort_keys=True): token for token, info in github.issued_tokens.items()}
    read_token = by_permission['{"contents": "read"}']
    write_tokens = [token for token in github.issued_tokens if token != read_token]
    assert write_tokens

    workspace = workspaces[0]
    seen_in_sandbox = "\n".join(workspace.commands) + json.dumps(workspace.environments)
    headers = [env["GIT_CONFIG_VALUE_0"] for env in workspace.environments if "GIT_CONFIG_VALUE_0" in env]
    assert len(headers) == 1  # only the clone command
    assert base64.b64decode(headers[0].split("basic ", 1)[1]).decode() == f"x-access-token:{read_token}"
    for token in write_tokens:  # write credentials never enter the sandbox
        assert token not in seen_in_sandbox
        assert base64.b64encode(f"x-access-token:{token}".encode()).decode() not in seen_in_sandbox
    # ...and no token of any kind reaches the client.
    for token in github.issued_tokens:
        assert token not in response.text


def test_secrets_are_redacted_from_logs_and_pull_requests(
    client: TestClient, token_env, install_deps, github: FakeGitHub
):
    leaked = FakeGitHub.PAT
    install_deps(
        ScriptedLLM(
            [
                [("run_command", {"command": f"echo token={leaked}"})],
                [("write_file", {"path": "NOTES.md", "content": "notes\n"})],
                [("finish", {"summary": f"Done. The token is {leaked}.", "pr_title": f"Use {leaked}"})],
            ]
        )
    )
    response = client.post("/api/v1/tasks/stream", json=TASK)
    assert leaked not in response.text
    assert "token=***" in response.text  # the command output is still shown, minus the secret
    pull = github.pulls["acme/demo"][0]
    assert leaked not in pull["title"] and leaked not in pull["body"]
    commit_message = git(github.remote.path_for("acme/demo"), "log", "-1", "--format=%B", pull["head"]["ref"])
    assert leaked not in commit_message and "Task-Id:" in commit_message


def test_every_task_leaves_one_line_in_the_server_log(
    client: TestClient, token_env, install_deps, github: FakeGitHub, caplog
):
    """The server log (on Vercel: the function log) says how each task ended,
    so a run can be diagnosed without the dashboard - and without secrets."""
    key = "gsk_" + "g" * 40

    def summary_for(script: ScriptedLLM) -> str:
        install_deps(script)
        caplog.clear()
        with caplog.at_level("INFO", logger="jules.runner"):
            _events, task = run_stream(client, TASK)
        records = [record for record in caplog.records if record.name == "jules.runner"]
        assert len(records) == 1, [record.getMessage() for record in records]
        line = records[0].getMessage()
        assert line.startswith(f"task {task['task_id'][:12]} on acme/demo finished after ")
        return f"{records[0].levelname} {line}"

    opened = summary_for(fix_add_script())
    assert opened.startswith("INFO ") and opened.endswith(github.pulls["acme/demo"][0]["html_url"])
    unchanged = summary_for(ScriptedLLM([[("finish", {"summary": "Nothing to do."})]]))
    assert unchanged.startswith("INFO ") and unchanged.endswith("no changes, no pull request")
    failed = summary_for(ScriptedLLM([LLMError(f"groq rejected the key {key}", kind="auth", status_code=401)]))
    assert failed.startswith("WARNING ") and key not in failed
    assert "FAILED at stage 'agent': groq rejected the key ***" in failed


# --------------------------------------------------------------------------- #
# Failures are reported as failures
# --------------------------------------------------------------------------- #
def assert_honest_failure(task: Dict[str, Any], stage: str, *fragments: str) -> None:
    assert task["status"] == "failed", task
    assert task["stage"] == stage
    assert task["pr_url"] is None and task["pr_number"] is None and task["outcome"] is None
    assert task["tests_passed"] is not True or task["verification"]
    for fragment in fragments:
        assert fragment in task["error"], task["error"]
    assert messages(task, "error")[-1] == task["error"]
    assert not any("ull request #" in message for message in messages(task))  # no claim of a PR anywhere


def test_unconfigured_server_fails_instead_of_simulating(client: TestClient, install_deps, github: FakeGitHub):
    """Regression for the reported bug: with no credentials at all, the old code
    'completed' the task in 0.1 s with tests_passed=true and a pull request link."""
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    assert_honest_failure(task, "configuration", "not ready to run tasks", "E2B_API_KEY", "LLM API key", "GitHub write credentials")
    assert task["tests_passed"] is None and task["tests_status"] is None
    assert github.requests == []  # nothing was attempted


def test_missing_repository(client: TestClient, token_env, install_deps):
    install_deps(fix_add_script())
    _events, task = run_stream(client, {**TASK, "repository": "acme/nope"})
    assert_honest_failure(task, "github", "acme/nope")


def test_missing_branch_names_the_default(client: TestClient, token_env, install_deps):
    install_deps(fix_add_script())
    _events, task = run_stream(client, {**TASK, "branch": "develop"})
    assert_honest_failure(task, "github", "Branch 'develop' does not exist in acme/demo", "default branch is 'main'")


def test_app_not_installed_gives_install_link(client: TestClient, app_env, install_deps, github: FakeGitHub):
    github.add_repo("acme/elsewhere", {"README.md": "x\n"}, installed=False)
    install_deps(fix_add_script())
    _events, task = run_stream(client, {"repository": "acme/elsewhere", "prompt": "x"})
    assert_honest_failure(task, "github", "not installed", "https://github.com/apps/jules-test-app/installations/new")


def test_read_only_and_archived_repositories(client: TestClient, token_env, install_deps, github: FakeGitHub):
    github.add_repo("acme/readonly", {"README.md": "x\n"}, push=False)
    github.add_repo("acme/old", {"README.md": "x\n"}, archived=True)
    install_deps(fix_add_script())
    _events, task = run_stream(client, {"repository": "acme/readonly", "prompt": "x"})
    assert_honest_failure(task, "github", "cannot push to acme/readonly")
    _events, task = run_stream(client, {"repository": "acme/old", "prompt": "x"})
    assert_honest_failure(task, "github", "archived")


def test_sandbox_failure(client: TestClient, token_env, install_deps):
    deps = install_deps(fix_add_script())

    def broken(_settings, _task_id, _lifetime):
        raise SandboxError("Could not start an E2B sandbox: 401 Unauthorized - check E2B_API_KEY.")

    deps.workspace_factory = broken
    _events, task = run_stream(client, TASK)
    assert_honest_failure(task, "sandbox", "Could not start an E2B sandbox", "E2B_API_KEY")


def test_llm_failure_is_not_papered_over(client: TestClient, token_env, install_deps, github: FakeGitHub):
    """Regression: an LLM error used to fall back silently to a simulated run."""
    install_deps(ScriptedLLM([LLMError("groq returned HTTP 401: Invalid API Key", kind="auth", status_code=401)]))
    _events, task = run_stream(client, TASK)
    assert_honest_failure(task, "agent", "groq returned HTTP 401: Invalid API Key")
    assert github.pulls["acme/demo"] == [] and github.calls("POST", r"/git/") == []


def test_agent_giving_up_discards_partial_work(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(
        ScriptedLLM(
            [
                [("search_code", {"query": "multiply("})],
                [("write_file", {"path": "half.py", "content": "x = 1\n"})],
                [("finish", {"summary": "I could not find the module to change.", "success": False})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert_honest_failure(task, "agent", "could not complete the task", "could not find the module")
    assert any("was discarded" in message for message in messages(task, "warn"))
    assert github.calls("POST", r"/git/|/pulls") == []


def test_pull_request_refused_by_github(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    github.fail_next(
        "POST", r"/repos/acme/demo/pulls$", 403, {"message": "Resource not accessible by integration"}
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "failed" and task["stage"] == "pull_request" and task["pr_url"] is None
    assert "pushed to branch" in task["error"] and "Resource not accessible by integration" in task["error"]
    # The work is not lost: the branch is there and the error links to GitHub's own "open PR" page.
    assert github.remote.rev("acme/demo", f"refs/heads/{task['head_branch']}") == task["commit_sha"]
    assert f"https://github.com/acme/demo/compare/main...{task['head_branch']}?expand=1" in task["error"]


def test_unexpected_errors_fail_the_task(client: TestClient, token_env, install_deps):
    deps = install_deps(fix_add_script())

    def explode(_settings, _task_id, _lifetime):
        raise RuntimeError("kaboom")

    deps.workspace_factory = explode
    _events, task = run_stream(client, TASK)
    assert_honest_failure(task, "sandbox", "Unexpected error (RuntimeError): kaboom")


def test_reported_pull_request_links_always_come_from_github(
    client: TestClient, token_env, install_deps, github: FakeGitHub
):
    """Whatever happens, ``pr_url`` is either empty or the URL GitHub returned for
    a pull request that exists. Never a guess such as ``/pulls`` or ``/pull/42``."""
    scripts = [
        fix_add_script(),
        ScriptedLLM([[("finish", {"summary": "nothing"})]]),
        ScriptedLLM([LLMError("groq returned HTTP 400: bad request", kind="bad_request", status_code=400)]),
        ScriptedLLM(["I am done.", "Really done.", "Nothing needed changing."]),
    ]
    for script in scripts:
        install_deps(script)
        _events, task = run_stream(client, TASK)
        if task["pr_url"] is None:
            assert task["pr_number"] is None and task["outcome"] != "pull_request"
            continue
        assert REAL_PR_URL.match(task["pr_url"])
        assert task["status"] == "completed" and task["outcome"] == "pull_request"
        known = {pull["html_url"]: pull["number"] for pull in github.pulls["acme/demo"]}
        assert known[task["pr_url"]] == task["pr_number"]
    assert len(github.pulls["acme/demo"]) == 1


# --------------------------------------------------------------------------- #
# Background mode, lookups and recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prefix", ["/api/v1/tasks", "/api/tasks"])
def test_background_task_and_polling(client: TestClient, token_env, install_deps, prefix):
    install_deps(fix_add_script())
    created = client.post(prefix, json=TASK)
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "pending" and body["stage"] == "queued" and body["pr_url"] is None
    assert body["logs"] and "Task queued for acme/demo" in body["logs"][0]

    # (The test client runs background work before returning, so it is finished by now.)
    task = client.get(f"{prefix}/{body['task_id']}").json()
    assert task["status"] == "completed" and task["pr_url"] == "https://github.com/acme/demo/pull/1"
    logs = client.get(f"{prefix}/{body['task_id']}/logs").json()
    assert logs["status"] == "completed" and logs["logs"] == task["logs"]
    listing = client.get(prefix).json()
    assert [item["task_id"] for item in listing] == [body["task_id"]]


def test_unknown_task_is_a_404_with_guidance(client: TestClient, token_env, install_deps):
    install_deps(fix_add_script())
    missing = new_task_id()
    response = client.get(f"/api/v1/tasks/{missing}")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "task_not_found"
    assert "?repository=owner/name" in response.json()["detail"]["message"]
    assert client.get(f"/api/v1/tasks/{missing}/logs").status_code == 404
    assert client.get("/api/v1/tasks/not-a-task-id?repository=acme/demo").status_code == 404
    with_repo = client.get(f"/api/v1/tasks/{missing}?repository=acme/demo")
    assert with_repo.status_code == 404 and "no pull request" in with_repo.json()["detail"]["message"]


def test_finished_task_is_recovered_from_github(client: TestClient, token_env, install_deps, github: FakeGitHub):
    """Serverless instances do not share memory: a poll can reach an instance
    that never saw the task. Its pull request is then looked up on GitHub."""
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    task_store.clear()  # "a different instance"

    assert client.get(f"/api/v1/tasks/{task['task_id']}").status_code == 404
    recovered = client.get(f"/api/v1/tasks/{task['task_id']}?repository=acme/demo")
    assert recovered.status_code == 200
    body = recovered.json()
    assert body["recovered"] is True and body["status"] == "completed" and body["outcome"] == "pull_request"
    assert body["unfinished"] is False
    assert (body["pr_url"], body["pr_number"]) == (task["pr_url"], task["pr_number"])
    assert body["head_branch"] == task["head_branch"] and body["base_branch"] == "main"
    # Dashed ids (as older clients stored them) work too.
    dashed = "-".join([task["task_id"][:8], task["task_id"][8:12], task["task_id"][12:16], task["task_id"][16:20], task["task_id"][20:]])
    assert client.get(f"/api/v1/tasks/{dashed}?repository=acme/demo").json()["pr_url"] == task["pr_url"]


def test_recovery_reports_github_problems(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    github.fail_next("GET", r"/repos/acme/demo/pulls$", 500, {"message": "Server Error"})
    response = client.get(f"/api/v1/tasks/{new_task_id()}?repository=acme/demo")
    assert response.status_code == 502 and response.json()["detail"]["code"] == "github_error"


def test_pull_request_state_is_read_from_github(client: TestClient, token_env, install_deps, github: FakeGitHub):
    """A task record says "pull request opened" for good. What became of the
    pull request - merged, closed - is asked from GitHub when someone looks."""
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    url = f"/api/v1/tasks/{task['task_id']}/pull-request"

    opened = client.get(url, params={"repository": "acme/demo"})
    assert opened.status_code == 200
    assert opened.json() == {
        "task_id": task["task_id"],
        "repository": "acme/demo",
        "number": 1,
        "url": "https://github.com/acme/demo/pull/1",
        "state": "open",
        "title": "Fix add()",
        "head_branch": task["head_branch"],
        "base_branch": "main",
        "head_sha": github.remote.rev("acme/demo", f"refs/heads/{task['head_branch']}"),
        "unfinished": False,
        "checks": {"sha": task["commit_sha"], "state": "none", "failing": [], "failed": 0, "passed": 0, "pending": 0, "notes": []},
    }

    pull = github.pulls["acme/demo"][0]
    pull.update(state="closed", merged_at="2026-11-11T15:40:00Z")
    task_store.clear()  # works on an instance that never saw the task, and with a GitHub URL
    merged = client.get(url, params={"repository": "https://github.com/acme/demo"}).json()
    assert merged["state"] == "merged" and merged["number"] == 1

    pull.update(state="closed", merged_at=None)
    assert client.get(url, params={"repository": "acme/demo"}).json()["state"] == "closed"


def test_pull_request_state_errors(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    unknown = client.get(f"/api/v1/tasks/{new_task_id()}/pull-request", params={"repository": "acme/demo"})
    assert unknown.status_code == 404 and unknown.json()["detail"]["code"] == "pull_request_not_found"
    assert client.get("/api/v1/tasks/not-a-task-id/pull-request", params={"repository": "acme/demo"}).status_code == 404
    assert client.get(f"/api/v1/tasks/{new_task_id()}/pull-request").status_code == 422  # repository is required
    bad = client.get(f"/api/v1/tasks/{new_task_id()}/pull-request", params={"repository": "what is this"})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_request"
    github.fail_next("GET", r"/repos/acme/demo/pulls$", 500, {"message": "Server Error"})
    failed = client.get(f"/api/v1/tasks/{new_task_id()}/pull-request", params={"repository": "acme/demo"})
    assert failed.status_code == 502 and failed.json()["detail"]["code"] == "github_error"


def test_pull_request_state_needs_the_access_key(client: TestClient, token_env, install_deps, monkeypatch):
    install_deps(fix_add_script())
    monkeypatch.setenv("JULES_ACCESS_KEY", "k" * 32)
    url = f"/api/v1/tasks/{new_task_id()}/pull-request"
    assert client.get(url, params={"repository": "acme/demo"}).status_code == 401
    with_key = client.get(url, params={"repository": "acme/demo"}, headers={"Authorization": "Bearer " + "k" * 32})
    assert with_key.status_code == 404


# --------------------------------------------------------------------------- #
# GitHub helper endpoints
# --------------------------------------------------------------------------- #
def test_repository_listing(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    github.add_repo("acme/private-one", {"README.md": "x\n"}, private=True)
    body = client.get("/api/v1/github/repos").json()
    assert body["auth_mode"] == "app" and body["error"] is None
    assert body["install_url"] == "https://github.com/apps/jules-test-app/installations/new"
    assert sorted(repo["full_name"] for repo in body["repositories"]) == ["acme/demo", "acme/private-one"]
    demo = next(repo for repo in body["repositories"] if repo["full_name"] == "acme/demo")
    assert demo["default_branch"] == "main" and demo["private"] is False


def test_repository_listing_without_credentials(client: TestClient, install_deps):
    install_deps(fix_add_script())
    body = client.get("/api/v1/github/repos").json()
    assert body == {
        "auth_mode": "none",
        "repositories": [],
        "total": 0,
        "truncated": False,
        "installations": [],
        "install_url": None,
        "error": "No GitHub credentials are configured.",
    }


def test_repository_check(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    ok = client.get("/api/v1/github/check", params={"repository": "https://github.com/acme/demo", "branch": "main"}).json()
    assert ok["ok"] is True and ok["repository"] == "acme/demo" and ok["default_branch"] == "main" and ok["auth_mode"] == "token"
    cases = [
        ({"repository": "acme/demo", "branch": "nope"}, "branch_not_found"),
        ({"repository": "acme/missing"}, "repo_not_found"),
        ({"repository": "what is this"}, "invalid_repository"),
    ]
    for params, code in cases:
        body = client.get("/api/v1/github/check", params=params).json()
        assert body["ok"] is False and body["code"] == code and body["error"], params


def test_github_app_redirects_land_on_the_dashboard(client: TestClient):
    """GitHub sends the browser to the App's Setup/Callback URL after an install.
    Those URLs must not dead-end on a 404."""
    cases = {
        "/api/v1/github/setup?installation_id=42&setup_action=install": "/?github_app=installed&installation_id=42&setup_action=install",
        "/api/v1/github/setup?setup_action=request": "/?github_app=requested&setup_action=request",
        "/api/v1/github/callback?code=abc123&state=xyz": "/?github_app=authorized",
        # URLs this API never defined, as an App might have been configured with:
        "/api/auth/callback/github?code=abc123": "/?github_app=authorized",
        "/api/github/callback?code=abc123&installation_id=7&setup_action=update": "/?github_app=installed&installation_id=7&setup_action=update",
        "/api/v1/webhooks/github?installation_id=9&setup_action=install": "/?github_app=installed&installation_id=9&setup_action=install",
    }
    for path, expected in cases.items():
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 307, path
        assert response.headers["location"] == expected, path
        assert "abc123" not in response.headers["location"]  # the OAuth code is never forwarded
    # Hostile parameter values are not reflected.
    response = client.get("/api/v1/github/setup?installation_id=1%0d%0aX&setup_action=<script>", follow_redirects=False)
    assert response.headers["location"] == "/?github_app=installed"


def test_unknown_api_paths_return_json_404(client: TestClient):
    response = client.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not Found" and response.json()["path"] == "/api/v1/does-not-exist"
    assert client.get("/api/v1/does-not-exist?code=abc", follow_redirects=False).status_code == 404


# --------------------------------------------------------------------------- #
# Webhooks
# --------------------------------------------------------------------------- #
SECRET = "webhook-secret-value"


def deliver(client: TestClient, event: str, payload: Dict[str, Any], *, secret: Optional[str] = SECRET, delivery: str = "d-1"):
    body = json.dumps(payload).encode()
    headers = {"X-GitHub-Event": event, "X-GitHub-Delivery": delivery, "Content-Type": "application/json"}
    if secret:
        headers["X-Hub-Signature-256"] = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post("/api/v1/webhooks/github", content=body, headers=headers)


def comment_event(body: str, association: str = "OWNER", **overrides: Any) -> Dict[str, Any]:
    payload = {
        "action": "created",
        "issue": {"number": 7, "title": "add() is wrong", "body": "It subtracts."},
        "comment": {"body": body, "author_association": association, "user": {"login": "kennedy"}},
        "repository": {"full_name": "acme/demo"},
        "sender": {"login": "kennedy", "type": "User"},
    }
    payload.update(overrides)
    return payload


def test_webhook_signature_and_ping(client: TestClient, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    ping = deliver(client, "ping", {"zen": "Keep it logically awesome."})
    assert ping.status_code == 200 and ping.json()["signature_verified"] is True
    assert deliver(client, "ping", {"zen": "x"}, secret="wrong-secret").status_code == 401
    assert deliver(client, "ping", {"zen": "x"}, secret=None).status_code == 401
    bad = client.post(
        "/api/v1/webhooks/github",
        content=b"{not json",
        headers={"X-GitHub-Event": "push", "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), b"{not json", hashlib.sha256).hexdigest()},
    )
    assert bad.status_code == 400


def test_webhook_command_runs_a_task_and_comments_the_result(
    client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch
):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    llm = fix_add_script()
    install_deps(llm)
    response = deliver(client, "issue_comment", comment_event("Thanks!\n/jules fix it and keep the style"))
    assert response.status_code == 202
    body = response.json()
    assert body["dispatched"] is True and body["task"]["issue_number"] == 7

    task = task_store.get(body["task"]["task_id"])
    assert task["status"] == "completed" and task["source"] == "webhook" and task["issue_number"] == 7
    assert task["pr_url"] == "https://github.com/acme/demo/pull/1"
    prompt = llm.requests[0][1]["content"]
    assert "GitHub issue #7: add() is wrong" in prompt and "It subtracts." in prompt
    assert "Instructions from @kennedy: fix it and keep the style" in prompt
    assert "Refs #7" in github.pulls["acme/demo"][0]["body"]

    (comment,) = github.comments
    assert comment["repository"] == "acme/demo" and comment["issue"] == 7
    assert "I opened #1: https://github.com/acme/demo/pull/1" in comment["body"]
    assert "Tests passed in the sandbox." in comment["body"]

    # GitHub redelivering the same event must not start a second task.
    again = deliver(client, "issue_comment", comment_event("/jules fix it"))
    assert again.status_code == 200 and again.json()["dispatched"] is False
    assert len(task_store.list()) == 1


@pytest.mark.parametrize(
    "tests_status, docs_only, expected, warns",
    [
        ("passed", False, "Tests passed in the sandbox.", False),
        ("failed", False, "The last test run failed - see the pull request.", True),
        ("failed", True, "The last test run failed - see the pull request.", True),  # a failure is never played down
        ("not_run", False, "No tests were run.", True),
        ("not_run", True, "The change only touches documentation, so no tests were needed.", False),
        ("no_tests", False, "The repository has no automated tests, so the change is untested.", False),
        ("no_tests", True, "The change only touches documentation, so no tests were needed.", False),
    ],
)
def test_result_comment_describes_the_testing_honestly(tests_status, docs_only, expected, warns):
    comment = build_result_comment(
        {
            "task_id": "a" * 32,
            "status": "completed",
            "outcome": "pull_request",
            "pr_number": 4,
            "pr_url": "https://github.com/acme/demo/pull/4",
            "summary": "Did it.",
            "tests_status": tests_status,
            "docs_only": docs_only,
        }
    )
    assert expected in comment and ("⚠️" in comment) is warns


def test_webhook_failure_is_commented_honestly(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    install_deps(ScriptedLLM([LLMError("groq returned HTTP 401: Invalid API Key", kind="auth", status_code=401)]))
    issue = {
        "action": "opened",
        "issue": {"number": 3, "title": "Broken", "body": "/jules please fix\nDetails here.", "author_association": "MEMBER", "user": {"login": "dev"}},
        "repository": {"full_name": "acme/demo"},
        "sender": {"login": "dev", "type": "User"},
    }
    assert deliver(client, "issues", issue).status_code == 202
    (comment,) = github.comments
    assert comment["issue"] == 3 and comment["body"].startswith("❌ I could not complete this task.")
    assert "Invalid API Key" in comment["body"] and "/pull/" not in comment["body"]


@pytest.mark.parametrize(
    "event, payload, reason",
    [
        ("issue_comment", comment_event("looks good to me"), "no line starting with /jules"),
        ("issue_comment", comment_event("/jules do it", association="NONE"), "only repository owners"),
        ("issue_comment", comment_event("/jules do it", association="CONTRIBUTOR"), "only repository owners"),
        ("issue_comment", comment_event("/jules do it", sender={"login": "x[bot]", "type": "Bot"}), "bots"),
        # Checks that did not fail, or failed somewhere that is not the agent's to fix.
        ("check_suite", {"action": "completed", "check_suite": {"conclusion": "success", "head_branch": "jules/task-1", "head_sha": "a" * 40}, "repository": {"full_name": "acme/demo"}}, "not a failure"),
        ("check_suite", {"action": "requested", "check_suite": {"head_branch": "jules/task-1", "head_sha": "a" * 40}, "repository": {"full_name": "acme/demo"}}, "are not results"),
        ("check_suite", {"action": "completed", "check_suite": {"conclusion": "failure", "head_branch": "feature/login", "head_sha": "a" * 40}, "repository": {"full_name": "acme/demo"}}, "only fixed unasked on pull requests opened by Jules"),
        ("workflow_run", {"action": "completed", "workflow_run": {"conclusion": "cancelled", "head_branch": "jules/task-1", "head_sha": "a" * 40}, "repository": {"full_name": "acme/demo"}}, "not a failure"),
        ("workflow_run", {"action": "completed", "workflow_run": {"conclusion": "failure", "head_branch": "jules/task-1", "head_sha": "a" * 40, "head_repository": {"full_name": "stranger/demo"}}, "repository": {"full_name": "acme/demo"}}, "fork"),
        ("status", {"state": "success", "sha": "a" * 40, "branches": [{"name": "jules/task-1"}], "repository": {"full_name": "acme/demo"}}, "not a failure"),
        ("status", {"state": "failure", "sha": "a" * 40, "branches": [{"name": "main"}], "context": "Vercel", "repository": {"full_name": "acme/demo"}}, "only fixed unasked"),
        ("status", {"state": "failure", "sha": "a" * 40, "branches": [], "context": "Vercel", "repository": {"full_name": "acme/demo"}}, "did not say which branch"),
        ("issue_comment", comment_event("/julesy do it"), "no line starting with /jules"),
        ("issue_comment", {**comment_event("/jules do it"), "action": "edited"}, "do not start tasks"),
        ("push", {"ref": "refs/heads/main"}, "do not start tasks"),
        ("pull_request", {"action": "opened"}, "do not start tasks"),
    ],
)
def test_webhook_events_that_do_not_start_tasks(client: TestClient, token_env, install_deps, monkeypatch, event, payload, reason):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    install_deps(fix_add_script())
    response = deliver(client, event, payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "received" and body["dispatched"] is False and body["task"] is None
    assert reason in body["reason"]
    assert task_store.list() == []


def test_webhook_without_a_secret_never_starts_tasks(client: TestClient, token_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    response = deliver(client, "issue_comment", comment_event("/jules fix it"), secret=None)
    assert response.status_code == 200 and response.json()["dispatched"] is False
    assert "GITHUB_WEBHOOK_SECRET" in response.json()["reason"]
    assert task_store.list() == [] and github.requests == []
