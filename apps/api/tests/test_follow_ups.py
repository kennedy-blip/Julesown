"""More work on a pull request that exists, failed checks, unfinished work,
commit attribution and keys sent along with a task - end to end against the
local stand-ins, like ``test_api.py``."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.api.services.ci import attempted_fixes, failure_note, fix_marker, handoff_from_commit, standing_instructions
from apps.api.services.github import log_excerpt
from apps.api.services.llm import LLMError
from apps.api.services.store import task_store

from .conftest import CHECK_ADD, DEMO_FILES, fix_add_script
from .support import FakeGitHub, ScriptedLLM
from .test_api import SECRET, TASK, comment_event, deliver, messages, run_stream

ME = {"name": "Ada Example", "email": "1234567+ada-example@users.noreply.github.com"}
BOT = "jules-test-app[bot] <123456+jules-test-app[bot]@users.noreply.github.com> / jules-test-app[bot]"
FINISH = [("finish", {"summary": "Added the notes.", "pr_title": "Add notes"})]
ADD_NOTES = [
    [("read_file", {"path": "calc.py"})],
    [("write_file", {"path": "NOTES.md", "content": "# Notes\n\nadd() adds.\n"})],
    FINISH,
]
ACTIONS_LOG = "\n".join(
    [f"2026-11-11T18:00:{second:02d}.1234567Z step {second}" for second in range(30)]
    + [
        "2026-11-11T18:01:00.0000000Z ##[group]Run npm run build",
        "2026-11-11T18:01:01.0000000Z > vite build",
        "2026-11-11T18:01:02.0000000Z \x1b[31merror during build:\x1b[0m",
        '2026-11-11T18:01:02.1000000Z Could not resolve "./components/AgentSwarm" from "src/App.tsx"',
        "2026-11-11T18:01:03.0000000Z ##[error]Process completed with exit code 1.",
        "2026-11-11T18:01:04.0000000Z Post job cleanup.",
    ]
)


def open_pull_request(client: TestClient, install_deps, github: FakeGitHub) -> Dict[str, Any]:
    """Run the usual task so that acme/demo has pull request #1 from a jules/ branch."""
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    assert task["outcome"] == "pull_request" and task["pr_number"] == 1
    return task


def head_of(github: FakeGitHub, task: Dict[str, Any]) -> str:
    return github.remote.rev("acme/demo", f"refs/heads/{task['head_branch']}")


def prompt_of(llm: ScriptedLLM) -> str:
    return llm.requests[0][1]["content"]


# --------------------------------------------------------------------------- #
# Who a commit is attributed to
# --------------------------------------------------------------------------- #
def test_commits_are_the_apps_unless_asked_otherwise(client: TestClient, app_env, install_deps, github: FakeGitHub):
    task = open_pull_request(client, install_deps, github)
    assert github.authors("acme/demo", task["head_branch"])[0] == BOT
    assert task["commit_authorship"] == "bot"
    # Nothing about who made it is sent: only then does GitHub sign the commit.
    request = next(payload for method, path, payload in github.payloads if path.endswith("/git/commits"))
    assert "author" not in request and "committer" not in request
    assert not github.calls("GET", r"/users/")


def test_commit_as_you(client: TestClient, app_env, install_deps, github: FakeGitHub):
    """The person is the author, the App that made the commit its committer."""
    install_deps(fix_add_script())
    _events, task = run_stream(client, {**TASK, "commit": {"mode": "user", **ME}})
    assert github.authors("acme/demo", task["head_branch"])[0] == f"{ME['name']} <{ME['email']}> / jules-test-app[bot]"
    assert task["commit_authorship"] == "user"
    assert f"The commit is authored as {ME['name']} <{ME['email']}>, committed by jules-test-app[bot]" in messages(task, "info")
    # GitHub would record the author as the committer too, unless the request names one.
    request = next(payload for method, path, payload in github.payloads if path.endswith("/git/commits"))
    assert request["author"] == ME
    assert request["committer"] == {"name": "jules-test-app[bot]", "email": "123456+jules-test-app[bot]@users.noreply.github.com"}


def test_commit_as_you_when_the_apps_account_cannot_be_looked_up(client: TestClient, app_env, install_deps, github: FakeGitHub):
    """The commit is still yours; GitHub then names you as its committer as well."""
    github.fail_next("GET", r"/users/jules-test-app\[bot\]", 404, {"message": "Not Found"})
    install_deps(fix_add_script())
    _events, task = run_stream(client, {**TASK, "commit": {"mode": "user", **ME}})
    assert task["status"] == "completed" and task["commit_authorship"] == "user"
    assert github.authors("acme/demo", task["head_branch"])[0] == f"{ME['name']} <{ME['email']}> / {ME['name']}"
    assert f"The commit is authored as {ME['name']} <{ME['email']}>" in messages(task, "info")


def test_co_authored_commit(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(fix_add_script())
    _events, task = run_stream(client, {**TASK, "commit": {"mode": "coauthor", **ME}})
    assert github.authors("acme/demo", task["head_branch"])[0] == BOT
    request = next(payload for method, path, payload in github.payloads if path.endswith("/git/commits"))
    # Trailers are the last paragraph: that is where GitHub looks for co-authors.
    assert request["message"].endswith(f"\n\nTask-Id: {task['task_id']}\nCo-authored-by: {ME['name']} <{ME['email']}>")
    assert "author" not in request and "committer" not in request  # still the App's own, signed commit
    assert task["commit_authorship"] == "coauthor"


def test_commit_identity_from_the_environment(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("JULES_COMMIT_AUTHORSHIP", "you")
    monkeypatch.setenv("JULES_COMMIT_AUTHOR_NAME", ME["name"])
    monkeypatch.setenv("JULES_COMMIT_AUTHOR_EMAIL", ME["email"])
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    assert github.authors("acme/demo", task["head_branch"])[0] == f"{ME['name']} <{ME['email']}> / Token Owner"
    health = client.get("/api/v1/health/env").json()
    assert health["commits"] == {"authorship": "user", "author_configured": True, "author_name": ME["name"]}
    # What the request says wins over the environment.
    install_deps(ScriptedLLM(ADD_NOTES))
    _events, other = run_stream(client, {**TASK, "commit": {"mode": "bot"}})
    assert github.authors("acme/demo", other["head_branch"])[0].startswith("Token Owner <owner@example.com>")


@pytest.mark.parametrize(
    "commit",
    [
        {"mode": "user"},
        {"mode": "user", "name": "Ada", "email": "not an address"},
        {"mode": "user", "name": "   ", "email": ME["email"]},
        {"mode": "coauthor", "email": ME["email"]},
    ],
)
def test_a_person_who_is_not_named_cannot_be_the_author(client: TestClient, app_env, install_deps, github: FakeGitHub, commit):
    install_deps(fix_add_script())
    _events, task = run_stream(client, {**TASK, "commit": commit})
    assert task["status"] == "completed" and task["commit_authorship"] == "bot"
    assert github.authors("acme/demo", task["head_branch"])[0] == BOT
    assert any("no name and e-mail address are configured" in line for line in messages(task, "info"))


def test_first_commit_of_an_empty_repository_is_yours_too(client: TestClient, app_env, install_deps, github: FakeGitHub):
    github.add_repo("acme/blank", None, empty=True)
    install_deps(
        ScriptedLLM(
            [
                [
                    ("write_file", {"path": "index.html", "content": "<!doctype html>\n<title>Hi</title>\n"}),
                    ("write_file", {"path": "README.md", "content": "# Hi\n\nOpen index.html.\n"}),
                ],
                [("finish", {"summary": "A page.", "pr_title": "Create the page"})],
            ]
        )
    )
    _events, task = run_stream(client, {"repository": "acme/blank", "prompt": "Build a page", "commit": {"mode": "user", **ME}})
    assert task["outcome"] == "initial_commit"
    # Both commits: the one that creates the branch and the one with the project.
    assert github.authors("acme/blank", "main") == [f"{ME['name']} <{ME['email']}> / jules-test-app[bot]"] * 2


def test_task_started_from_github_can_be_committed_as_whoever_asked(
    client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch
):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("JULES_COMMIT_AUTHORSHIP", "user")  # and no name: it is the requester
    install_deps(fix_add_script())
    event = comment_event("/jules fix it")
    event["comment"]["user"] = {"login": "ada-example", "id": 1234567}
    task_id = deliver(client, "issue_comment", event).json()["task"]["task_id"]
    task = task_store.get(task_id)
    assert github.authors("acme/demo", task["head_branch"])[0] == (
        "ada-example <1234567+ada-example@users.noreply.github.com> / Token Owner"
    )


# --------------------------------------------------------------------------- #
# More work on a pull request that exists
# --------------------------------------------------------------------------- #
def test_follow_up_is_added_to_the_pull_request(client: TestClient, app_env, install_deps, github: FakeGitHub):
    first = open_pull_request(client, install_deps, github)
    before, main_before = head_of(github, first), github.remote.rev("acme/demo", "refs/heads/main")
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    events, task = run_stream(client, {"repository": "acme/demo", "prompt": "Add a NOTES.md", "pull_request": 1})

    assert task["status"] == "completed" and task["outcome"] == "pr_updated"
    assert (task["pr_number"], task["pr_url"]) == (1, "https://github.com/acme/demo/pull/1")
    assert task["pull_request"] == 1 and task["head_branch"] == first["head_branch"] and task["base_branch"] == "main"
    assert task["branch"] == first["head_branch"]  # where the work started from
    head = head_of(github, first)
    assert head == task["commit_sha"] != before and task["commit_url"] == f"https://github.com/acme/demo/commit/{head}"
    # One new commit on top of the old head; main and the list of pull requests are as they were.
    assert github.remote.log("acme/demo", first["head_branch"])[:2] == ["Add notes", "Fix add()"]
    assert github.remote.rev("acme/demo", "refs/heads/main") == main_before and len(github.pulls["acme/demo"]) == 1
    assert github.remote.show("acme/demo", first["head_branch"], "NOTES.md") == "# Notes\n\nadd() adds."
    # No new branch: the existing one is moved forward, and never by force.
    moves = [(path, payload) for method, path, payload in github.payloads if method == "PATCH" and "/git/refs/heads/" in path]
    assert moves == [(f"/repos/acme/demo/git/refs/heads/{first['head_branch']}", {"sha": head, "force": False})]

    # The agent knows what it is working on and how its work is delivered.
    assert "Add a NOTES.md" in prompt_of(llm)
    assert f'You are working on pull request #1 "Fix add()" (branch `{first["head_branch"]}`, to be merged into `main`)' in prompt_of(llm)
    system = llm.requests[0][0]["content"]
    assert "adds your working tree as a new commit to the existing pull request" in system
    assert "opens the pull request for you" not in system

    # The pull request is told what was pushed.
    (comment,) = [item for item in github.comments if item["issue"] == 1]
    assert comment["body"].startswith(f"✅ I pushed [`{head[:10]}`](https://github.com/acme/demo/commit/{head}).")
    assert "Added the notes." in comment["body"] and "Changed: `NOTES.md`" in comment["body"]
    assert f"Pull request #1 updated with commit {head[:10]}: https://github.com/acme/demo/pull/1" in messages(task, "success")
    stages = [event["fields"]["stage"] for event in events if event["type"] == "update" and "stage" in event["fields"]]
    assert stages == ["configuration", "github", "sandbox", "clone", "agent", "commit", "pull_request", "done"]


def test_follow_up_is_recovered_from_github(client: TestClient, app_env, install_deps, github: FakeGitHub):
    """Another server instance answers the lookup: it finds the commit on the pull request."""
    first = open_pull_request(client, install_deps, github)
    install_deps(ScriptedLLM(ADD_NOTES))
    _events, task = run_stream(client, {"repository": "acme/demo", "prompt": "Add a NOTES.md", "pull_request": 1})
    task_store.clear()
    url = f"/api/v1/tasks/{task['task_id']}"

    # A follow-up has no branch named after it, so without the number there is nothing to go by.
    assert client.get(url, params={"repository": "acme/demo"}).status_code == 404
    found = client.get(url, params={"repository": "acme/demo", "pull_request": 1}).json()
    assert found["recovered"] is True and found["status"] == "completed" and found["outcome"] == "pr_updated"
    assert (found["pr_number"], found["pull_request"], found["pr_url"]) == (1, 1, "https://github.com/acme/demo/pull/1")
    assert found["commit_sha"] == task["commit_sha"] and found["commit_url"] == task["commit_url"]
    assert found["head_branch"] == first["head_branch"] and found["summary"] == "Add notes" and found["unfinished"] is False
    assert found["logs"] == [f"Recovered from GitHub: commit {task['commit_sha'][:10]} on pull request #1"]

    # The commit is still found once the pull request is merged and its branch deleted.
    pull = github.pulls["acme/demo"][0]
    pull.update(state="closed", merged_at="2026-11-11T20:00:00Z")
    pull["head"]["sha"] = task["commit_sha"]  # GitHub keeps the head it was merged with
    github.remote.delete_branch("acme/demo", first["head_branch"])
    assert client.get(url, params={"repository": "acme/demo", "pull_request": 1}).json()["commit_sha"] == task["commit_sha"]
    # The task that opened the pull request is not mistaken for one that added to it, and a
    # follow-up that never pushed is still unknown.
    assert client.get(f"/api/v1/tasks/{'0' * 32}", params={"repository": "acme/demo", "pull_request": 1}).status_code == 404
    assert client.get(url, params={"repository": "acme/demo", "pull_request": 99}).status_code == 404


def test_follow_up_on_a_pull_request_that_cannot_take_it(client: TestClient, app_env, install_deps, github: FakeGitHub):
    open_pull_request(client, install_deps, github)
    pull = github.pulls["acme/demo"][0]
    follow_up = {"repository": "acme/demo", "prompt": "More", "pull_request": 1}

    def fails_with(fragment: str) -> None:
        install_deps(ScriptedLLM([]))
        _events, task = run_stream(client, follow_up)
        assert task["status"] == "failed" and task["stage"] == "github" and fragment in task["error"]

    pull.update(state="closed")
    fails_with("Pull request #1 is closed, so nothing can be added to it.")
    pull.update(merged_at="2026-11-11T19:00:00Z")
    fails_with("Pull request #1 is merged")
    pull.update(state="open", merged_at=None, head_repository="stranger/demo")
    fails_with("comes from a fork (stranger/demo)")
    install_deps(ScriptedLLM([]))
    _events, task = run_stream(client, {**follow_up, "pull_request": 99})
    assert task["status"] == "failed" and "acme/demo has no pull request #99" in task["error"]


def test_branch_that_moved_meanwhile_is_not_overwritten(client: TestClient, app_env, install_deps, github: FakeGitHub):
    first = open_pull_request(client, install_deps, github)
    bare = github.remote.path_for("acme/demo")

    def someone_pushes(_messages: Any) -> Any:
        # While the agent works, a person adds a commit to the pull request.
        from .support import git

        tree = git(bare, "rev-parse", f"refs/heads/{first['head_branch']}^{{tree}}")
        commit = git(bare, "commit-tree", tree, "-p", head_of(github, first), "-m", "Review fix by a person")
        git(bare, "update-ref", f"refs/heads/{first['head_branch']}", commit)
        return FINISH

    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], someone_pushes]))
    _events, task = run_stream(client, {"repository": "acme/demo", "prompt": "Add notes", "pull_request": 1})
    assert task["status"] == "failed" and task["stage"] == "pull_request"
    assert "changed while the agent was working, so its commit was not pushed over that" in task["error"]
    assert github.remote.log("acme/demo", first["head_branch"])[0] == "Review fix by a person"
    assert [item for item in github.comments if item["issue"] == 1] == []


def test_follow_up_that_changes_nothing(client: TestClient, app_env, install_deps, github: FakeGitHub):
    first = open_pull_request(client, install_deps, github)
    before = head_of(github, first)
    install_deps(ScriptedLLM([[("read_file", {"path": "calc.py"})], [("finish", {"summary": "It is fine as it is."})]]))
    _events, task = run_stream(client, {"repository": "acme/demo", "prompt": "Check it", "pull_request": 1})
    assert task["status"] == "completed" and task["outcome"] == "no_changes" and task["pr_number"] == 1
    assert "Finished without changing any files, so nothing was added to pull request #1." in messages(task, "success")
    assert head_of(github, first) == before


# --------------------------------------------------------------------------- #
# Failed checks
# --------------------------------------------------------------------------- #
def break_the_build(github: FakeGitHub, sha: str) -> None:
    github.add_check_run(
        "acme/demo",
        sha,
        "build",
        "failure",
        title="Build failed",
        annotations=[
            {"path": "src/App.tsx", "start_line": 4, "annotation_level": "failure", "message": 'Could not resolve "./components/AgentSwarm"'},
            {"path": ".github", "start_line": 1, "annotation_level": "failure", "message": "Process completed with exit code 1."},
            {"path": "a.js", "start_line": 1, "annotation_level": "notice", "message": "just so you know"},
        ],
        log=ACTIONS_LOG,
    )
    github.add_check_run("acme/demo", sha, "lint", "success")
    github.add_status("acme/demo", sha, "Vercel", "failure", "Deployment has failed", "https://vercel.com/acme/demo/abc")


def test_failed_checks_become_the_task(client: TestClient, app_env, install_deps, github: FakeGitHub):
    first = open_pull_request(client, install_deps, github)
    failed_sha = head_of(github, first)
    break_the_build(github, failed_sha)
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"})],
            [("write_file", {"path": "agent_swarm.py", "content": "SWARM = []\n"})],
            [("run_tests", {"command": CHECK_ADD})],
            [("finish", {"summary": "The module the build imports was missing; added it.", "pr_title": "Add the missing module"})],
        ]
    )
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/demo", "pull_request": 1, "fix_checks": True})

    assert task["status"] == "completed" and task["outcome"] == "pr_updated" and task["fix_checks"] is True
    assert task["prompt"] == "Fix the failed checks of pull request #1"
    assert task["failed_checks"] == [
        {"name": "build", "kind": "check_run", "url": "https://github.com/acme/demo/actions/runs/1/job/9001"},
        {"name": "Vercel", "kind": "status", "url": "https://vercel.com/acme/demo/abc"},
    ]
    assert f"Failed checks on {failed_sha[:10]}: build, Vercel" in messages(task, "info")

    prompt = prompt_of(llm)
    assert f'The automated checks fail on pull request #1 "Fix add()" (branch `{first["head_branch"]}`) (commit {failed_sha[:10]}).' in prompt
    assert "1. build (GitHub Actions): failure - Build failed" in prompt
    assert '   - src/App.tsx:4: Could not resolve "./components/AgentSwarm"' in prompt
    assert "   - Process completed with exit code 1." in prompt and "just so you know" not in prompt
    # The log: the part around the failure, without timestamps and colour codes.
    assert '   | Could not resolve "./components/AgentSwarm" from "src/App.tsx"' in prompt
    assert "   | ##[error]Process completed with exit code 1." in prompt and "   | error during build:" in prompt
    assert "2026-11-11T18" not in prompt and "\x1b[" not in prompt
    # What a commit cannot fix is to be reported, not papered over.
    assert "If the cause is not in the repository" in prompt and "say in the summary what has to be set up" in prompt
    # A commit status has no log: the agent is told to reproduce it.
    assert "2. Vercel (commit status): failure - Deployment has failed" in prompt
    assert "GitHub has no log for this check (its page: https://vercel.com/acme/demo/abc). Reproduce the failure in the sandbox" in prompt
    assert "Do not delete, skip or weaken tests and checks to make them pass" in prompt

    head = head_of(github, first)
    comment = [item for item in github.comments if item["issue"] == 1][-1]["body"]
    assert comment.startswith(f"🔧 I pushed [`{head[:10]}`](https://github.com/acme/demo/commit/{head}) to fix the failed checks (build, Vercel).")
    assert "Last test run in the sandbox:" in comment and "✅ passed" in comment
    assert "GitHub runs the checks again for the new commit; their result shows on the pull request." in messages(task, "info")


def test_log_download_does_not_carry_the_api_token(client: TestClient, app_env, install_deps, github: FakeGitHub):
    """GitHub redirects a log download to a storage host with a signed URL.
    The installation token must not travel there."""
    first = open_pull_request(client, install_deps, github)
    break_the_build(github, head_of(github, first))
    install_deps(ScriptedLLM([[("finish", {"summary": "Nothing I can do."})]]))
    run_stream(client, {"repository": "acme/demo", "pull_request": 1, "fix_checks": True})
    (download,) = github.log_downloads
    assert download.url.host == FakeGitHub.LOG_HOST and "authorization" not in download.headers


def test_nothing_failed_nothing_to_fix(client: TestClient, app_env, install_deps, github: FakeGitHub):
    first = open_pull_request(client, install_deps, github)
    sha = head_of(github, first)
    request = {"repository": "acme/demo", "pull_request": 1, "fix_checks": True}

    def nothing_to_fix(expected: str) -> None:
        deps = install_deps(ScriptedLLM([]))
        _events, task = run_stream(client, request)
        assert task["status"] == "completed" and task["outcome"] == "no_changes"
        assert messages(task, "success") == [f"Nothing to fix: no check has failed on {sha[:10]} ({expected})."]
        assert deps.settings_seen == [] and task["sandbox"] is None  # decided before anything is started

    nothing_to_fix("it has no checks")
    github.add_check_run("acme/demo", sha, "build", "success")
    github.add_check_run("acme/demo", sha, "e2e", None)
    github.add_check_run("acme/demo", sha, "superseded", "cancelled")
    github.add_status("acme/demo", sha, "Vercel", "pending")
    nothing_to_fix("2 passed, 2 still running")


def test_checks_the_app_may_not_read(client: TestClient, app_env, install_deps, github: FakeGitHub):
    first = open_pull_request(client, install_deps, github)
    github.add_check_run("acme/demo", head_of(github, first), "build", "failure")
    github.unreadable = {"checks", "statuses"}
    install_deps(ScriptedLLM([]))
    _events, task = run_stream(client, {"repository": "acme/demo", "pull_request": 1, "fix_checks": True})
    assert task["status"] == "failed" and task["stage"] == "github"
    assert "GitHub App permission 'Checks: Read-only' is missing" in task["error"]
    assert "GitHub App permission 'Commit statuses: Read-only' is missing" in task["error"]

    # One readable source is enough to work with; the other is mentioned.
    github.unreadable = {"statuses"}
    install_deps(ScriptedLLM([[("finish", {"summary": "Looked."})]]))
    _events, task = run_stream(client, {"repository": "acme/demo", "pull_request": 1, "fix_checks": True})
    assert task["status"] == "completed"
    assert any("Commit statuses could not be read" in line for line in messages(task, "warn"))


def test_failed_checks_of_a_branch_are_fixed_in_a_new_pull_request(client: TestClient, app_env, install_deps, github: FakeGitHub):
    """For example the first commit of a new project whose deployment fails."""
    main = github.remote.rev("acme/demo", "refs/heads/main")
    github.add_status("acme/demo", main, "Vercel", "error", "Build failed")
    llm = fix_add_script()
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/demo", "fix_checks": True, "prompt": "It is the add function."})
    assert task["outcome"] == "pull_request" and task["pr_number"] == 1 and task["pull_request"] is None
    assert github.remote.rev("acme/demo", "refs/heads/main") == main
    prompt = prompt_of(llm)
    assert f"The automated checks fail on branch `main` (commit {main[:10]})." in prompt
    assert "1. Vercel (commit status): error - Build failed" in prompt
    assert "Additional instructions:\nIt is the add function." in prompt
    assert task["prompt"] == "Fix the failed checks of branch default\n\nIt is the add function."

    github.add_repo("acme/blank", None, empty=True)
    install_deps(ScriptedLLM([]))
    _events, task = run_stream(client, {"repository": "acme/blank", "fix_checks": True})
    assert task["status"] == "failed" and "it has no commits and therefore no checks to fix" in task["error"]


def test_log_excerpt():
    excerpt = log_excerpt(ACTIONS_LOG)
    assert excerpt.splitlines()[-1] == "Post job cleanup." and "##[error]Process completed with exit code 1." in excerpt
    assert "step 0" in excerpt and "2026-" not in excerpt and "\x1b" not in excerpt
    # A long log: the lines before the failure, not the beginning.
    long_log = "\n".join(f"line {number}" for number in range(2000)) + "\n##[error]boom\nafter 1\nafter 2\nafter 3\nafter 4"
    lines = log_excerpt(long_log).splitlines()
    assert "line 0" not in lines and "line 1999" in lines and "##[error]boom" in lines
    assert lines[-1] == "after 3" and len(lines) <= 140
    assert log_excerpt("x" * 50_000).startswith("[...]\n") and len(log_excerpt("x" * 50_000)) < 5100
    assert log_excerpt("no marker\njust the end") == "no marker\njust the end" and log_excerpt("") == ""


# --------------------------------------------------------------------------- #
# Work that was cut short
# --------------------------------------------------------------------------- #
RATE_LIMIT = LLMError(
    "groq returned HTTP 429: Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
    "Limit 8000, Used 7441, Requested 4788. Please try again in 31.7175s.",
    kind="rate_limit",
    status_code=429,
    retry_after=900.0,
)


def test_work_interrupted_by_the_provider_is_kept_as_an_unfinished_pull_request(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    """Production, 2026-11-11: twelve files written, then the provider's rate
    limit ended the run and everything was thrown away."""
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    _events, task = run_stream(client, TASK)

    assert task["status"] == "completed" and task["outcome"] == "pull_request"
    assert task["unfinished"] is True and task["stop_reason"] == "llm_error"
    assert task["unfinished_reason"].startswith("groq returned HTTP 429")  # why, for the dashboard
    pull = github.pulls["acme/demo"][0]
    assert pull["title"] == "[Unfinished] add() returns the wrong result, fix it"
    assert pull["body"].startswith("> [!WARNING]\n> **Unfinished.** The agent was stopped before it was done: groq returned HTTP 429")
    assert "do not merge it as it is" in pull["body"]
    assert github.remote.show("acme/demo", task["head_branch"], "NOTES.md").startswith("# Notes")
    commit = next(payload for method, path, payload in github.payloads if path.endswith("/git/commits"))["message"]
    assert commit.startswith("[Unfinished] add() returns") and "Unfinished: the agent was stopped before it was done." in commit
    warnings = messages(task, "warn")
    assert any(line.startswith("The agent was stopped before it was done (groq returned HTTP 429") for line in warnings)
    assert warnings[-1].startswith("The pull request is marked as unfinished. Start a follow-up on it")
    assert not any("discarded" in line for line in warnings)
    # A server instance that only has GitHub to go by still knows the work is unfinished.
    task_store.clear()
    found = client.get(f"/api/v1/tasks/{task['task_id']}", params={"repository": "acme/demo"}).json()
    assert found["recovered"] is True and found["outcome"] == "pull_request" and found["unfinished"] is True


def test_unfinished_new_project_is_pushed_so_that_it_can_be_continued(client: TestClient, app_env, install_deps, github: FakeGitHub):
    github.add_repo("acme/blank", None, empty=True)
    install_deps(
        ScriptedLLM(
            [
                [
                    ("write_file", {"path": "package.json", "content": '{"name": "omni-pulse", "version": "0.1.0"}\n'}),
                    ("write_file", {"path": "src/App.tsx", "content": "export default function App() { return null; }\n"}),
                ],
                RATE_LIMIT,
            ]
        )
    )
    _events, task = run_stream(client, {"repository": "acme/blank", "prompt": "Build OmniPulse"})
    assert task["status"] == "completed" and task["outcome"] == "initial_commit" and task["unfinished"] is True
    assert github.remote.files("acme/blank", "main") == ["package.json", "src/App.tsx"]
    assert github.remote.log("acme/blank", "main")[0] == "[Unfinished] Build OmniPulse"
    assert any("The project is unfinished" in line and "Run another task" in line for line in messages(task, "warn"))
    task_store.clear()
    found = client.get(f"/api/v1/tasks/{task['task_id']}", params={"repository": "acme/blank"}).json()
    assert found["recovered"] is True and found["outcome"] == "initial_commit" and found["unfinished"] is True


def test_continuing_finishes_the_pull_request_and_takes_the_mark_off(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    _events, first = run_stream(client, TASK)
    assert github.pulls["acme/demo"][0]["title"].startswith("[Unfinished] ")

    status = client.get(f"/api/v1/tasks/{first['task_id']}/pull-request", params={"repository": "acme/demo"}).json()
    assert status["unfinished"] is True and status["state"] == "open"

    llm = fix_add_script()
    install_deps(llm)
    _events, task = run_stream(
        client, {"repository": "acme/demo", "prompt": TASK["prompt"], "pull_request": 1, "continue_work": True}
    )
    assert task["outcome"] == "pr_updated" and task["unfinished"] is False and task["continue_work"] is True
    assert github.pulls["acme/demo"][0]["title"] == "add() returns the wrong result, fix it"
    assert "Pull request #1 is no longer marked as unfinished" in messages(task, "info")
    prompt = prompt_of(llm)
    assert "A previous run started this work and stopped before it was finished." in prompt
    assert 'pull request #1 "add() returns the wrong result, fix it"' in prompt  # without the mark
    # A follow-up that is cut short as well leaves the mark where it is.
    install_deps(ScriptedLLM([[("read_file", {"path": "calc.py"})], [("write_file", {"path": "MORE.md", "content": "more\n"})], RATE_LIMIT]))
    _events, again = run_stream(client, {"repository": "acme/demo", "prompt": "More", "pull_request": 1})
    assert again["outcome"] == "pr_updated" and again["unfinished"] is True
    assert github.pulls["acme/demo"][0]["title"] == "[Unfinished] add() returns the wrong result, fix it"
    assert [item for item in github.comments if item["issue"] == 1][-1]["body"].startswith("⚠️ I pushed")


def test_continuing_a_project_that_was_pushed_unfinished(client: TestClient, app_env, install_deps, github: FakeGitHub):
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/demo", "prompt": "Build OmniPulse", "continue_work": True})
    assert task["outcome"] == "pull_request"
    assert prompt_of(llm).count("A previous run started this work and stopped before it was finished.") == 1
    assert "do not start over" in prompt_of(llm) and "What the previous run left behind" not in prompt_of(llm)


BUILD = "echo \"vite.config.ts(8,18): error TS2580: Cannot find name 'process'.\" >&2; exit 2"
ACCOUNT = (
    "I scaffolded the app: package.json, vite.config.ts and src/App.tsx. What is missing: the build does "
    "not pass yet - vite.config.ts needs @types/node. Nothing was committed."
)
CUT_SHORT = [
    [
        ("write_file", {"path": "package.json", "content": '{"name": "omni-pulse", "version": "1.0.0", "scripts": {"build": "tsc -b && vite build"}}\n'}),
        ("write_file", {"path": "vite.config.ts", "content": "export default { base: process.env.BASE };\n"}),
        ("write_file", {"path": "src/App.tsx", "content": "export default function App() { return null; }\n"}),
        ("write_file", {"path": "README.md", "content": "# OmniPulse\n"}),
        ("write_file", {"path": ".gitignore", "content": "node_modules\ndist\n"}),
        ("run_command", {"command": BUILD}),
    ],
    # Asked to wrap up, the agent says - truthfully - that the work is not complete.
    [("finish", {"summary": ACCOUNT, "success": False})],
]


def test_work_cut_short_by_the_clock_is_pushed_and_says_where_it_stands(
    client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch
):
    """Production, 2026-11-12, an empty repository: 27 files written, the first
    build failed as the time ran out, the agent - asked to wrap up - reported
    that the build does not pass, and everything it had written was thrown
    away. Now the project is pushed, marked as unfinished, and the task says
    what a follow-up run needs to know."""
    # With 35 s for the agent the wrap-up is due after its first reply.
    monkeypatch.setenv("JULES_TASK_BUDGET_SECONDS", "60")
    github.add_repo("acme/blank", None, empty=True)
    llm = ScriptedLLM(CUT_SHORT)
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/blank", "prompt": "Build OmniPulse"})

    assert "Time is almost up for this run" in llm.requests[1][-1]["content"]
    assert task["status"] == "completed" and task["outcome"] == "initial_commit"
    assert task["unfinished"] is True and task["stop_reason"] == "timeout"
    assert task["unfinished_reason"] == "The time budget ran out before the agent finished."
    assert task["summary"] == ACCOUNT
    assert github.remote.files("acme/blank", "main") == [".gitignore", "README.md", "package.json", "src/App.tsx", "vite.config.ts"]
    assert github.remote.log("acme/blank", "main")[0] == "[Unfinished] Build OmniPulse"
    log = messages(task)
    assert "The time budget is nearly used up: the agent was asked to wrap up" in log
    assert not any("discarded" in line for line in log) and not any("could not complete the task" in line for line in log)
    assert (
        "The agent was stopped before it was done (The time budget ran out before the agent finished). "
        "What it changed until then is kept and delivered marked as unfinished."
    ) in messages(task, "warn")
    # What the next run starts from: the agent's account, the command that fails, why it stopped.
    assert task["handoff"] == (
        f"Its own account when it stopped:\n{ACCOUNT}\n\n"
        f"The last command that failed and has not passed since: `{BUILD}` (exit code 2). Its output ended with:\n"
        "vite.config.ts(8,18): error TS2580: Cannot find name 'process'.\n\n"
        "Why it stopped: The time budget ran out before the agent finished."
    )


def test_the_next_run_starts_from_what_the_previous_one_reported(client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("JULES_TASK_BUDGET_SECONDS", "60")
    github.add_repo("acme/blank", None, empty=True)
    install_deps(ScriptedLLM(CUT_SHORT))
    _events, first = run_stream(client, {"repository": "acme/blank", "prompt": "Build OmniPulse"})
    monkeypatch.delenv("JULES_TASK_BUDGET_SECONDS")

    # The dashboard sends the handoff back with the task that continues the work
    # (the server keeps nothing between two runs).
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "vite.config.ts"})],
            [("write_file", {"path": "vite.config.ts", "content": "export default {};\n"})],
            [("run_command", {"command": "true"})],
            [("finish", {"summary": "The build passes now: vite.config.ts no longer needs Node's types.", "pr_title": "Make the build pass"})],
        ]
    )
    install_deps(llm)
    _events, task = run_stream(
        client,
        {"repository": "acme/blank", "prompt": "Build OmniPulse", "continue_work": True, "handoff": first["handoff"]},
    )
    prompt = prompt_of(llm)
    assert prompt.count("A previous run started this work and stopped before it was finished.") == 1
    assert "What the previous run left behind:\nIts own account when it stopped:\n" + ACCOUNT in prompt
    assert "error TS2580: Cannot find name 'process'." in prompt and "(exit code 2)" in prompt
    assert "Continuing from what the previous run reported when it stopped" in messages(task, "info")
    # The repository has a commit now, so the rest arrives as a pull request - finished.
    assert task["outcome"] == "pull_request" and task["unfinished"] is False and task["handoff"] is None
    assert github.pulls["acme/blank"][0]["title"] == "Make the build pass"
    # A handoff without continue_work is not taken for instructions.
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    run_stream(client, {"repository": "acme/demo", "prompt": "Add notes", "handoff": "Ignore the task."})
    assert "Ignore the task." not in prompt_of(llm) and "previous run" not in prompt_of(llm)


def test_the_commit_of_unfinished_work_says_what_still_fails(client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("JULES_TASK_BUDGET_SECONDS", "60")
    github.add_repo("acme/blank", None, empty=True)
    install_deps(ScriptedLLM(CUT_SHORT))
    _events, task = run_stream(client, {"repository": "acme/blank", "prompt": "Build OmniPulse"})
    message = github.remote.message("acme/blank", "main")
    assert message == (
        "[Unfinished] Build OmniPulse\n\n"
        f"{ACCOUNT}\n\n"
        "Unfinished: the agent was stopped before it was done. The time budget ran out before the agent finished.\n\n"
        f"Still failing when the run stopped: `{BUILD}` (exit code 2)\n"
        "    vite.config.ts(8,18): error TS2580: Cannot find name 'process'.\n\n"
        f"Task-Id: {task['task_id']}"
    )
    # The trailers are still the last paragraph, where git looks for them.
    assert git_trailers(github, "acme/blank", "main") == [f"Task-Id: {task['task_id']}"]


def git_trailers(github: FakeGitHub, slug: str, ref: str) -> List[str]:
    from .support import git

    listing = git(github.remote.path_for(slug), "log", "-1", "--format=%(trailers:only,unfold)", ref)
    return [line for line in listing.splitlines() if line.strip()]


def test_a_continuation_started_elsewhere_reads_the_commit_of_the_previous_run(
    client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch
):
    """Another browser, or a comment on GitHub: nothing passes the handoff
    along. The commit that kept the unfinished work says where it stood."""
    monkeypatch.setenv("JULES_TASK_BUDGET_SECONDS", "60")
    github.add_repo("acme/blank", None, empty=True)
    install_deps(ScriptedLLM(CUT_SHORT))
    run_stream(client, {"repository": "acme/blank", "prompt": "Build OmniPulse"})
    monkeypatch.delenv("JULES_TASK_BUDGET_SECONDS")
    task_store.clear()

    llm = ScriptedLLM(
        [
            [("read_file", {"path": "vite.config.ts"})],
            [("write_file", {"path": "vite.config.ts", "content": "export default {};\n"})],
            [("run_command", {"command": "true"})],
            [("finish", {"summary": "The build passes now.", "pr_title": "Make the build pass"})],
        ]
    )
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/blank", "prompt": "Build OmniPulse", "continue_work": True})
    prompt = prompt_of(llm)
    assert prompt.count("A previous run started this work and stopped before it was finished.") == 1
    assert f"What the previous run left behind:\n{ACCOUNT}\n\nUnfinished: the agent was stopped before it was done." in prompt
    assert f"Still failing when the run stopped: `{BUILD}` (exit code 2)\n    vite.config.ts(8,18): error TS2580" in prompt
    assert "Task-Id:" not in prompt and "[Unfinished]" not in prompt
    assert "Continuing from what the commit of the previous run says about where it stopped" in messages(task, "info")
    assert task["outcome"] == "pull_request" and task["unfinished"] is False

    # A commit that is not the unfinished work of a task says nothing of the kind.
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/demo", "prompt": "Carry on", "continue_work": True})
    assert "What the previous run left behind" not in prompt_of(llm)
    assert not any("previous run" in line for line in messages(task, "info"))


def test_what_a_commit_says_about_unfinished_work():
    message = (
        "[Unfinished] Build OmniPulse\n\nI scaffolded the app.\n\nIt has six panels.\n\n"
        "Unfinished: the agent was stopped before it was done. The time budget ran out.\n\n"
        "Still failing when the run stopped: `npm run build` (exit code 2)\n    error TS2580\n\n"
        "Task-Id: abc123\nCo-authored-by: Ada Example <1234567+ada-example@users.noreply.github.com>"
    )
    assert handoff_from_commit(message) == (
        "I scaffolded the app.\n\nIt has six panels.\n\n"
        "Unfinished: the agent was stopped before it was done. The time budget ran out.\n\n"
        "Still failing when the run stopped: `npm run build` (exit code 2)\n    error TS2580"
    )
    # Not unfinished work, not a commit of a task, or nothing said about it.
    assert handoff_from_commit(message[len("[Unfinished] ") :]) == ""
    assert handoff_from_commit(message.replace("Task-Id: abc123\n", "")) == ""
    assert handoff_from_commit("[Unfinished] Build OmniPulse\n\nSome text.\n\nTask-Id: abc123") == ""
    assert handoff_from_commit("[Unfinished] Build OmniPulse") == "" and handoff_from_commit("") == ""
    # A long message is cut, keeping its beginning and its end.
    long = message.replace("I scaffolded the app.", "word " * 2000)
    assert len(handoff_from_commit(long)) < 3700 and "error TS2580" in handoff_from_commit(long)


def test_the_note_about_the_command_that_still_fails():
    assert failure_note(None) == "" and failure_note({}) == "" and failure_note({"command": ""}) == ""
    note = failure_note({"command": "npm run\n  build", "exit_code": 2, "output": "> build\n\n\nsrc/a.ts(1,1): error TS1\n", "timed_out": False})
    assert note == "Still failing when the run stopped: `npm run build` (exit code 2)\n    > build\n    src/a.ts(1,1): error TS1"
    assert failure_note({"command": "npm test", "exit_code": None, "output": "", "timed_out": True}) == (
        "Still failing when the run stopped: `npm test` (timed out)"
    )
    # Long output is cut down to its beginning and, mostly, its end; no line of it starts a paragraph of its own.
    long = failure_note({"command": "make", "exit_code": 1, "output": "\n".join(f"line {n}" for n in range(500))})
    assert len(long) < 1600 and "    line 0\n" in long and long.endswith("    line 499") and "\n\n" not in long
    assert "line 250" not in long


def test_a_follow_up_that_finds_nothing_left_to_do_takes_the_mark_off(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    run_stream(client, TASK)
    assert github.pulls["acme/demo"][0]["title"].startswith("[Unfinished] ")

    install_deps(ScriptedLLM([[("read_file", {"path": "NOTES.md"})], [("finish", {"summary": "The notes are complete as they are."})]]))
    _events, task = run_stream(
        client, {"repository": "acme/demo", "prompt": TASK["prompt"], "pull_request": 1, "continue_work": True}
    )
    assert task["outcome"] == "no_changes" and task["pr_number"] == 1
    assert github.pulls["acme/demo"][0]["title"] == "add() returns the wrong result, fix it"
    assert "Pull request #1 is no longer marked as unfinished" in messages(task, "info")
    comment = [item for item in github.comments if item["issue"] == 1][-1]["body"]
    assert comment.startswith("✅ I went over this pull request to finish it and found nothing left to do")
    assert "The notes are complete as they are." in comment
    # Other work on the pull request that changes nothing says nothing about whether it is complete.
    install_deps(ScriptedLLM([ADD_NOTES[0], [("write_file", {"path": "MORE.md", "content": "more\n"})], RATE_LIMIT]))
    run_stream(client, {"repository": "acme/demo", "prompt": "More", "pull_request": 1})
    assert github.pulls["acme/demo"][0]["title"].startswith("[Unfinished] ")
    install_deps(ScriptedLLM([[("read_file", {"path": "NOTES.md"})], [("finish", {"summary": "Nothing to do."})]]))
    run_stream(client, {"repository": "acme/demo", "prompt": "Check the wording", "pull_request": 1})
    assert github.pulls["acme/demo"][0]["title"].startswith("[Unfinished] ")


@pytest.mark.parametrize(
    "steps, kept",
    [
        # Nothing was changed yet: there is nothing to keep.
        ([[("read_file", {"path": "calc.py"})], RATE_LIMIT], False),
        # The agent says itself that it cannot complete the task: what it changed is
        # kept all the same, for the owner of the repository to judge.
        ([ADD_NOTES[0], ADD_NOTES[1], [("finish", {"summary": "This cannot work.", "success": False})]], True),
        ([ADD_NOTES[0], ADD_NOTES[1], LLMError("groq returned HTTP 401: Invalid API Key", kind="auth", status_code=401)], True),
    ],
)
def test_what_is_kept_and_what_is_not(client: TestClient, app_env, install_deps, github: FakeGitHub, steps, kept):
    install_deps(ScriptedLLM(steps))
    _events, task = run_stream(client, TASK)
    assert (task["status"] == "completed") is kept and bool(github.pulls["acme/demo"]) is kept
    assert task["unfinished"] is kept


def test_keeping_unfinished_work_can_be_switched_off(client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("JULES_KEEP_UNFINISHED_WORK", "false")
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    _events, task = run_stream(client, TASK)
    assert task["status"] == "failed" and task["unfinished"] is False and github.pulls["acme/demo"] == []
    assert any("Unfinished work in the sandbox was discarded" in line for line in messages(task, "warn"))


def test_agent_that_runs_out_of_turns_delivers_what_it_has(client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("JULES_MAX_AGENT_TURNS", "2")
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], FINISH]))
    _events, task = run_stream(client, TASK)
    assert task["unfinished"] is True and task["stop_reason"] == "max_turns"
    assert task["unfinished_reason"] == "The agent used all 2 turns without finishing the task."
    assert "The agent used all 2 turns without finishing the task." in github.pulls["acme/demo"][0]["body"]


# --------------------------------------------------------------------------- #
# Keys and choices sent along with a task
# --------------------------------------------------------------------------- #
GEMINI_KEY = "AQ.test-browser-key-for-tests-only"


def test_keys_sent_with_a_task_are_used_for_it_and_shown_nowhere(client: TestClient, app_env, install_deps, github: FakeGitHub):
    def leaky(notice):
        # A model that prints what it should not: the key ends up in command output.
        return ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("run_command", {"command": f"echo key={GEMINI_KEY}"})],
                [("write_file", {"path": "NOTES.md", "content": f"key: {GEMINI_KEY}\n"})],
                [("finish", {"summary": f"Done with {GEMINI_KEY}.", "pr_title": "Notes"})],
            ]
        )

    deps = install_deps(leaky)
    events, task = run_stream(
        client,
        {**TASK, "llm": {"primary": "gemini", "models": {"gemini": "gemini-4-argon"}}},
        headers={"X-Gemini-Key": GEMINI_KEY, "X-OpenRouter-Key": "sk-or-browser", "X-Nonsense-Key": "x"},
    )
    assert task["status"] == "completed"
    (settings,) = deps.settings_seen
    assert settings.gemini_api_key == GEMINI_KEY and settings.openrouter_api_key == "sk-or-browser"
    assert settings.llm_chain == ["gemini", "groq", "openrouter"] and settings.gemini_model == "gemini-4-argon"
    # Nowhere: not in what is streamed, not in the record, not in the pull request.
    everything = json.dumps(events) + json.dumps(task_store.get(task["task_id"])) + json.dumps(github.pulls["acme/demo"])
    assert GEMINI_KEY not in everything and "sk-or-browser" not in everything
    assert "key=***" in everything  # what the command printed, with the key masked
    # The server's own configuration is what it was.
    assert client.get("/api/v1/health/env").json()["variables"]["GEMINI_API_KEY"] is False


def test_a_server_without_a_model_key_runs_tasks_with_the_callers(client: TestClient, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "e2b_" + "e" * 40)
    monkeypatch.setenv("GITHUB_TOKEN", FakeGitHub.PAT)
    assert client.get("/api/v1/health/env").json()["status"] == "INCOMPLETE"
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    assert task["status"] == "failed" and "an LLM API key" in task["error"]
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK, headers={"X-Gemini-Key": GEMINI_KEY})
    assert task["status"] == "completed" and task["pr_number"] == 1


@pytest.mark.parametrize("value", ["", "   ", "has spaces in it", "x" * 600, "bad\tkey"])
def test_values_that_cannot_be_keys_are_ignored(client: TestClient, app_env, install_deps, value):
    deps = install_deps(fix_add_script())
    run_stream(client, TASK, headers={"X-Gemini-Key": value})
    assert deps.settings_seen[0].gemini_api_key is None


def test_models_that_answered_are_recorded(client: TestClient, app_env, install_deps, github: FakeGitHub):
    from apps.api.services.llm import FailoverClient

    def pool(_notice):
        groq = ScriptedLLM([[("read_file", {"path": "calc.py"})], RATE_LIMIT], model="openai/gpt-oss-120b")
        groq.provider = "groq"
        gemini = ScriptedLLM(fix_add_script().steps[1:], model="gemini-flash-latest")
        gemini.provider = "gemini"
        return FailoverClient([groq, gemini])

    install_deps(pool)
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["unfinished"] is False
    assert task["llm_chain"] == ["groq / openai/gpt-oss-120b", "gemini / gemini-flash-latest"]
    assert [(item["provider"], item["requests"], item["failures"]) for item in task["llm_usage"]] == [("groq", 1, 1), ("gemini", 3, 0)]
    assert (task["llm_provider"], task["llm_model"]) == ("gemini", "gemini-flash-latest")
    log = messages(task)
    assert "If that provider reaches a limit or fails, the agent carries on with: gemini / gemini-flash-latest" in log
    assert "groq's rate limit is reached (8,000 tokens per minute): switching to gemini / gemini-flash-latest" in log
    assert "model `groq/openai/gpt-oss-120b, gemini/gemini-flash-latest`" in github.pulls["acme/demo"][0]["body"]


def provider_transport(handler):
    """Route the provider calls of /llm/test to a handler."""
    import apps.api.routers.llm as llm_router

    original = llm_router.probe_provider

    def patched(settings, provider, **kwargs):
        return original(settings, provider, transport=httpx.MockTransport(handler), **kwargs)

    return llm_router, original, patched


def test_trying_a_key_out(client: TestClient, app_env, monkeypatch):
    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "generativelanguage.googleapis.com":
            if request.headers["x-goog-api-key"] != GEMINI_KEY:
                return httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API key."}})
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"functionCall": {"name": "confirm", "args": {"ready": True}}}]}, "finishReason": "STOP"}]})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "confirm", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]})

    module, _original, patched = provider_transport(handler)
    monkeypatch.setattr(module, "probe_provider", patched)

    good = client.post("/api/v1/llm/test", json={"provider": "gemini"}, headers={"X-Gemini-Key": GEMINI_KEY}).json()
    assert good["ok"] is True and good["tool_call"] is True and good["key_source"] == "request"
    assert good["model"] == "gemini-flash-latest" and "called the test tool" in good["message"]
    # Both models a task would use with this provider were tried, in their order.
    assert [request.url.path for request in seen] == [
        "/v1beta/models/gemini-flash-latest:generateContent",
        "/v1beta/models/gemini-3.5-flash-lite:generateContent",
    ]
    assert [(item["model"], item["ok"], item["tool_call"]) for item in good["models"]] == [
        ("gemini-flash-latest", True, True),
        ("gemini-3.5-flash-lite", True, True),
    ]

    bad = client.post("/api/v1/llm/test", json={"provider": "gemini", "model": "gemini-4-argon"}, headers={"X-Gemini-Key": "AQ.wrong"}).json()
    assert bad["ok"] is False and bad["kind"] == "auth" and "API key not valid" in bad["message"] and "AQ.wrong" not in json.dumps(bad)
    assert "gemini-4-argon" in seen[-1].url.path and "models" not in bad  # one model was named: one is tried

    # A list of models from the settings page: each is tried.
    seen.clear()
    listed = client.post(
        "/api/v1/llm/test", json={"provider": "gemini", "model": "gemini-3.8-flash, gemini-3.6-flash"}, headers={"X-Gemini-Key": GEMINI_KEY}
    ).json()
    assert [item["model"] for item in listed["models"]] == ["gemini-3.8-flash", "gemini-3.6-flash"] and listed["model"] == "gemini-3.8-flash"
    assert len(seen) == 2
    # Four long names are a long setting; it is not refused for its length.
    long_names = ", ".join(f"vendor/{'m' * 150}-{index}" for index in range(4))
    wordy = client.post("/api/v1/llm/test", json={"provider": "gemini", "model": long_names}, headers={"X-Gemini-Key": GEMINI_KEY})
    assert wordy.status_code == 200 and len(wordy.json()["models"]) == 4

    # Without a key in the request the server's own is tried - here Groq has one.
    server = client.post("/api/v1/llm/test", json={"provider": "GROQ"}).json()
    assert server["ok"] is True and server["key_source"] == "server" and seen[-1].url.host == "api.groq.com"
    missing = client.post("/api/v1/llm/test", json={"provider": "openrouter"}).json()
    assert missing["ok"] is False and missing["kind"] == "no_key" and missing["key_source"] is None
    assert client.post("/api/v1/llm/test", json={"provider": "skynet"}).json()["kind"] == "unknown_provider"


def test_trying_a_key_out_needs_the_access_key(client: TestClient, app_env, monkeypatch):
    monkeypatch.setenv("JULES_ACCESS_KEY", "open-sesame-123")
    assert client.post("/api/v1/llm/test", json={"provider": "gemini"}, headers={"X-Gemini-Key": GEMINI_KEY}).status_code == 401


# --------------------------------------------------------------------------- #
# What the dashboard asks GitHub through the API
# --------------------------------------------------------------------------- #
def test_checks_of_a_commit_or_branch(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(ScriptedLLM([]))
    main = github.remote.rev("acme/demo", "refs/heads/main")
    url = "/api/v1/github/checks"
    assert client.get(url, params={"repository": "acme/demo", "ref": "main"}).json() == {
        "ok": True, "repository": "acme/demo", "sha": main, "state": "none", "error": None,
        "failing": [], "failed": 0, "passed": 0, "pending": 0, "notes": [],
    }
    github.add_status("acme/demo", main, "Vercel", "failure", "Deployment has failed", "https://vercel.com/x")
    github.add_check_run("acme/demo", main, "build", "success")
    failing = client.get(url, params={"repository": "https://github.com/acme/demo", "ref": main}).json()
    assert failing["state"] == "failure" and failing["passed"] == 1 and failing["failed"] == 1
    assert failing["failing"] == [
        {"kind": "status", "id": 7001, "name": "Vercel", "conclusion": "failure", "app": "", "url": "https://vercel.com/x", "summary": "Deployment has failed"}
    ]
    assert client.get(url, params={"repository": "acme/demo", "ref": "nope"}).json()["error"] == "acme/demo has no branch or commit 'nope'."
    assert client.get(url, params={"repository": "not a repo", "ref": "main"}).json()["ok"] is False


def test_pull_request_status_reports_the_checks(client: TestClient, app_env, install_deps, github: FakeGitHub):
    first = open_pull_request(client, install_deps, github)
    install_deps(ScriptedLLM(ADD_NOTES))
    _events, follow_up = run_stream(client, {"repository": "acme/demo", "prompt": "Notes", "pull_request": 1})
    head = head_of(github, first)
    github.add_check_run("acme/demo", head, "build", "failure", title="Build failed")

    # A follow-up task has no branch of its own: its pull request is found by number.
    url = f"/api/v1/tasks/{follow_up['task_id']}/pull-request"
    assert client.get(url, params={"repository": "acme/demo"}).status_code == 404
    status = client.get(url, params={"repository": "acme/demo", "number": 1}).json()
    assert status["number"] == 1 and status["head_sha"] == head and status["state"] == "open"
    assert status["checks"]["state"] == "failure" and [item["name"] for item in status["checks"]["failing"]] == ["build"]
    assert client.get(url, params={"repository": "acme/demo", "number": 42}).status_code == 404

    # A merged pull request is not asked about its checks any more.
    github.pulls["acme/demo"][0].update(state="closed", merged_at="2026-11-11T20:00:00Z")
    merged = client.get(url, params={"repository": "acme/demo", "number": 1}).json()
    assert merged["state"] == "merged" and merged["checks"] is None


def test_identity_of_a_github_account(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(ScriptedLLM([]))
    found = client.get("/api/v1/github/identity", params={"login": "@Ada-Example"}).json()
    assert found == {
        "ok": True,
        "login": "ada-example",
        "name": "Ada Example",
        "email": "1234567+ada-example@users.noreply.github.com",
        "avatar_url": "https://avatars.githubusercontent.com/u/1234567",
        "html_url": "https://github.com/ada-example",
        "error": None,
    }
    assert client.get("/api/v1/github/identity", params={"login": "octocat"}).json()["name"] == "octocat"  # no display name set
    assert client.get("/api/v1/github/identity", params={"login": "nobody-here"}).json()["error"] == "There is no GitHub user 'nobody-here'."
    assert "is not a GitHub user name" in client.get("/api/v1/github/identity", params={"login": "../app"}).json()["error"]


def test_automation_status_says_what_the_app_is_missing(client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch):
    install_deps(ScriptedLLM([]))
    body = client.get("/api/v1/github/automation").json()
    assert body["ci_autofix"] is True and body["attempts"] == 2 and body["webhook_secret"] is False and body["ready"] is False
    assert body["app"]["settings_url"] == "https://github.com/settings/apps/jules-test-app"
    assert {item["id"]: item["subscribed"] for item in body["events"]} == {
        "check_suite": False, "status": False, "workflow_run": False, "issue_comment": True
    }
    assert {item["id"]: item["granted"] for item in body["permissions"]} == {"checks": False, "statuses": False, "actions": False}

    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    github.app_events += ["check_suite", "status"]
    github.app_permissions.update(checks="read", statuses="read")
    body = client.get("/api/v1/github/automation").json()
    assert body["ready"] is True and body["webhook_secret"] is True
    assert {item["id"]: item["granted"] for item in body["permissions"]} == {"checks": True, "statuses": True, "actions": False}
    monkeypatch.setenv("JULES_CI_AUTOFIX", "off")
    assert client.get("/api/v1/github/automation").json()["ready"] is False


def test_an_app_owned_by_an_organization_is_configured_there(client: TestClient, app_env, install_deps, github: FakeGitHub):
    github.app_owner = {"login": "acme", "type": "Organization"}
    install_deps(ScriptedLLM([]))
    app = client.get("/api/v1/github/automation").json()["app"]
    assert app["settings_url"] == "https://github.com/organizations/acme/settings/apps/jules-test-app"
    assert app["permissions_url"] == "https://github.com/organizations/acme/settings/apps/jules-test-app/permissions"


# --------------------------------------------------------------------------- #
# Started from GitHub
# --------------------------------------------------------------------------- #
# Who writes the comments of these tests (the login in the shared webhook payload).
COMMENTER = comment_event("")["comment"]["user"]["login"]


def pr_comment(body: str, **overrides: Any) -> Dict[str, Any]:
    return comment_event(body, issue={"number": 1, "title": "Fix add()", "pull_request": {"url": "x"}}, **overrides)


def test_jules_comment_on_a_pull_request_adds_to_it(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    first = open_pull_request(client, install_deps, github)
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    response = deliver(client, "issue_comment", pr_comment("/jules add a NOTES.md that explains add()"), delivery="d-pr")
    assert response.status_code == 202 and response.json()["dispatched"] is True
    task = task_store.get(response.json()["task"]["task_id"])
    assert task["outcome"] == "pr_updated" and task["pull_request"] == 1 and task["source"] == "webhook"
    assert "Instructions from @kennedy on pull request #1: add a NOTES.md that explains add()" in prompt_of(llm)
    assert github.remote.log("acme/demo", first["head_branch"])[0] == "Add notes"
    # One comment on the pull request: the runner's own about what it pushed.
    (comment,) = [item for item in github.comments if item["issue"] == 1]
    assert comment["body"].startswith("✅ I pushed")


@pytest.mark.parametrize("text", ["/jules", "/jules fix the checks", "/jules Fix CI", "/jules please repair the failing build.", "/jules fix the tests"])
def test_jules_comment_asking_for_the_checks_to_be_fixed(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch, text):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    first = open_pull_request(client, install_deps, github)
    break_the_build(github, head_of(github, first))
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    task = task_store.get(deliver(client, "issue_comment", pr_comment(text), delivery="d-fix").json()["task"]["task_id"])
    assert task["fix_checks"] is True and task["outcome"] == "pr_updated"
    assert "The automated checks fail on pull request #1" in prompt_of(llm)


def test_jules_comment_with_nothing_to_fix_says_so(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    open_pull_request(client, install_deps, github)
    install_deps(ScriptedLLM([]))
    deliver(client, "issue_comment", pr_comment("/jules"), delivery="d-none")
    (comment,) = [item for item in github.comments if item["issue"] == 1]
    assert comment["body"].startswith("ℹ️ There was nothing for me to fix: no check has failed on the head commit of this pull request.")


@pytest.mark.parametrize("text", ["/jules continue", "/jules Continue.", "/jules finish it", "/jules please carry on", "/jules finish the work!"])
def test_jules_continue_on_a_pull_request_finishes_what_was_left(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch, text):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    run_stream(client, TASK)
    assert github.pulls["acme/demo"][0]["title"].startswith("[Unfinished] ")
    llm = fix_add_script()
    install_deps(llm)
    response = deliver(client, "issue_comment", pr_comment(text), delivery="d-continue")
    task = task_store.get(response.json()["task"]["task_id"])
    assert task["continue_work"] is True and task["fix_checks"] is False and task["outcome"] == "pr_updated"
    prompt = prompt_of(llm)
    assert "Finish the work on pull request #1 that was left unfinished." in prompt
    assert "A previous run started this work and stopped before it was finished." in prompt
    # Where the work stood is read from the commit that kept it.
    assert "What the previous run left behind:\nAdded" not in prompt  # that run wrote no account
    assert "What the previous run left behind:\nUnfinished: the agent was stopped before it was done. groq returned HTTP 429" in prompt
    assert github.pulls["acme/demo"][0]["title"] == "add() returns the wrong result, fix it"  # the mark is gone


def test_jules_continue_takes_further_instructions(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    open_pull_request(client, install_deps, github)
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    deliver(client, "issue_comment", pr_comment("/jules continue\nand keep the README short"), delivery="d-more")
    assert f"Instructions from @{COMMENTER}: and keep the README short" in prompt_of(llm)
    # "continue" inside other instructions is not the command.
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)
    response = deliver(client, "issue_comment", pr_comment("/jules continue the refactoring of calc.py"), delivery="d-other")
    assert task_store.get(response.json()["task"]["task_id"])["continue_work"] is False
    assert f"Instructions from @{COMMENTER} on pull request #1: continue the refactoring of calc.py" in prompt_of(llm)


def test_a_task_from_an_issue_that_is_cut_short_says_so_and_how_to_continue(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    response = deliver(client, "issue_comment", comment_event("/jules fix this"), delivery="d-cut")
    task = task_store.get(response.json()["task"]["task_id"])
    assert task["unfinished"] is True and task["outcome"] == "pull_request"
    (comment,) = [item for item in github.comments if item["issue"] == 7]
    assert comment["body"].startswith(f"⚠️ I opened #1 ({task['pr_url']}), but the work in it is unfinished: groq returned HTTP 429")
    assert "Comment `/jules continue` on #1 to have it finished; do not merge it as it is." in comment["body"]
    assert "✅" not in comment["body"]


def suite_failed(branch: str, sha: str) -> Dict[str, Any]:
    return {
        "action": "completed",
        "check_suite": {"conclusion": "failure", "head_branch": branch, "head_sha": sha, "app": {"name": "GitHub Actions"}},
        "repository": {"full_name": "acme/demo"},
    }


def test_failed_checks_on_the_agents_own_pull_request_are_fixed_unasked(
    client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch
):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    first = open_pull_request(client, install_deps, github)
    failed_sha = head_of(github, first)
    break_the_build(github, failed_sha)
    llm = ScriptedLLM(ADD_NOTES)
    install_deps(llm)

    response = deliver(client, "check_suite", suite_failed(first["head_branch"], failed_sha), delivery="d-ci-1")
    assert response.status_code == 202
    body = response.json()
    # No task is promised: whether one starts is decided after GitHub has its answer.
    assert body["status"] == "accepted" and body["dispatched"] is False and body["task"] is None
    assert f"failed on {first['head_branch']} @ {failed_sha[:10]}" in body["checking"]

    task = next(item for item in task_store.list() if item["fix_checks"])
    assert task["outcome"] == "pr_updated" and task["source"] == "webhook" and task["pull_request"] == 1
    assert "The automated checks fail on pull request #1" in prompt_of(llm)
    comments = [item["body"] for item in github.comments if item["issue"] == 1]
    assert comments[0] == (
        f"🔧 The checks failed on `{failed_sha[:10]}` (build, Vercel). I am working on a fix (attempt 1 of 2).\n\n"
        f"{fix_marker(failed_sha)}"
    )
    assert comments[1].startswith("🔧 I pushed") and len(comments) == 2
    assert attempted_fixes([{"body": text} for text in comments]) == [failed_sha]

    # The same failure arriving as other events (run, status) does not start a second fix.
    status = {"state": "failure", "sha": failed_sha, "context": "Vercel", "branches": [{"name": first["head_branch"]}], "repository": {"full_name": "acme/demo"}}
    again = deliver(client, "status", status, delivery="d-ci-2").json()
    assert again["dispatched"] is False and "already being looked at" in again["reason"]
    assert len([item for item in task_store.list() if item["fix_checks"]]) == 1


def test_automatic_fixes_stop_after_the_configured_number_of_attempts(
    client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch
):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    first = open_pull_request(client, install_deps, github)
    branch = first["head_branch"]

    def fail_and_report(delivery: str, steps: List[Any]) -> str:
        sha = head_of(github, first)
        github.add_check_run("acme/demo", sha, "build", "failure")
        install_deps(ScriptedLLM(steps))
        deliver(client, "check_suite", suite_failed(branch, sha), delivery=delivery)
        return sha

    fail_and_report("a1", ADD_NOTES)
    fail_and_report("a2", [[("read_file", {"path": "calc.py"})], [("write_file", {"path": "MORE.md", "content": "x\n"})], FINISH])
    assert len([item for item in task_store.list() if item["fix_checks"]]) == 2
    third = fail_and_report("a3", [])  # a third failure: no third fix
    assert len([item for item in task_store.list() if item["fix_checks"]]) == 2 and head_of(github, first) == third
    comments = [item["body"] for item in github.comments if item["issue"] == 1]
    assert comments[-1].startswith("⚠️ The checks still fail after 2 automatic fixes, so I am leaving it at that.")
    # Said once, not for every further failed check.
    github.add_status("acme/demo", third, "Vercel", "failure")
    from apps.api.routers import webhooks

    webhooks._claimed_commits.clear()
    deliver(client, "status", {"state": "failure", "sha": third, "context": "Vercel", "branches": [{"name": branch}], "repository": {"full_name": "acme/demo"}}, delivery="a4")
    assert len([item for item in github.comments if "leaving it at that" in item["body"]]) == 1
    # Asked for by a person, it is tried again.
    install_deps(ScriptedLLM([[("read_file", {"path": "calc.py"})], [("write_file", {"path": "THIRD.md", "content": "x\n"})], FINISH]))
    deliver(client, "issue_comment", pr_comment("/jules fix the checks"), delivery="a5")
    assert head_of(github, first) != third


@pytest.mark.parametrize("case", ["stale", "merged", "passed_meanwhile", "switched_off", "no_secret", "zero_attempts"])
def test_failed_checks_that_are_left_alone(client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch, case):
    if case != "no_secret":
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    first = open_pull_request(client, install_deps, github)
    sha = head_of(github, first)
    github.add_check_run("acme/demo", sha, "build", "failure")
    event_sha = sha
    if case == "stale":
        event_sha = github.remote.rev("acme/demo", "refs/heads/main")  # about a commit that is not the head any more
    elif case == "merged":
        github.pulls["acme/demo"][0].update(state="closed", merged_at="2026-11-11T20:00:00Z")
    elif case == "passed_meanwhile":
        github.check_runs.clear()
        github.add_check_run("acme/demo", sha, "build", "success")
    elif case == "switched_off":
        monkeypatch.setenv("JULES_CI_AUTOFIX", "false")
    elif case == "zero_attempts":
        monkeypatch.setenv("JULES_CI_AUTOFIX_ATTEMPTS", "0")
    install_deps(ScriptedLLM([]))
    response = deliver(client, "check_suite", suite_failed(first["head_branch"], event_sha), delivery=f"d-{case}", secret=None if case == "no_secret" else SECRET)
    assert [item for item in task_store.list() if item["fix_checks"]] == []
    assert [item for item in github.comments if item["issue"] == 1] == []
    assert head_of(github, first) == sha
    if case in ("switched_off", "zero_attempts"):
        assert "switched off" in response.json()["reason"]
    if case == "no_secret":
        assert "GITHUB_WEBHOOK_SECRET" in response.json()["reason"]


# --------------------------------------------------------------------------- #
# Standing instructions
# --------------------------------------------------------------------------- #
RULES = "Use import.meta.env instead of process.env in src/.\nAdd @types/node to devDependencies when vite.config.ts uses Node APIs."


def test_standing_instructions_follow_every_task(client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch):
    # The server's own (one line in a hosting dashboard, with \\n for the line breaks) ...
    monkeypatch.setenv("JULES_TASK_GUIDELINES", "Keep sample data short.\\nNo long comments.")
    assert client.get("/api/v1/health/env").json()["automation"]["task_guidelines"] is True
    llm = fix_add_script()
    install_deps(llm)
    # ... and the ones a caller sends along.
    _events, task = run_stream(client, {**TASK, "guidelines": f"  {RULES}  "})
    prompt = prompt_of(llm)
    block = (
        "Standing instructions (they apply to every task here; where they contradict the task above, the task wins):\n"
        f"Keep sample data short.\nNo long comments.\n\n{RULES}"
    )
    assert block in prompt
    # After the task, not instead of it.
    assert prompt.index(TASK["prompt"]) < prompt.index("Standing instructions")
    assert "Standing instructions are added to the task" in messages(task, "info")
    # They are the caller's rules, not the task: the result and the pull request show the task as written.
    assert task["prompt"] == TASK["prompt"] and "guidelines" not in task and "previous_handoff" not in task
    assert not any("guidelines" in (event.get("task") or {}) for event in _events)
    body = github.pulls["acme/demo"][0]["body"]
    assert "import.meta.env" not in body and "Keep sample data short" not in body


def test_tasks_without_standing_instructions_are_not_told_about_any(client: TestClient, app_env, install_deps):
    assert client.get("/api/v1/health/env").json()["automation"]["task_guidelines"] is False
    llm = fix_add_script()
    install_deps(llm)
    _events, task = run_stream(client, {**TASK, "guidelines": "   "})
    assert "Standing instructions" not in prompt_of(llm)
    assert not any("Standing instructions" in line for line in messages(task))


def test_standing_instructions_have_a_limit(client: TestClient, app_env, install_deps):
    install_deps(fix_add_script())
    response = client.post("/api/v1/tasks/stream", json={**TASK, "guidelines": "x" * 4001})
    assert response.status_code == 422
    assert standing_instructions("", "  ") == "" and standing_instructions() == ""
    assert standing_instructions("a", "a", "b").endswith(":\na\n\nb")  # the same text twice is said once


def test_standing_instructions_of_the_server_reach_tasks_started_on_github(
    client: TestClient, token_env, install_deps, github: FakeGitHub, monkeypatch
):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("JULES_TASK_GUIDELINES", RULES)
    llm = fix_add_script()
    install_deps(llm)
    response = deliver(client, "issue_comment", comment_event("/jules fix this"), delivery="d-rules")
    assert task_store.get(response.json()["task"]["task_id"])["status"] == "completed"
    assert f"task wins):\n{RULES}" in prompt_of(llm)


def test_standing_instructions_are_kept_when_work_is_continued(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    run_stream(client, TASK)
    llm = fix_add_script()
    install_deps(llm)
    run_stream(client, {"repository": "acme/demo", "prompt": TASK["prompt"], "pull_request": 1, "continue_work": True, "guidelines": RULES})
    prompt = prompt_of(llm)
    # The task, where the previous run stopped, the pull request - and the rules last.
    assert prompt.index("What the previous run left behind") < prompt.index("You are working on pull request #1") < prompt.index("Standing instructions")
    assert prompt.rstrip().endswith(RULES) or RULES in prompt


# --------------------------------------------------------------------------- #
# A model name that stands for a choice (openrouter/free, gemini-flash-latest)
# --------------------------------------------------------------------------- #
def test_the_models_a_router_answered_with_are_part_of_the_result(client: TestClient, app_env, install_deps):
    llm = ScriptedLLM(
        ADD_NOTES,
        model="openrouter/free",
        answers_as=["upstage/solar-pro-3:free", "upstage/solar-pro-3:free", "google/gemma-4-31b-it:free"],
    )
    llm.provider = "openrouter"
    install_deps(llm)
    _events, task = run_stream(client, TASK)
    assert task["outcome"] == "pull_request"
    # The task names what was asked for; which models answered is said next to it.
    assert (task["llm_provider"], task["llm_model"]) == ("openrouter", "openrouter/free")
    assert task["llm_usage"] == [
        {
            "provider": "openrouter",
            "model": "openrouter/free",
            "served": {"upstage/solar-pro-3:free": 2, "google/gemma-4-31b-it:free": 1},
        }
    ]
    # The log says it once per change of model, not with every reply.
    said = [line for line in messages(task, "info") if "answered with" in line]
    assert said == ["openrouter answered with upstage/solar-pro-3:free", "openrouter answered with google/gemma-4-31b-it:free"]


def test_a_model_that_answers_under_its_own_name_is_not_listed_twice(client: TestClient, app_env, install_deps):
    install_deps(fix_add_script())
    _events, task = run_stream(client, TASK)
    assert all("served" not in item for item in task["llm_usage"])
    assert not any("answered with" in line for line in messages(task))


def test_what_the_previous_run_left_behind_has_a_limit(client: TestClient, app_env, install_deps):
    install_deps(fix_add_script())
    too_long = {**TASK, "continue_work": True, "handoff": "x" * 8001}
    assert client.post("/api/v1/tasks/stream", json=too_long).status_code == 422
    assert client.post("/api/v1/tasks", json=too_long).status_code == 422


def test_what_a_caller_sent_along_is_not_handed_back(client: TestClient, app_env, install_deps):
    """The handoff and the standing instructions are the caller's own text: no
    route returns them with the task."""
    install_deps(fix_add_script())
    sent = {**TASK, "continue_work": True, "handoff": "The build failed on vite.config.ts.", "guidelines": RULES}
    created = client.post("/api/v1/tasks", json=sent)
    assert created.status_code in (200, 201, 202), created.text
    task_id = created.json()["task_id"]
    for body in (created.json(), client.get(f"/api/v1/tasks/{task_id}").json(), *client.get("/api/v1/tasks").json()):
        assert "guidelines" not in body and "previous_handoff" not in body
        assert "import.meta.env" not in json.dumps(body) and "The build failed on vite.config.ts." not in json.dumps(body)
    # `handoff` in a result is what *this* run left behind: nothing, when it finished.
    assert client.get(f"/api/v1/tasks/{task_id}").json()["handoff"] is None


# --------------------------------------------------------------------------- #
# Unfinished work says how it is continued
# --------------------------------------------------------------------------- #
def test_unfinished_work_names_the_comment_that_continues_it_where_github_reaches_the_server(
    client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch
):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", SECRET)
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    _events, task = run_stream(client, TASK)
    assert task["unfinished"] is True and task["outcome"] == "pull_request"
    body = github.pulls["acme/demo"][0]["body"]
    assert "Comment `/jules continue` on this pull request to have it finished; do not merge it as it is." in body
    assert "A follow-up task can finish it" not in body

    # More work on it that is cut short as well: the comment about the push says the same.
    install_deps(ScriptedLLM([[("read_file", {"path": "calc.py"})], [("write_file", {"path": "MORE.md", "content": "# More\n"})], RATE_LIMIT]))
    more = {"repository": "acme/demo", "prompt": TASK["prompt"], "pull_request": task["pr_number"], "continue_work": True}
    _events, again = run_stream(client, more)
    assert again["outcome"] == "pr_updated" and again["unfinished"] is True
    comment = github.comments[-1]["body"]
    assert comment.startswith("⚠️ I pushed ") and "\n\nComment `/jules continue` to have it finished.\n\n<sub>" in comment


def test_without_a_webhook_the_pull_request_does_not_promise_a_comment_command(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    install_deps(ScriptedLLM([ADD_NOTES[0], ADD_NOTES[1], RATE_LIMIT]))
    _events, task = run_stream(client, TASK)
    body = github.pulls["acme/demo"][0]["body"]
    assert "A follow-up task can finish it; do not merge it as it is." in body and "/jules continue" not in body
    install_deps(ScriptedLLM([[("read_file", {"path": "calc.py"})], [("write_file", {"path": "MORE.md", "content": "# More\n"})], RATE_LIMIT]))
    more = {"repository": "acme/demo", "prompt": TASK["prompt"], "pull_request": task["pr_number"], "continue_work": True}
    _events, again = run_stream(client, more)
    assert again["outcome"] == "pr_updated" and again["unfinished"] is True
    assert github.comments[-1]["body"].startswith("⚠️ I pushed ") and "/jules continue" not in github.comments[-1]["body"]


# --------------------------------------------------------------------------- #
# A file that cannot be committed does not cost the result
# --------------------------------------------------------------------------- #
def test_a_finished_task_is_delivered_without_the_file_that_cannot_be_committed(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    """The agent is done when its changes are collected: nobody can remove the
    bundle any more. The pull request is opened without it and says so."""
    bundle = "mkdir -p dist && head -c 1200000 /dev/zero | tr '\\0' 'x' > dist/bundle.js"
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})],
                [("run_command", {"command": bundle})],
                [("run_tests", {"command": CHECK_ADD})],
                [("finish", {"summary": "add() adds.", "pr_title": "Fix add()"})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["outcome"] == "pull_request" and not task.get("unfinished")
    assert [change["path"] for change in task["changed_files"]] == ["calc.py"]
    warning = next(line for line in messages(task, "warn") if "Left out so that the rest can be delivered" in line)
    assert warning.startswith("These changed files are larger than 1000 kB and cannot be committed by the agent: dist/bundle.js.")
    assert warning.endswith("Left out so that the rest can be delivered: dist/bundle.js (1,200 kB, too large to commit)")
    body = github.pulls["acme/demo"][0]["body"]
    assert (
        "Not included, because it is build output or too large to commit: "
        "`dist/bundle.js` (1,200 kB, too large to commit)"
    ) in body
    assert github.remote.files("acme/demo", task["head_branch"]) == sorted(DEMO_FILES)  # no dist/ among them


def test_build_output_that_can_be_committed_is_still_committed_by_a_finished_task(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    """Nothing changes for a task that finishes with a change that can be
    committed as it is: what the agent left in the work tree is what is delivered."""
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})],
                [("run_command", {"command": "mkdir -p dist && echo built > dist/out.txt"})],
                [("run_tests", {"command": CHECK_ADD})],
                [("finish", {"summary": "add() adds.", "pr_title": "Fix add()"})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert sorted(change["path"] for change in task["changed_files"]) == ["calc.py", "dist/out.txt"]
    assert "Not included" not in github.pulls["acme/demo"][0]["body"]
    assert not any("Left out" in line for line in messages(task, "warn"))


def test_a_task_whose_whole_result_cannot_be_committed_still_fails_for_that_reason(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    """Leaving the file out would leave nothing: an empty result would hide why."""
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("run_command", {"command": "head -c 1200000 /dev/zero | tr '\\0' 'x' > dump.sql"})],
                [("finish", {"summary": "Exported the data.", "pr_title": "Add the export"})],
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "failed" and not github.pulls.get("acme/demo")
    assert "larger than 1000 kB and cannot be committed by the agent: dump.sql" in task["error"]
    assert not any("Left out" in line for line in messages(task, "warn"))


def test_an_interrupted_run_that_only_left_build_output_has_nothing_to_keep(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    install_deps(
        ScriptedLLM(
            [
                [("read_file", {"path": "calc.py"})],
                [("run_command", {"command": "mkdir -p .cache/tool && echo built > .cache/tool/out.js"})],
                RATE_LIMIT,
            ]
        )
    )
    _events, task = run_stream(client, TASK)
    assert task["status"] == "failed" and not task.get("unfinished") and not github.pulls.get("acme/demo")
    assert (
        "Nothing of the unfinished work can be kept. All of it is build output or too large to commit: "
        ".cache/ (build output, 1 file)"
    ) in messages(task, "warn")
