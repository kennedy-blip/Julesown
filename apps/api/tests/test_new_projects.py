"""Repositories without code: empty ones, and ones that only have a README.

An empty repository has no branch a pull request could be opened against. A
task there builds the project from scratch and its result becomes the
repository's first commit. Everything below runs through the real pipeline
against the local stand-ins (``support.py``): the "GitHub" refuses its Git
database endpoints for a repository without commits exactly as github.com does.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from apps.api.routers.webhooks import build_result_comment
from apps.api.services.agent import (
    NEW_PROJECT_PROMPT,
    SYSTEM_PROMPT,
    TIME_UP,
    WRAP_UP_SECONDS,
    WRAP_UP_SECONDS_RATE_LIMITED,
    CodingAgent,
    system_prompt_for,
)
from apps.api.services.common import Deadline
from apps.api.services.llm import LLMError, LLMResponse
from apps.api.services.sandbox import EMPTY_TREE_SHA, RepoCheckout
from apps.api.services.store import task_store
from apps.api.services.verification import has_no_code

from .conftest import DEMO_FILES
from .support import FakeGitHub, GitRemote, LocalWorkspace, ScriptedLLM, git

COMMIT_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+/commit/[0-9a-f]{40}$")
INDEX_HTML = '<!doctype html>\n<html>\n<head><title>Tasks</title></head>\n<body><script src="app.js"></script></body>\n</html>\n'
README = "# Tasks\n\nOpen index.html in a browser. Deploy on any static host.\n"
PROJECT = [
    ("write_file", {"path": "index.html", "content": INDEX_HTML}),
    ("write_file", {"path": "app.js", "content": "console.log('tasks');\n"}),
    ("write_file", {"path": "README.md", "content": README}),
]
FINISH = [("finish", {"summary": "A static task list app.", "pr_title": "Create the task list app"})]


def run_stream(client: TestClient, payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    response = client.post("/api/v1/tasks/stream", json=payload)
    assert response.status_code == 200, response.text
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events[0]["type"] == "task" and events[-1]["type"] == "done"
    return events, events[-1]["task"]


def messages(task: Dict[str, Any], level: Optional[str] = None) -> List[str]:
    return [entry["message"] for entry in task["log_entries"] if level is None or entry["level"] == level]


@pytest.fixture
def blank(github: FakeGitHub) -> str:
    github.add_repo("acme/blank", None, empty=True)
    return "acme/blank"


# --------------------------------------------------------------------------- #
# Which repositories count as "no code yet"
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "files, expected",
    [
        ([], True),  # no commits at all
        (["README.md"], True),  # created on github.com with "Add a README"
        (["README.md", "LICENSE", ".gitignore"], True),
        (["docs/notes.md", ".gitattributes", ".editorconfig"], True),
        (["README.md", "index.html"], False),
        (["package.json"], False),
        (["requirements.txt"], False),  # .txt is only a document for well-known names
        ([".github/workflows/ci.yml"], False),
    ],
)
def test_has_no_code(files, expected):
    assert has_no_code(files, len(files)) is expected


def test_has_no_code_needs_the_whole_listing():
    # Seven documents are a documentation project; a cut-off listing proves nothing.
    assert has_no_code([f"doc{index}.md" for index in range(7)], 7) is False
    assert has_no_code(["README.md"], 250) is False


# --------------------------------------------------------------------------- #
# The work tree of an empty repository
# --------------------------------------------------------------------------- #
def test_empty_work_tree_reports_everything_as_added(tmp_path):
    workspace = LocalWorkspace(tmp_path / "sandbox")
    checkout = RepoCheckout.init_empty(workspace, branch="trunk", clone_url="https://github.com/acme/blank.git")
    assert checkout.empty and checkout.base_sha == "" and checkout.diff_base == EMPTY_TREE_SHA
    assert checkout.list_files() == ([], 0) and checkout.key_files() == []
    assert checkout.collect_changes().empty
    assert git(checkout.directory, "symbolic-ref", "HEAD") == "refs/heads/trunk"
    assert git(checkout.directory, "remote", "get-url", "origin") == "https://github.com/acme/blank.git"

    workspace.write_file(f"{checkout.directory}/README.md", "# Hi\n")
    workspace.write_file(f"{checkout.directory}/src/app.py", "print('hi')\nprint('there')\n")
    workspace.write_file(f"{checkout.directory}/node_modules/dep/index.js", "junk\n")
    workspace.run("printf '#!/bin/sh\\n' > run.sh && chmod +x run.sh", cwd=checkout.directory)
    changeset = checkout.collect_changes()
    summary = {change.path: (change.status, change.mode, change.additions) for change in changeset.changes}
    assert summary == {
        "README.md": ("A", "100644", 1),
        "run.sh": ("A", "100755", 1),
        "src/app.py": ("A", "100644", 2),
    }
    assert len(changeset.tree_sha) == 40 and all(change.content_b64 is not None for change in changeset.changes)
    assert "3 files changed" in checkout.diff_stat()


# --------------------------------------------------------------------------- #
# The agent in a repository without code
# --------------------------------------------------------------------------- #
def empty_checkout(tmp_path) -> RepoCheckout:
    return RepoCheckout.init_empty(LocalWorkspace(tmp_path / "sandbox"), branch="main")


def readme_only_checkout(remote: GitRemote, tmp_path, readme: str = "# Blank\n") -> RepoCheckout:
    remote.create("acme/readme-only", {"README.md": readme})
    workspace = LocalWorkspace(tmp_path / "sandbox")
    return RepoCheckout.clone(workspace, clone_url=remote.clone_url("acme/readme-only"), branch="main")


def make_agent(checkout: RepoCheckout, llm: ScriptedLLM, **options: Any) -> Tuple[CodingAgent, List[Tuple[str, str]]]:
    events: List[Tuple[str, str]] = []
    options.setdefault("deadline", Deadline(300))
    options.setdefault("sleep", lambda _seconds: None)
    agent = CodingAgent(llm=llm, checkout=checkout, emit=lambda level, message: events.append((level, message)), **options)
    return agent, events


def run(agent: CodingAgent, prompt: str = "Build a task list app", **options: Any):
    return agent.run(prompt=prompt, repository="acme/blank", branch="main", **options)


def test_the_two_sets_of_instructions():
    assert system_prompt_for(new_project=False, first_commit=False) == SYSTEM_PROMPT
    first = system_prompt_for(new_project=True, first_commit=True)
    by_pull_request = system_prompt_for(new_project=True, first_commit=False)
    assert "{delivery}" not in first and "{delivery}" not in by_pull_request
    assert "as the repository's first commit" in first and "pull request" not in first
    assert "opens the pull request for you" in by_pull_request
    for prompt in (first, by_pull_request):
        # What makes the result usable afterwards, on any host.
        for phrase in ("README.md", ".gitignore", ".env.example", "index.html at the repository root", "PORT", "Vercel"):
            assert phrase in prompt
        # Rules for changing existing code have no place in an empty repository.
        assert "Look before you write" not in prompt and "smallest change" not in prompt
        # How a run that has little time gets something that builds.
        for phrase in ("Keep files lean", "then install and run the build once", "fix what it reports before you write anything else"):
            assert phrase in prompt
    assert "fix what it reports before you do anything else" in SYSTEM_PROMPT
    # It is sent with every request of a run: it has to stay small.
    assert len(NEW_PROJECT_PROMPT) < 3000


def test_empty_repository_gets_the_new_project_instructions(tmp_path):
    llm = ScriptedLLM([PROJECT, FINISH])
    agent, events = make_agent(empty_checkout(tmp_path), llm)
    outcome = run(agent)
    assert outcome.finished and agent.new_project
    system, task = llm.requests[0][0]["content"], llm.requests[0][1]["content"]
    assert system == system_prompt_for(new_project=True, first_commit=True)
    assert task.startswith("Repository: acme/blank - empty, it has no commits yet.")
    assert "becomes its first commit on branch `main`" in task and "Build a task list app" in task
    # How much time there is - and that running out of it does not lose the work.
    assert "Time budget: about 5 minutes for this run, including installs and builds." in task
    assert "Get a small version working first" in task
    assert "If the time runs out, what you have is kept and a follow-up run continues from it." in task
    assert "commit " not in task.split("Task:")[0].replace("first commit", "")  # there is no commit to name


def test_writing_into_an_empty_repository_is_not_held_back(tmp_path):
    """"Look before you write" protects existing code. Here there is none."""
    checkout = empty_checkout(tmp_path)
    llm = ScriptedLLM([PROJECT, FINISH])
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 2
    assert not [message for _level, message in events if "held back" in message]
    assert [message for level, message in events if level == "edit"] == [
        "write_file index.html (5 lines)",
        "write_file app.js (1 lines)",
        "write_file README.md (3 lines)",
    ]
    assert sorted(change.path for change in checkout.collect_changes().changes) == ["README.md", "app.js", "index.html"]
    assert outcome.files_read == []


def test_unfinished_new_project_is_sent_back_once(tmp_path):
    checkout = empty_checkout(tmp_path)
    llm = ScriptedLLM(
        [
            [
                ("write_file", {"path": "package.json", "content": '{"name": "tasks", }\n'}),
                ("write_file", {"path": "server.js", "content": "require('http');\n"}),
                ("write_file", {"path": ".env", "content": "API_KEY=secret\n"}),
            ],
            FINISH,
            FINISH,
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 3
    held = [message for level, message in events if level == "info" and "finish held back" in message]
    assert held == [
        "finish held back: the new project is not ready to be committed (README.md is missing; "
        ".gitignore is missing although the project installs dependencies; package.json is not valid JSON; "
        ".env would be committed) - the agent was asked to complete it"
    ]
    told = llm.tool_results()[-1]
    assert told.startswith("Not finished yet. Before the project is committed:")
    for expected in (
        "README.md is missing - add one that says what the project is, how to run it locally and how to deploy it",
        ".gitignore is missing",
        "package.json is not valid JSON",
        ".env would be committed - list the variable names without values in .env.example and add .env to .gitignore",
    ):
        assert expected in told
    # Once: the model may know better, and a second objection would only cost time.
    assert "call finish again and say so in the summary" in told


def test_a_file_too_large_to_commit_is_named_while_the_agent_can_still_act(tmp_path):
    """A lock file of a large dependency tree can exceed what a task may commit.
    Found only when collecting the changes, that would lose the whole project."""
    checkout = empty_checkout(tmp_path)
    llm = ScriptedLLM(
        [
            PROJECT,
            [("run_command", {"command": "head -c 1200000 /dev/zero | tr '\\0' 'x' > package-lock.json"})],
            FINISH,
            [("run_command", {"command": "echo package-lock.json >> .gitignore"})],
            FINISH,
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 5
    held = [message for level, message in events if level == "info" and "finish held back" in message]
    assert len(held) == 1 and "package-lock.json is larger than 1000 kB, more than a task can commit" in held[0]
    told = [result for result in llm.tool_results() if result.startswith("Not finished yet")]
    assert len(told) == 1 and "delete it or add it to .gitignore" in told[0]
    # What is committed now fits.
    paths = [change.path for change in checkout.collect_changes().changes]
    assert "package-lock.json" not in paths and "index.html" in paths


def test_too_many_files_are_counted_before_the_commit_fails(tmp_path):
    checkout = empty_checkout(tmp_path)
    llm = ScriptedLLM(
        [
            PROJECT,
            [("run_command", {"command": "mkdir -p out && for i in $(seq 1 320); do echo $i > out/page$i.html; done"})],
            FINISH,
            [("run_command", {"command": "echo out/ > .gitignore"})],
            FINISH,
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 5
    told = [result for result in llm.tool_results() if result.startswith("Not finished yet")]
    assert len(told) == 1 and "files would be committed, more than the limit of 300" in told[0]
    assert len(checkout.collect_changes().changes) < 20


def test_complete_new_project_finishes_at_once(tmp_path):
    llm = ScriptedLLM(
        [
            [
                ("write_file", {"path": "package.json", "content": '{"name": "tasks", "version": "1.0.0"}\n'}),
                ("write_file", {"path": "server.js", "content": "require('http');\n"}),
                ("write_file", {"path": ".gitignore", "content": "node_modules/\n.env\n"}),
                ("write_file", {"path": ".env", "content": "PORT=3000\n"}),  # ignored, so never committed
                ("write_file", {"path": ".env.example", "content": "PORT=\n"}),
                ("write_file", {"path": "README.md", "content": README}),
            ],
            FINISH,
        ]
    )
    checkout = empty_checkout(tmp_path)
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 2
    assert not [message for _level, message in events if "held back" in message]
    assert ".env" not in [change.path for change in checkout.collect_changes().changes]


def test_a_project_without_dependencies_needs_no_gitignore(tmp_path):
    llm = ScriptedLLM([PROJECT, FINISH])
    agent, events = make_agent(empty_checkout(tmp_path), llm)
    assert run(agent).turns == 2


def test_finishing_without_creating_anything_is_not_questioned(tmp_path):
    llm = ScriptedLLM([FINISH])
    agent, events = make_agent(empty_checkout(tmp_path), llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 1  # reported as "no changes" by the runner


def test_repository_with_only_a_readme_is_a_new_project_delivered_by_pull_request(remote: GitRemote, tmp_path):
    checkout = readme_only_checkout(remote, tmp_path)
    llm = ScriptedLLM([PROJECT, FINISH])
    agent, events = make_agent(checkout, llm, new_project=True)
    outcome = run(agent, file_listing="Repository files (1 tracked):\nREADME.md", key_files=checkout.key_files())
    assert outcome.finished and outcome.turns == 2
    system, task = llm.requests[0][0]["content"], llm.requests[0][1]["content"]
    assert system == system_prompt_for(new_project=True, first_commit=False)
    assert f"commit {checkout.base_sha[:10]}) - it has no code yet." in task
    assert "--- README.md (8 bytes) ---" in task and "Time budget" in task
    # The short README was shown in full, so replacing it is not "writing blind".
    assert not [message for _level, message in events if "held back" in message]


def test_files_generated_during_the_run_can_be_replaced_but_the_users_cannot_unread(remote: GitRemote, tmp_path):
    checkout = readme_only_checkout(remote, tmp_path, readme="# Notes\n\n" + "A line of the owner's notes.\n" * 40)
    llm = ScriptedLLM(
        [
            [("run_command", {"command": "printf '{\"name\": \"generated\"}\\n' > package.json"})],
            # Generated a moment ago by a command: nothing of the user's is in it.
            [("write_file", {"path": "package.json", "content": '{"name": "tasks", "version": "1.0.0"}\n'})],
            # The user's own README, of which only the beginning was shown.
            [("write_file", {"path": "README.md", "content": README})],
            [("read_file", {"path": "README.md"})],
            [("write_file", {"path": "README.md", "content": README})],
            [("write_file", {"path": ".gitignore", "content": "node_modules/\n"})],
            FINISH,
        ]
    )
    agent, events = make_agent(checkout, llm, new_project=True)
    outcome = run(agent, key_files=checkout.key_files())
    assert outcome.finished
    held = [message for _level, message in events if "held back" in message]
    assert held == ["write_file README.md held back: the existing file has not been read - the agent was asked to read it first"]
    assert ("edit", "write_file package.json (1 lines)") in events


def test_existing_code_keeps_the_usual_rules(remote: GitRemote, tmp_path):
    remote.create("acme/demo", DEMO_FILES)
    checkout = RepoCheckout.clone(LocalWorkspace(tmp_path / "sandbox"), clone_url=remote.clone_url("acme/demo"), branch="main")
    llm = ScriptedLLM([[("write_file", {"path": "notes.md", "content": "x\n"})], [("write_file", {"path": "notes.md", "content": "x\n"})], FINISH])
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and not agent.new_project
    assert llm.requests[0][0]["content"] == SYSTEM_PROMPT
    assert "Time budget" not in llm.requests[0][1]["content"]
    assert [message for _level, message in events if "held back" in message] == [
        "write_file notes.md held back: nothing has been read yet - the agent was asked to look at the repository first"
    ]


def test_cut_off_reply_in_a_new_project_asks_for_smaller_files(tmp_path):
    class CutOff(ScriptedLLM):
        def complete(self, messages, tools, *, max_output_tokens, timeout):
            if not self.requests:
                self.requests.append(json.loads(json.dumps(messages)))
                return LLMResponse(
                    content="", tool_calls=[], finish_reason="length", model=self.model,
                    assistant_message={"role": "assistant", "content": ""},
                )
            return super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)

    llm = CutOff([PROJECT, FINISH])
    agent, _events = make_agent(empty_checkout(tmp_path), llm)
    assert run(agent).finished
    assert llm.requests[1][-1] == {
        "role": "user",
        "content": (
            "Your reply was cut off because it was too long. Work in smaller steps: write fewer files per "
            "reply and keep each file small (split a large one into modules)."
        ),
    }


# --------------------------------------------------------------------------- #
# Running out of time
# --------------------------------------------------------------------------- #
class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class SlowLLM(ScriptedLLM):
    """Every reply takes ``seconds_per_reply`` of the (fake) clock."""

    def __init__(self, steps, clock: Clock, seconds_per_reply: float) -> None:
        super().__init__(steps)
        self.clock, self.seconds_per_reply = clock, seconds_per_reply

    def complete(self, messages, tools, *, max_output_tokens, timeout):
        self.clock.now += self.seconds_per_reply
        return super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)


def test_agent_is_asked_to_wrap_up_before_the_time_is_gone(tmp_path):
    """With room for one more reply left, the model is told so - once - and
    asked for an account of where the work stands."""
    clock = Clock()
    # 200 s for the agent (225 minus the 25 s kept for the commit); each reply takes 60 s.
    deadline = Deadline(225, clock=clock)
    llm = SlowLLM(
        [
            [("write_file", {"path": "index.html", "content": INDEX_HTML})],
            [("write_file", {"path": "app.js", "content": "console.log(1);\n"})],
            [("write_file", {"path": "style.css", "content": "body {}\n"})],
            [("finish", {"summary": "The page and its script work; there is no README yet.", "pr_title": "Create the app"})],
        ],
        clock,
        seconds_per_reply=60,
    )
    agent, events = make_agent(empty_checkout(tmp_path), llm, deadline=deadline)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 4
    # Asked before the fourth reply (20 s left), not earlier (80 s left after the second).
    wrap_up = [request[-1] for request in llm.requests if request[-1]["role"] == "user" and "Time is almost up" in request[-1]["content"]]
    assert len(wrap_up) == 1 and llm.requests[3][-1] == wrap_up[0] and len(llm.requests) == 4
    assert "say what is done, what is still missing or unchecked" in wrap_up[0]["content"]
    # Nothing is lost by stopping here, and the model is told so: it has no reason to rush something in.
    assert "Your work is kept as it is and a follow-up run continues from your summary." in wrap_up[0]["content"]
    assert "Pass success=false if the work is not complete or the last build or test run failed." in wrap_up[0]["content"]
    assert ("info", "The time budget is nearly used up: the agent was asked to wrap up") in events
    # The missing README would normally send the model back. Not now: there is no time for it.
    assert not [message for _level, message in events if "held back" in message]
    assert outcome.summary == "The page and its script work; there is no README yet."
    # Earlier - with half of its time gone - it was told how much is left.
    checks = [request[-1] for request in llm.requests if request[-1]["role"] == "user" and request[-1]["content"].startswith("Time check:")]
    assert len(checks) == 1 and llm.requests[2][-1] == checks[0]
    assert "about 40 seconds of working time are left" in checks[0]["content"]
    assert "check it now with the project's build or test command" in checks[0]["content"]
    assert ("info", "About 40s of working time are left: the agent was told to check its work") in events


def wrap_up_run(tmp_path, last_steps, *, build: str = ""):
    """A run of 200 s in which every reply takes 60 s: the wrap-up is asked for
    before the fourth reply. ``build`` is a command the agent runs third."""
    clock = Clock()
    third = [("run_command", {"command": build})] if build else [("write_file", {"path": "style.css", "content": "body {}\n"})]
    llm = SlowLLM(
        [
            [("write_file", {"path": "index.html", "content": INDEX_HTML})],
            [("write_file", {"path": "app.js", "content": "console.log(1);\n"})],
            third,
            last_steps,
        ],
        clock,
        seconds_per_reply=60,
    )
    agent, events = make_agent(empty_checkout(tmp_path), llm, deadline=Deadline(225, clock=clock))
    return run(agent), llm, events


def test_work_that_is_not_done_when_time_is_up_is_reported_as_cut_short(tmp_path):
    """Asked to wrap up, the model says the work is not complete. That is not
    the agent giving up on the task: it ran out of time, and what it wrote is
    kept (the runner delivers it as unfinished)."""
    outcome, llm, _events = wrap_up_run(
        tmp_path,
        [("finish", {"summary": "The page is there. Missing: the README and the styles for the list.", "success": False})],
    )
    assert "Time is almost up" in llm.requests[3][-1]["content"]
    assert outcome.stop_reason == "timeout" and not outcome.finished
    assert outcome.error == TIME_UP == "The time budget ran out before the agent finished."
    # Its account of where the work stands is what a follow-up run starts from.
    assert outcome.summary == "The page is there. Missing: the README and the styles for the list."


def test_a_failed_last_command_means_unfinished_whatever_the_model_says(tmp_path):
    outcome, _llm, _events = wrap_up_run(
        tmp_path,
        [("finish", {"summary": "Built the app.", "pr_title": "Create the app"})],
        build="echo 'vite.config.ts(8,18): error TS2580: Cannot find name process.' >&2; exit 2",
    )
    assert outcome.stop_reason == "timeout" and outcome.error == TIME_UP
    assert outcome.summary == "Built the app." and outcome.pr_title == "Create the app"
    assert outcome.last_failure == {
        "command": "echo 'vite.config.ts(8,18): error TS2580: Cannot find name process.' >&2; exit 2",
        "exit_code": 2,
        "timed_out": False,
        "output": "vite.config.ts(8,18): error TS2580: Cannot find name process.",
    }


def test_work_that_is_done_when_time_is_up_is_finished(tmp_path):
    outcome, _llm, _events = wrap_up_run(
        tmp_path,
        [("finish", {"summary": "Built the app; the build passes.", "pr_title": "Create the app"})],
        build="echo built",
    )
    assert outcome.finished and outcome.last_failure is None


def test_giving_up_with_time_left_is_still_giving_up(tmp_path):
    llm = ScriptedLLM([PROJECT, [("finish", {"summary": "This cannot be built without a paid API.", "success": False})]])
    agent, _events = make_agent(empty_checkout(tmp_path), llm)
    outcome = run(agent)
    assert outcome.stop_reason == "gave_up" and outcome.error == "This cannot be built without a paid API."


def test_the_command_that_still_fails_is_remembered(tmp_path):
    """Where a follow-up run has to start: the last command that failed and has
    not passed since."""
    broken = [("write_file", {"path": "build.sh", "content": "exit 3\n"}), ("write_file", {"path": "README.md", "content": README})]
    steps = [
        broken,
        [("run_command", {"command": "sh build.sh"})],  # fails
        [("run_command", {"command": "cat build.sh"})],  # looking at it is not fixing it
        FINISH,
    ]
    agent, _events = make_agent(empty_checkout(tmp_path), ScriptedLLM(steps), deadline=Deadline(300))
    outcome = run(agent)
    assert outcome.last_failure is not None
    assert outcome.last_failure["command"] == "sh build.sh" and outcome.last_failure["exit_code"] == 3

    # Run again with success, it is settled.
    steps = [
        broken,
        [("run_command", {"command": "sh build.sh"})],
        [("write_file", {"path": "build.sh", "content": "exit 0\n"})],
        [("run_command", {"command": "sh build.sh"})],
        FINISH,
    ]
    agent, _events = make_agent(empty_checkout(tmp_path / "second"), ScriptedLLM(steps), deadline=Deadline(300))
    assert run(agent).last_failure is None

    # A failure the agent has long moved on from is not where the work stands.
    steps = [
        broken,
        [("run_command", {"command": "sh build.sh"})],
        [("run_command", {"command": "true"})],
        [("run_command", {"command": "echo 1"})],
        [("run_command", {"command": "echo 2"})],
        [("run_command", {"command": "echo 3"})],
        FINISH,
    ]
    agent, _events = make_agent(empty_checkout(tmp_path / "third"), ScriptedLLM(steps), deadline=Deadline(300))
    assert run(agent).last_failure is None


def test_no_time_check_in_a_short_run(tmp_path):
    """Half of a minute is no news to anybody."""
    clock = Clock()
    llm = SlowLLM([PROJECT, [("write_file", {"path": "style.css", "content": "body {}\n"})], FINISH], clock, seconds_per_reply=30)
    agent, _events = make_agent(empty_checkout(tmp_path), llm, deadline=Deadline(135, clock=clock))  # 110 s for the agent
    assert run(agent).finished
    assert not any(str(message.get("content")).startswith("Time check:") for request in llm.requests for message in request)


def test_wrap_up_comes_earlier_when_the_rate_limit_slows_the_run(tmp_path):
    assert WRAP_UP_SECONDS_RATE_LIMITED > WRAP_UP_SECONDS
    clock = Clock()
    deadline = Deadline(225, clock=clock)
    llm = SlowLLM(
        [
            LLMError("groq returned HTTP 429: slow down", kind="rate_limit", retry_after=1.0),
            [("write_file", {"path": "index.html", "content": INDEX_HTML})],
            [("write_file", {"path": "README.md", "content": README})],
            FINISH,
        ],
        clock,
        seconds_per_reply=45,
    )
    agent, events = make_agent(empty_checkout(tmp_path), llm, deadline=deadline)
    assert run(agent).finished
    # 65 s were left before the third reply: more than the usual 40 s, but the next
    # reply may be a rate-limit pause away.
    assert "Time is almost up" in llm.requests[3][-1]["content"]
    assert not any("Time is almost up" in str(request[-1].get("content")) for request in llm.requests[:3])


def test_no_wrap_up_when_there_is_time(tmp_path):
    llm = ScriptedLLM([PROJECT, FINISH])
    agent, events = make_agent(empty_checkout(tmp_path), llm)
    assert run(agent).finished
    assert not any("Time is almost up" in str(message.get("content")) for request in llm.requests for message in request)


def test_wrap_up_also_applies_to_existing_repositories(remote: GitRemote, tmp_path):
    remote.create("acme/demo", DEMO_FILES)
    checkout = RepoCheckout.clone(LocalWorkspace(tmp_path / "sandbox"), clone_url=remote.clone_url("acme/demo"), branch="main")
    clock = Clock()
    llm = SlowLLM(
        [
            [("read_file", {"path": "calc.py"})],
            [("replace_in_file", {"path": "calc.py", "old_text": "a - b", "new_text": "a + b"})],
            [("finish", {"summary": "Fixed add(); the tests were not run for lack of time.", "pr_title": "Fix add()"})],
        ],
        clock,
        seconds_per_reply=90,
    )
    agent, events = make_agent(checkout, llm, deadline=Deadline(225, clock=clock))
    outcome = agent.run(prompt="Fix add()", repository="acme/demo", branch="main")
    assert outcome.finished and "Time is almost up" in llm.requests[2][-1]["content"]


# --------------------------------------------------------------------------- #
# End to end: a task on an empty repository
# --------------------------------------------------------------------------- #
def test_task_on_an_empty_repository_pushes_the_first_commit(
    client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str
):
    llm = ScriptedLLM([PROJECT, [("run_command", {"command": "test -s index.html"})], FINISH])
    install_deps(llm)
    events, task = run_stream(client, {"repository": blank, "prompt": "Build a small task list web app"})

    assert task["status"] == "completed" and task["outcome"] == "initial_commit" and task["error"] is None
    assert task["empty_repository"] is True and task["new_project"] is True
    # No pull request - and none is claimed.
    assert task["pr_url"] is None and task["pr_number"] is None and task["head_branch"] is None
    assert github.pulls["acme/blank"] == [] and github.calls("POST", r"/pulls$") == []
    # The commit is the one GitHub has on the default branch.
    head = github.remote.rev(blank, "refs/heads/main")
    assert task["commit_sha"] == head and COMMIT_URL.match(task["commit_url"])
    assert task["commit_url"].endswith(head) and task["repository_url"] == "https://github.com/acme/blank"
    assert task["branch"] == "main" and task["base_branch"] == "main"
    assert sorted(github.remote.files(blank, "main")) == ["README.md", "app.js", "index.html"]
    assert github.remote.show(blank, "main", "index.html") == INDEX_HTML.strip()
    assert github.remote.log(blank, "main") == ["Create the task list app", "Initial commit"]
    assert git(github.remote.path_for(blank), "for-each-ref", "--format=%(refname)") == "refs/heads/main"
    message = git(github.remote.path_for(blank), "log", "-1", "--format=%B", "main")
    assert "A static task list app." in message and f"Task-Id: {task['task_id']}" in message.splitlines()

    assert [f["status"] for f in task["changed_files"]] == ["A", "A", "A"]
    assert task["tests_status"] == "not_run" and task["docs_only"] is False
    stages = [event["fields"]["stage"] for event in events if event["type"] == "update" and "stage" in event["fields"]]
    assert stages == ["configuration", "github", "sandbox", "clone", "agent", "commit", "push", "done"]
    log = messages(task)
    assert "acme/blank is empty: the agent builds the project from scratch and the result becomes the first commit on main" in log
    assert "acme/blank has no commits to clone: starting from an empty work tree on main" in log
    assert f"First commit {head[:10]} pushed to main (3 files): {task['commit_url']}" in messages(task, "success")
    assert "The new project has no automated tests. Try it before you deploy it." in messages(task, "info")
    assert messages(task, "warn") == [] and messages(task, "error") == []


def test_empty_repository_with_github_app_credentials(
    client: TestClient, app_env, install_deps, github: FakeGitHub, blank: str
):
    install_deps(ScriptedLLM([PROJECT, FINISH]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build a small task list web app"})
    assert task["status"] == "completed" and task["outcome"] == "initial_commit"
    assert sorted(github.remote.files(blank, "main")) == ["README.md", "app.js", "index.html"]
    assert "GitHub access confirmed for acme/blank via GitHub App installation" in messages(task)
    # A public repository: nothing was cloned, so no clone token was requested either.
    assert all(request.get("permissions") is None for request in github.token_requests)


def test_first_commit_is_exactly_the_work_tree(client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str):
    install_deps(
        ScriptedLLM(
            [
                PROJECT
                + [
                    ("write_file", {"path": "bin/start.sh", "content": "#!/bin/sh\nexec python3 -m http.server\n"}),
                    ("write_file", {"path": "package.json", "content": '{"name": "tasks", "version": "1.0.0"}\n'}),
                    ("write_file", {"path": ".gitignore", "content": "node_modules/\n"}),
                ],
                [("run_command", {"command": "chmod +x bin/start.sh && mkdir -p node_modules/dep && echo junk > node_modules/dep/index.js"})],
                FINISH,
            ]
        )
    )
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    assert task["outcome"] == "initial_commit"
    files = github.remote.files(blank, "main")
    assert sorted(files) == [".gitignore", "README.md", "app.js", "bin/start.sh", "index.html", "package.json"]
    assert github.remote.mode(blank, "main", "bin/start.sh") == "100755"
    assert not any(path.startswith("node_modules") for path in files)


def test_nothing_created_means_nothing_pushed(client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str):
    install_deps(ScriptedLLM([[("finish", {"summary": "The task was unclear, so nothing was created."})]]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "hm"})
    assert task["status"] == "completed" and task["outcome"] == "no_changes"
    assert task["commit_sha"] is None and task["commit_url"] is None and task["pr_url"] is None
    assert github.remote.is_empty(blank)
    assert messages(task, "success") == ["Finished without creating any files, so nothing was pushed."]
    assert github.calls("PUT", r"/contents/") == []


def test_named_branch_on_an_empty_repository(client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str):
    install_deps(ScriptedLLM([PROJECT, FINISH]))
    _events, task = run_stream(client, {"repository": blank, "branch": "develop", "prompt": "Build it"})
    assert task["status"] == "failed" and task["stage"] == "github"
    assert task["error"] == (
        "acme/blank is empty, so it has no branch 'develop' to start from. Leave the base branch empty: "
        "the first commit is made on the default branch 'main'."
    )
    assert github.remote.is_empty(blank)

    # Naming the default branch is the same as naming none.
    _events, task = run_stream(client, {"repository": blank, "branch": "main", "prompt": "Build it"})
    assert task["outcome"] == "initial_commit" and task["branch"] == "main"


def test_repository_that_got_content_during_the_run_is_not_overwritten(
    client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str, tmp_path
):
    def someone_pushes(_messages):
        scratch = tmp_path / "theirs"
        scratch.mkdir()
        git(scratch, "init", "--quiet", "--initial-branch=main")
        (scratch / "README.md").write_text("# Theirs\n")
        git(scratch, "add", "-A")
        git(scratch, "commit", "--quiet", "-m", "Their first commit")
        git(scratch, "push", "--quiet", str(github.remote.path_for(blank)), "HEAD:refs/heads/main")
        return FINISH

    install_deps(ScriptedLLM([PROJECT, someone_pushes]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    assert task["status"] == "failed" and task["stage"] == "push"
    assert task["error"] == (
        "acme/blank received its first commit while the task was running, so the result was not pushed "
        "over it. Run the task again: it will now arrive as a pull request."
    )
    assert github.remote.log(blank, "main") == ["Their first commit"]
    assert github.remote.show(blank, "main", "README.md") == "# Theirs"
    assert task["commit_url"] is None and task["outcome"] is None


def test_half_made_first_commit_is_reported_as_it_is(
    client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str
):
    install_deps(ScriptedLLM([PROJECT, FINISH]))
    # A single hiccup would be retried: GitHub stays down here.
    github.fail_next("POST", r"/git/trees$", 502, {"message": "Server Error"}, times=99)
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    assert task["status"] == "failed" and task["stage"] == "push" and task["outcome"] is None
    assert task["error"].startswith(
        "The first commit was created in acme/blank with README.md, but the other 2 files could not be pushed: "
    )
    assert "HTTP 502" in task["error"]
    assert task["error"].endswith(
        "Run the task again - the repository is no longer empty, so the result will arrive as a pull request."
    )
    # Exactly what the message says is there.
    assert github.remote.files(blank, "main") == ["README.md"]
    assert task["commit_url"] is None and task["pr_url"] is None


def test_failing_tests_are_reported_and_the_project_is_still_pushed(
    client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str
):
    failing = f"{sys.executable} -c 'assert 1 + 1 == 3, \"sum is wrong\"'"
    install_deps(ScriptedLLM([PROJECT, [("run_tests", {"command": failing})], FINISH]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    assert task["outcome"] == "initial_commit" and task["tests_status"] == "failed" and task["tests_passed"] is False
    assert sorted(github.remote.files(blank, "main")) == ["README.md", "app.js", "index.html"]
    assert messages(task, "warn") == [
        "The last test run failed, and the files were pushed anyway: an empty repository has no pull "
        "request that could hold them back. Check the project before you rely on it."
    ]


def test_passing_tests_of_a_new_project(client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str):
    passing = f"{sys.executable} -c 'assert 1 + 1 == 2'"
    install_deps(ScriptedLLM([PROJECT, [("run_tests", {"command": passing})], FINISH]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    assert task["outcome"] == "initial_commit" and task["tests_status"] == "passed"
    assert messages(task, "warn") == [] and not any("no automated tests" in line for line in messages(task))


def test_agent_failure_leaves_the_repository_empty(client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str):
    """An agent that gives up before it has created anything leaves nothing behind."""
    install_deps(ScriptedLLM([[("finish", {"summary": "This cannot be built here.", "success": False})]]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    assert task["status"] == "failed" and task["stage"] == "agent"
    assert task["error"] == "The agent reported that it could not complete the task: This cannot be built here."
    assert github.remote.is_empty(blank) and github.calls("PUT", r"/contents/") == []


def test_a_project_the_agent_gave_up_on_is_pushed_as_unfinished(client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str):
    """What was built is not thrown away: it is pushed, and says that it is unfinished."""
    install_deps(ScriptedLLM([PROJECT, [("finish", {"summary": "This cannot be built here.", "success": False})]]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    assert task["status"] == "completed" and task["outcome"] == "initial_commit"
    assert task["unfinished"] is True and task["stop_reason"] == "gave_up"
    assert github.remote.files(blank, "main") == ["README.md", "app.js", "index.html"]
    assert github.remote.log(blank, "main")[0].startswith("[Unfinished] Build it")
    assert not any("discarded" in line for line in messages(task, "warn"))
    assert any(line.startswith("The project is unfinished: what the agent had built when it stopped was pushed") for line in messages(task, "warn"))


def test_build_output_does_not_cost_an_interrupted_run_its_work(
    client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str
):
    """The first build has run, the .gitignore is not written yet, and the run
    is cut short: the bundle is too large to commit. Everything else is kept."""
    bundle = sys.executable + " -c \"import os; os.makedirs('dist/assets'); open('dist/assets/index.js', 'w').write('x' * 1200000)\""
    install_deps(
        ScriptedLLM(
            [
                [
                    ("write_file", {"path": "package.json", "content": '{"name": "pulse", "version": "1.0.0", "scripts": {"build": "vite build"}}\n'}),
                    ("write_file", {"path": "src/main.js", "content": "console.log('pulse');\n"}),
                ],
                [("run_command", {"command": bundle})],
                LLMError("groq returned HTTP 429: Rate limit reached.", kind="rate_limit", status_code=429, retry_after=900.0),
            ]
        )
    )
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build Pulse"})
    assert task["status"] == "completed" and task["outcome"] == "initial_commit" and task["unfinished"] is True
    assert github.remote.files(blank, "main") == ["package.json", "src/main.js"]
    warnings = messages(task, "warn")
    assert "Not kept, because it is no part of the work or cannot be committed: dist/ (build output, 1 file)" in warnings
    assert not any("cannot be kept" in line or "discarded" in line for line in warnings)
    # The run that continues is told that the bundle is not in the repository.
    assert "Not committed (build output, or too large to commit): dist/ (build output, 1 file)" in task["handoff"]


def test_a_private_empty_repository_needs_no_clone_token(
    client: TestClient, app_env, install_deps, github: FakeGitHub
):
    github.add_repo("acme/secret", None, empty=True, private=True)
    workspaces: List[LocalWorkspace] = []
    install_deps(ScriptedLLM([PROJECT, FINISH]), workspaces)
    _events, task = run_stream(client, {"repository": "acme/secret", "prompt": "Build it"})
    assert task["outcome"] == "initial_commit"
    # Nothing is cloned, so no credential of any kind enters the sandbox.
    assert all(request.get("permissions") is None for request in github.token_requests)
    for environment in workspaces[0].environments:
        assert not any("AUTHORIZATION" in str(value) for value in environment.values())


# --------------------------------------------------------------------------- #
# End to end: a repository that only has a README
# --------------------------------------------------------------------------- #
def test_repository_with_only_a_readme_gets_a_new_project_as_a_pull_request(
    client: TestClient, token_env, install_deps, github: FakeGitHub
):
    github.add_repo("acme/fresh", {"README.md": "# fresh\n"})
    base = github.remote.rev("acme/fresh", "refs/heads/main")
    llm = ScriptedLLM([PROJECT, FINISH])
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/fresh", "prompt": "Build a small task list web app"})
    assert task["status"] == "completed" and task["outcome"] == "pull_request"
    assert task["new_project"] is True and task["empty_repository"] is False
    assert task["pr_url"] == "https://github.com/acme/fresh/pull/1" and task["commit_url"] is None
    # It has commits, so its default branch is not written to.
    assert github.remote.rev("acme/fresh", "refs/heads/main") == base
    assert sorted(github.remote.files("acme/fresh", task["head_branch"])) == ["README.md", "app.js", "index.html"]
    assert llm.requests[0][0]["content"] == system_prompt_for(new_project=True, first_commit=False)
    assert "acme/fresh has no code yet (only README.md): the agent builds the project from scratch" in messages(task)
    body = github.pulls["acme/fresh"][0]["body"]
    assert "The repository had no code yet: this pull request creates the project from scratch." in body
    assert "did not open any file" not in body and "Files the agent looked at" not in body


def test_existing_project_is_not_a_new_project(client: TestClient, token_env, install_deps, github: FakeGitHub):
    from .conftest import fix_add_script

    llm = fix_add_script()
    install_deps(llm)
    _events, task = run_stream(client, {"repository": "acme/demo", "prompt": "Fix add()"})
    assert task["outcome"] == "pull_request" and task["new_project"] is False and task["empty_repository"] is False
    assert llm.requests[0][0]["content"] == SYSTEM_PROMPT
    assert "Files the agent looked at" in github.pulls["acme/demo"][0]["body"]


# --------------------------------------------------------------------------- #
# Around the run: pre-flight check, recovery, issue comments
# --------------------------------------------------------------------------- #
def test_check_endpoint_says_when_a_repository_is_empty(
    client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str
):
    install_deps(ScriptedLLM([]))
    check = lambda **params: client.get("/api/v1/github/check", params=params).json()  # noqa: E731
    assert check(repository=blank) == {
        "ok": True,
        "repository": "acme/blank",
        "default_branch": "main",
        "private": False,
        "empty": True,
        "auth_mode": "token",
        "code": None,
        "error": None,
    }
    assert check(repository=blank, branch="main")["ok"] is True
    other = check(repository=blank, branch="develop")
    assert other["ok"] is False and other["empty"] is True and other["code"] == "branch_not_found"
    assert other["error"] == (
        "acme/blank is empty, so it has no branch 'develop'. Leave the base branch empty: the first commit is made on 'main'."
    )
    demo = check(repository="acme/demo")
    assert demo["ok"] is True and demo["empty"] is False
    assert check(repository="acme/demo", branch="nope")["code"] == "branch_not_found"


def test_first_commit_is_recovered_from_github(client: TestClient, token_env, install_deps, github: FakeGitHub, blank: str):
    """Another server instance answers the lookup: it only has GitHub to go by."""
    install_deps(ScriptedLLM([PROJECT, FINISH]))
    _events, task = run_stream(client, {"repository": blank, "prompt": "Build it"})
    task_store.clear()

    found = client.get(f"/api/v1/tasks/{task['task_id']}", params={"repository": blank}).json()
    assert found["recovered"] is True and found["status"] == "completed" and found["outcome"] == "initial_commit"
    assert found["commit_sha"] == task["commit_sha"] and found["commit_url"] == task["commit_url"]
    assert found["empty_repository"] is True and found["pr_url"] is None
    assert found["summary"] == "Create the task list app"
    assert found["logs"] == [f"Recovered from GitHub: first commit {task['commit_sha'][:10]}"]

    # There is no pull request to report a state for.
    state = client.get(f"/api/v1/tasks/{task['task_id']}/pull-request", params={"repository": blank})
    assert state.status_code == 404 and state.json()["detail"]["code"] == "pull_request_not_found"
    # A task that never ran is still unknown, in an empty repository too.
    github.add_repo("acme/blank2", None, empty=True)
    for repository in (blank, "acme/blank2"):
        missing = client.get(f"/api/v1/tasks/{'0' * 32}", params={"repository": repository})
        assert missing.status_code == 404 and "no pull request or commit" in missing.json()["detail"]["message"]


def test_issue_comment_for_a_first_commit():
    task = {
        "task_id": "t" * 32,
        "status": "completed",
        "outcome": "initial_commit",
        "commit_sha": "a" * 40,
        "commit_url": "https://github.com/acme/blank/commit/" + "a" * 40,
        "base_branch": "main",
        "summary": "A static task list app.",
        "tests_status": "not_run",
        "docs_only": False,
    }
    comment = build_result_comment(task)
    assert comment.startswith(
        "✅ The repository was empty, so there was nothing to open a pull request against. I committed the "
        "project directly to `main`: https://github.com/acme/blank/commit/" + "a" * 40
    )
    assert "A static task list app." in comment and "The new project has no automated tests yet." in comment
    assert "⚠️ The last test run failed" in build_result_comment({**task, "tests_status": "failed"})
    assert "Tests passed in the sandbox." in build_result_comment({**task, "tests_status": "passed"})
    assert "automated tests" not in build_result_comment({**task, "docs_only": True})


def test_listing_endpoint_returns_every_repository(client: TestClient, app_env, install_deps, github: FakeGitHub):
    install_deps(ScriptedLLM([]))
    for index in range(250):
        github.add_listed_repo(f"acme/repo-{index:03d}", language="TypeScript")
    github.installations[github.installation_for("acme")]["repository_selection"] = "all"
    body = client.get("/api/v1/github/repos").json()
    assert len(body["repositories"]) == 251 and body["total"] == 251
    assert body["truncated"] is False and body["error"] is None
    # Most recently pushed first.
    assert [repo["full_name"] for repo in body["repositories"][:2]] == ["acme/repo-249", "acme/repo-248"]
    assert body["repositories"][0]["language"] == "TypeScript" and body["repositories"][0]["archived"] is False
    assert body["installations"] == [
        {
            "id": github.installation_for("acme"),
            "account": "acme",
            "account_type": "User",
            "repository_selection": "all",
            "manage_url": f"https://github.com/settings/installations/{github.installation_for('acme')}",
            "suspended": False,
            "repository_count": 251,
        }
    ]
