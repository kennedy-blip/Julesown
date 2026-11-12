"""The agent loop and its tools, running against a real git clone on disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from apps.api.services.agent import (
    AGENT_TOOLS,
    MAX_FORMAT_RETRIES,
    SYSTEM_PROMPT,
    TOOL_NAMES,
    CodingAgent,
    compact_history,
    history_size,
)
from apps.api.services.common import Deadline, Redactor, clip, plain_text
from apps.api.services.llm import LLMError, LLMResponse, ToolCall
from apps.api.services.sandbox import KEY_FILE_BYTES, KEY_FILES_TOTAL_BYTES, README_BYTES, RepoCheckout, SandboxError

from .conftest import CHECK_ADD, DEMO_FILES
from .support import GitRemote, LocalWorkspace, ScriptedLLM, git


@pytest.fixture
def checkout(remote: GitRemote, tmp_path) -> RepoCheckout:
    remote.create("acme/demo", DEMO_FILES)
    workspace = LocalWorkspace(tmp_path / "sandbox")
    return RepoCheckout.clone(workspace, clone_url=remote.clone_url("acme/demo"), branch="main")


def make_agent(checkout: RepoCheckout, llm: ScriptedLLM, **options: Any) -> Tuple[CodingAgent, List[Tuple[str, str]]]:
    events: List[Tuple[str, str]] = []
    options.setdefault("deadline", Deadline(300))
    options.setdefault("sleep", lambda _seconds: None)
    agent = CodingAgent(llm=llm, checkout=checkout, emit=lambda level, message: events.append((level, message)), **options)
    return agent, events


def run(agent: CodingAgent, prompt: str = "Fix add()", **options: Any):
    return agent.run(
        prompt=prompt, repository="acme/demo", branch="main", file_listing="Repository files:\ncalc.py", **options
    )


def checkout_of(remote: GitRemote, tmp_path, files: dict, name: str = "acme/other") -> RepoCheckout:
    remote.create(name, files)
    workspace = LocalWorkspace(tmp_path / f"sandbox-{name.replace('/', '-')}")
    return RepoCheckout.clone(workspace, clone_url=remote.clone_url(name), branch="main")


# npm's placeholder test script, without needing npm.
NO_TESTS_COMMAND = """sh -c 'echo "Error: no test specified" && exit 1'"""


# --------------------------------------------------------------------------- #
# Checkout
# --------------------------------------------------------------------------- #
def test_clone_records_base_commit_and_lists_files(checkout: RepoCheckout, remote: GitRemote):
    assert checkout.base_sha == remote.rev("acme/demo", "refs/heads/main")
    assert len(checkout.base_tree_sha) == 40
    files, total = checkout.list_files()
    assert total == 4 and "calc.py" in files and "tests/test_calc.py" in files
    assert checkout.collect_changes().empty


def test_clone_failure_is_reported(remote: GitRemote, tmp_path):
    remote.create("acme/demo", DEMO_FILES)
    workspace = LocalWorkspace(tmp_path / "sandbox")
    with pytest.raises(SandboxError) as caught:
        RepoCheckout.clone(workspace, clone_url=remote.clone_url("acme/demo"), branch="no-such-branch")
    assert "git clone failed" in str(caught.value)


def test_clone_quotes_hostile_branch_names(remote: GitRemote, tmp_path):
    """Branch names reach a shell; they must never be interpreted by it."""
    bare = remote.create("acme/demo", DEMO_FILES)
    hostile = "x$(touch${IFS}pwned)`id`"  # a legal git ref name
    git(bare, "branch", hostile, "main")
    workspace = LocalWorkspace(tmp_path / "sandbox")
    cloned = RepoCheckout.clone(workspace, clone_url=remote.clone_url("acme/demo"), branch=hostile)
    assert cloned.base_sha == remote.rev("acme/demo", "refs/heads/main")
    assert not list(Path(workspace.workdir).rglob("pwned"))


def test_clone_passes_the_token_through_the_environment_only(remote: GitRemote, tmp_path):
    remote.create("acme/demo", DEMO_FILES)
    workspace = LocalWorkspace(tmp_path / "sandbox")
    token = "ghs_" + "s" * 36
    cloned = RepoCheckout.clone(workspace, clone_url=remote.clone_url("acme/demo"), branch="main", token=token)

    assert all(token not in command for command in workspace.commands)
    clone_index = next(i for i, command in enumerate(workspace.commands) if "git clone" in command)
    clone_env = workspace.environments[clone_index]
    assert clone_env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert clone_env["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: basic ")
    # Only the clone command ever sees it, and nothing is persisted in the clone.
    assert all("GIT_CONFIG_VALUE_0" not in env for i, env in enumerate(workspace.environments) if i != clone_index)
    config = (Path(cloned.directory) / ".git" / "config").read_text()
    assert "extraheader" not in config.lower() and token not in config


def test_collect_changes_describes_adds_edits_deletes_and_modes(checkout: RepoCheckout):
    root = Path(checkout.directory)
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "README.md").unlink()
    (root / "docs").mkdir()
    (root / "docs" / "notes ünïcode.md").write_text("héllo\n", encoding="utf-8")
    (root / "run.sh").write_text("#!/bin/sh\necho hi\n")
    (root / "run.sh").chmod(0o755)
    (root / "blob.bin").write_bytes(bytes(range(256)) * 4)
    # Build output and dependencies must never be committed.
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "index.js").write_text("module.exports = 1\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "calc.cpython-311.pyc").write_bytes(b"\x00\x01")
    (root / "debug.log").write_text("ignored by the repository's own .gitignore\n")

    changeset = checkout.collect_changes()
    by_path = {change.path: change for change in changeset.changes}
    assert sorted(by_path) == ["README.md", "blob.bin", "calc.py", "docs/notes ünïcode.md", "run.sh"]
    assert (by_path["calc.py"].status, by_path["calc.py"].additions, by_path["calc.py"].deletions) == ("M", 1, 1)
    assert by_path["README.md"].status == "D" and by_path["README.md"].content_b64 is None
    assert by_path["run.sh"].mode == "100755" and by_path["docs/notes ünïcode.md"].mode == "100644"
    assert by_path["blob.bin"].additions is None  # binary
    import base64

    assert base64.b64decode(by_path["blob.bin"].content_b64) == bytes(range(256)) * 4
    assert base64.b64decode(by_path["docs/notes ünïcode.md"].content_b64).decode("utf-8") == "héllo\n"
    # The tree sha is what `git write-tree` produced for exactly these changes.
    assert changeset.tree_sha == git(root, "write-tree")
    assert changeset.total_bytes > 1024


def test_collect_changes_includes_work_committed_inside_the_sandbox(checkout: RepoCheckout):
    """If a command commits (the model is told not to), nothing is lost: the
    diff is taken against the commit that was cloned, not against HEAD."""
    root = Path(checkout.directory)
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git(root, "add", "-A")
    git(root, "commit", "--quiet", "-m", "sneaky commit")
    (root / "extra.txt").write_text("uncommitted\n")
    paths = sorted(change.path for change in checkout.collect_changes().changes)
    assert paths == ["calc.py", "extra.txt"]


def test_oversized_change_is_refused(checkout: RepoCheckout):
    (Path(checkout.directory) / "huge.bin").write_bytes(b"0" * 1_200_000)
    with pytest.raises(SandboxError) as caught:
        checkout.collect_changes()
    assert "huge.bin" in str(caught.value)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_agent_fixes_bug_and_reports_real_tool_output(checkout: RepoCheckout):
    llm = ScriptedLLM(
        [
            [("list_files", {})],
            [("read_file", {"path": "calc.py"})],
            [("search_code", {"query": "add("})],
            [("run_tests", {"command": CHECK_ADD})],  # fails before the fix
            [("replace_in_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"})],
            [("run_tests", {"command": CHECK_ADD})],  # passes after the fix
            [("finish", {"summary": "Fixed the operator.", "pr_title": "Fix add()"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)

    assert outcome.finished and outcome.turns == 7
    assert (outcome.summary, outcome.pr_title) == ("Fixed the operator.", "Fix add()")
    assert [verification.passed for verification in outcome.verification] == [False, True]
    assert outcome.tests_status == "passed"

    listing, source, matches, failed, edited, passed = llm.tool_results()
    assert "calc.py" in listing.splitlines() and "tests/test_calc.py" in listing
    assert source.startswith("calc.py - lines 1-2 of 2\n") and "return a - b" in source
    assert "calc.py:1:def add(a, b):" in matches
    assert failed.startswith("exit code: 1") and "AssertionError" in failed
    assert edited == "Updated calc.py."
    assert passed.startswith("exit code: 0")
    assert (Path(checkout.directory) / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"

    levels = [level for level, _message in events]
    assert "command" in levels and "edit" in levels and "error" in levels and "success" in levels
    assert ("command", f"$ {CHECK_ADD}") in events


def test_first_message_contains_task_repository_and_files(checkout: RepoCheckout):
    llm = ScriptedLLM([[("finish", {"summary": "nothing to do"})]])
    agent, _events = make_agent(checkout, llm)
    run(agent, prompt="Please fix add()")
    system, user = llm.requests[0][:2]
    assert system["role"] == "system" and "finish" in system["content"]
    assert "acme/demo" in user["content"] and "Please fix add()" in user["content"]
    assert checkout.base_sha[:10] in user["content"] and "calc.py" in user["content"]


def test_tool_errors_are_returned_to_the_model(checkout: RepoCheckout):
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "missing.py"})],
            [("read_file", {"path": "../../etc/passwd"})],
            [("read_file", {"path": "/etc/passwd"})],
            [("write_file", {"path": ".git/hooks/pre-commit", "content": "#!/bin/sh\n"})],
            [("replace_in_file", {"path": "calc.py", "old_text": "not there", "new_text": "x"})],
            [("replace_in_file", {"path": "tests/test_calc.py", "old_text": "add", "new_text": "sub"})],
            [("read_file", {"path": "tests"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    assert run(agent).finished
    results = llm.tool_results()
    assert results[0] == "Error: missing.py does not exist"
    assert results[1] == "Error: ../../etc/passwd is outside the repository"
    assert results[2] == "Error: /etc/passwd is outside the repository"
    assert results[3] == "Error: the .git directory is managed by the platform"
    assert results[4].startswith("Error: old_text was not found in calc.py")
    assert results[5].startswith("Error: old_text occurs 3 times in tests/test_calc.py")
    assert results[6] == "Error: tests is a directory; use list_files"
    assert not (Path(checkout.directory) / ".git" / "hooks" / "pre-commit").exists()


def test_read_file_ranges_binary_and_new_files(checkout: RepoCheckout):
    root = Path(checkout.directory)
    (root / "long.txt").write_text("".join(f"line {n}\n" for n in range(1, 1001)))
    (root / "data.bin").write_bytes(b"\x00\x01\x02\xff" * 10)
    (root / "blank.txt").write_text("\n\n\n")
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "long.txt"})],
            [("read_file", {"path": "long.txt", "start_line": 990, "end_line": 2000})],
            [("read_file", {"path": "data.bin"})],
            [("read_file", {"path": "blank.txt"})],
            [("write_file", {"path": "pkg/new/module.py", "content": "VALUE = 1\n"})],
            [("search_code", {"query": "VALUE = 1"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    run(agent)
    first, tail, binary, blank, wrote, found = llm.tool_results()
    assert first.startswith("long.txt - lines 1-") and "\nline 1\nline 2\n" in first
    assert "[more lines follow; call read_file with start_line=" in first
    assert tail.startswith("long.txt - lines 990-1000 of 1000") and tail.rstrip().endswith("line 1000")
    assert binary == "data.bin is a binary file (40 bytes); it cannot be shown as text."
    assert blank.startswith("blank.txt - lines 1-3 of 3")  # blank lines are text, not binary
    assert wrote == "Wrote pkg/new/module.py (1 lines)."
    assert found == "pkg/new/module.py:1:VALUE = 1"  # untracked files are searchable


def test_replace_in_file_refuses_to_corrupt_non_utf8_files(checkout: RepoCheckout):
    target = Path(checkout.directory) / "legacy.txt"
    original = "caf\xe9 = 1\nvalue = 2\n".encode("latin-1")
    target.write_bytes(original)
    llm = ScriptedLLM(
        [
            [("replace_in_file", {"path": "legacy.txt", "old_text": "value = 2", "new_text": "value = 3"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    run(agent)
    assert "not valid UTF-8" in llm.tool_results()[0] and "run_command" in llm.tool_results()[0]
    assert target.read_bytes() == original  # untouched


def test_commands_run_in_the_repository_with_a_time_limit(checkout: RepoCheckout):
    llm = ScriptedLLM(
        [
            [("run_command", {"command": "pwd && ls calc.py && echo $CI"})],
            [("run_command", {"command": "sleep 30", "timeout_seconds": 1})],
            [("run_command", {"command": "printf '\\033[31mred\\033[0m\\n'; exit 3"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm, command_timeout=6)
    outcome = run(agent)
    where, slow, coloured = llm.tool_results()
    assert where.splitlines() == ["exit code: 0", checkout.directory, "calc.py", "true"]
    assert slow.startswith("timed out after 5 seconds")
    assert coloured == "exit code: 3\nred"  # ANSI colour codes are stripped
    assert outcome.verification == []  # run_command is not a verification run
    assert outcome.tests_status == "not_run"
    assert ("error", "timed out after 5s") in events


# --------------------------------------------------------------------------- #
# Test commands that have nothing to test
# --------------------------------------------------------------------------- #
def test_test_command_with_nothing_to_test_is_not_a_failure(checkout: RepoCheckout):
    """Regression (first production run): `npm test` ran npm's placeholder
    script, exited with 1 and was reported as "the last test run failed"."""
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "README.md"})],
            [("run_tests", {"command": NO_TESTS_COMMAND})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)

    (verification,) = outcome.verification
    assert (verification.exit_code, verification.outcome, verification.passed) == (1, "no_tests", False)
    assert verification.no_tests == "the \"test\" script in package.json is npm's placeholder"
    assert outcome.tests_status == "no_tests"
    # The model is told that this is not its failure, so it does not try to "fix" it.
    result = llm.tool_results()[-1]
    assert result.startswith("exit code: 1\nError: no test specified")
    assert "There are no tests to run here" in result and "nothing to fix" in result
    # ...and the log does not show a red failure.
    assert not [message for level, message in events if level == "error"]
    assert any(
        level == "info" and message.startswith("exit 1 · ") and message.endswith("no tests to run: " + verification.no_tests)
        for level, message in events
    )


def test_a_real_failure_after_an_empty_run_is_still_a_failure(checkout: RepoCheckout):
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"})],
            [("run_tests", {"command": CHECK_ADD})],  # fails: add() subtracts
            [("run_tests", {"command": NO_TESTS_COMMAND})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert [item.outcome for item in outcome.verification] == ["failed", "no_tests"]
    assert outcome.tests_status == "failed"
    assert "There are no tests to run here" not in llm.tool_results()[1]
    assert len([message for level, message in events if level == "error" and message.startswith("exit 1 · ")]) == 1


def test_run_command_is_never_classified(checkout: RepoCheckout):
    """Only run_tests is a verification run; the same output from run_command is just output."""
    llm = ScriptedLLM([[("run_command", {"command": NO_TESTS_COMMAND})], [("finish", {"summary": "done"})]])
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.verification == [] and outcome.tests_status == "not_run"
    assert llm.tool_results()[0] == "exit code: 1\nError: no test specified"
    assert any(level == "error" and message.startswith("exit 1") for level, message in events)


# --------------------------------------------------------------------------- #
# Looking before writing
# --------------------------------------------------------------------------- #
def test_writing_before_reading_anything_is_refused_once(checkout: RepoCheckout):
    """Regression (first production run): asked to document the project's
    dependencies, the model wrote the file as its very first action - without
    opening package.json - and every dependency it listed was invented."""
    spec = "# Spec\n\nDepends on react and webpack.\n"
    llm = ScriptedLLM(
        [
            [("write_file", {"path": "spec.md", "content": spec})],
            [("list_files", {})],  # a listing shows names, not content: still nothing was read
            [("read_file", {"path": "calc.py"})],
            [("write_file", {"path": "spec.md", "content": "# Spec\n\nA calculator.\n"})],
            [("write_file", {"path": "more.md", "content": "more\n"})],  # no second refusal
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    refused, _listing, _source, wrote, wrote_more = llm.tool_results()
    assert refused.startswith("Error: spec.md was not written: you have not opened any file of this repository yet.")
    assert "call write_file again" in refused
    assert wrote == "Wrote spec.md (3 lines)." and wrote_more == "Wrote more.md (1 lines)."
    assert (Path(checkout.directory) / "spec.md").read_text() == "# Spec\n\nA calculator.\n"
    assert outcome.finished and outcome.files_read == ["calc.py"]
    # The log line is short; the full explanation goes to the model only.
    assert (
        "info",
        "write_file spec.md held back: nothing has been read yet - the agent was asked to look at the repository first",
    ) in events
    assert not [message for level, message in events if level in ("warn", "error")]  # steering, not a problem


def test_a_model_that_insists_may_write_without_reading(checkout: RepoCheckout):
    """Some files need no knowledge of the repository (a licence text, say)."""
    licence = "MIT License\n"
    llm = ScriptedLLM(
        [
            [("write_file", {"path": "LICENSE", "content": licence})],
            [("write_file", {"path": "LICENSE", "content": licence})],
            [("finish", {"summary": "Added the licence."})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent)
    assert llm.tool_results()[1] == "Wrote LICENSE (1 lines)."
    assert outcome.finished and outcome.files_read == []
    assert (Path(checkout.directory) / "LICENSE").read_text() == licence


def test_existing_file_is_not_overwritten_unread(checkout: RepoCheckout):
    readme = Path(checkout.directory) / "README.md"
    original = readme.read_text()
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"})],
            [("write_file", {"path": "README.md", "content": "# New\n"})],  # README.md was never read
            [("read_file", {"path": "README.md"})],
            [("write_file", {"path": "README.md", "content": original + "\nMore.\n"})],
            [("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})],  # was read: fine
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    assert run(agent).finished
    _source, refused, _readme, wrote, wrote_calc = llm.tool_results()
    assert refused.startswith("Error: README.md already exists (3 lines) and you have not read it in this run")
    assert "replace_in_file" in refused
    assert wrote == "Wrote README.md (5 lines)." and wrote_calc == "Wrote calc.py (2 lines)."
    assert readme.read_text() == original + "\nMore.\n"
    assert (
        "info",
        "write_file README.md held back: the existing file has not been read - the agent was asked to read it first",
    ) in events


def test_filling_an_empty_file_is_not_an_overwrite(checkout: RepoCheckout):
    """Kensheets, the repository of the first production run, has a 0-byte README.md."""
    (Path(checkout.directory) / "EMPTY.md").write_text("")
    (Path(checkout.directory) / "one-line.txt").write_text("no newline at the end")
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"})],
            [("write_file", {"path": "EMPTY.md", "content": "# Now with content\n"})],
            [("write_file", {"path": "one-line.txt", "content": "replaced\n"})],
            [("finish", {"summary": "done"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    assert run(agent).finished
    _source, wrote, refused = llm.tool_results()[:3]
    assert wrote == "Wrote EMPTY.md (1 lines)."
    assert refused.startswith("Error: one-line.txt already exists (1 lines) and you have not read it")


def test_overwriting_unread_is_accepted_when_repeated_and_never_asked_twice(checkout: RepoCheckout):
    llm = ScriptedLLM(
        [
            [("search_code", {"query": "Demo"})],
            [("write_file", {"path": "README.md", "content": "# One\n"})],
            [("write_file", {"path": "README.md", "content": "# Two\n"})],
            [("write_file", {"path": "README.md", "content": "# Three\n"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    assert run(agent).finished
    results = llm.tool_results()
    assert results[1].startswith("Error: README.md already exists")
    assert results[2:] == ["Wrote README.md (1 lines).", "Wrote README.md (1 lines)."]
    assert (Path(checkout.directory) / "README.md").read_text() == "# Three\n"


def test_finish_is_questioned_when_a_refused_write_was_never_repeated(checkout: RepoCheckout):
    """A model that ignores the refusal and reports success has not done the work."""
    llm = ScriptedLLM(
        [
            [("write_file", {"path": "spec.md", "content": "# Spec\n"})],
            [("finish", {"summary": "Created spec.md."})],
            [("read_file", {"path": "calc.py"})],
            [("write_file", {"path": "spec.md", "content": "# Spec\n\nA calculator.\n"})],
            [("finish", {"summary": "Created spec.md.", "pr_title": "Add spec"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    questioned = llm.tool_results()[1]
    assert questioned.startswith("Not finished: spec.md has not been written")
    assert outcome.finished and outcome.turns == 5 and outcome.pr_title == "Add spec"
    assert (Path(checkout.directory) / "spec.md").exists()
    assert ("warn", "finish held back: spec.md was never written - the agent was asked to complete it") in events


def test_finish_is_questioned_only_once(checkout: RepoCheckout):
    llm = ScriptedLLM(
        [
            [("write_file", {"path": "spec.md", "content": "# Spec\n"})],
            [("finish", {"summary": "Not needed after all."})],
            [("finish", {"summary": "Not needed after all."})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 3 and outcome.summary == "Not needed after all."
    assert not (Path(checkout.directory) / "spec.md").exists()


def test_replace_in_file_settles_a_refused_overwrite(checkout: RepoCheckout):
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"})],
            [("write_file", {"path": "README.md", "content": "# New\n"})],  # refused: unread
            [("replace_in_file", {"path": "README.md", "old_text": "projct", "new_text": "project"})],
            [("finish", {"summary": "Fixed the typo."})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 4  # finish was not questioned
    assert "A small demo project." in (Path(checkout.directory) / "README.md").read_text()


# --------------------------------------------------------------------------- #
# Key files given with the task
# --------------------------------------------------------------------------- #
def test_key_files_are_the_manifests_and_the_start_of_the_readme(remote: GitRemote, tmp_path):
    manifest = '{\n  "name": "demo",\n  "scripts": {"test": "jest"},\n  "dependencies": {"left-pad": "^1.3.0"}\n}\n'
    readme = "# Demo\n\n" + "".join(f"Paragraph {n} of the introduction.\n" for n in range(60))
    checkout = checkout_of(
        remote,
        tmp_path,
        {
            "package.json": manifest,
            "README.md": readme,
            "requirements.txt": "",  # empty: nothing to show
            "src/index.js": "module.exports = 1\n",
            "sub/package.json": '{"name": "nested"}\n',  # only the repository root is looked at
        },
    )
    root = Path(checkout.directory)
    (root / "Makefile").write_bytes(b"all:\n\t\x00\xff\n")  # binary: skipped
    (root / "go.mod").symlink_to("/etc/hostname")  # a link out of the repository: skipped

    files = checkout.key_files()
    assert [item.path for item in files] == ["package.json", "README.md"]
    package, intro = files
    assert (package.content, package.size, package.truncated) == (manifest, len(manifest), False)
    assert intro.truncated and intro.size == len(readme)
    assert readme.startswith(intro.content) and intro.content.endswith("\n")  # whole lines only
    assert 0 < len(intro.content) <= README_BYTES


def test_key_files_stay_within_their_budget(remote: GitRemote, tmp_path):
    big = "x = 1\n" * 2000
    checkout = checkout_of(
        remote,
        tmp_path,
        {"pyproject.toml": big, "requirements.txt": "httpx\n", "setup.py": big, "Cargo.toml": big, "README.rst": "Demo\n"},
    )
    files = checkout.key_files()
    assert all(len(item.content) <= KEY_FILE_BYTES for item in files)
    assert sum(len(item.content) for item in files) <= KEY_FILES_TOTAL_BYTES
    by_path = {item.path: item for item in files}
    assert by_path["pyproject.toml"].truncated and by_path["requirements.txt"].content == "httpx\n"
    assert "README.rst" in by_path  # small files still fit after a large one was skipped

    assert checkout_of(remote, tmp_path, {"main.c": "int main(void) { return 0; }\n"}, "acme/bare").key_files() == []


def test_key_files_are_part_of_the_first_message(remote: GitRemote, tmp_path):
    manifest = '{"name": "demo", "license": "ISC", "dependencies": {"papaparse": "^5.5.3"}}\n'
    checkout = checkout_of(remote, tmp_path, {"package.json": manifest, "README.md": "# Demo\n" + "More text.\n" * 100})
    llm = ScriptedLLM(
        [
            [("write_file", {"path": "package.json", "content": manifest})],  # shown in full: known
            [("write_file", {"path": "README.md", "content": "# Demo\n"})],  # shown only in part: unread
            [("finish", {"summary": "done"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent, key_files=checkout.key_files())

    first = llm.requests[0][1]["content"]
    assert "--- package.json (%d bytes) ---\n%s" % (len(manifest), manifest.rstrip()) in first
    assert "--- README.md (beginning only; the file has 1107 bytes) ---\n# Demo\nMore text." in first
    assert "use read_file for any other file" in first
    assert ("tool", "Shown to the agent up front: package.json, README.md (beginning)") in events
    # Seeing the first lines of a file is not having read it.
    assert outcome.files_read == ["package.json"]
    wrote, refused = llm.tool_results()[:2]
    assert wrote == "Wrote package.json (1 lines)."
    assert refused.startswith("Error: README.md already exists")


def test_being_shown_the_manifest_is_not_the_same_as_looking(remote: GitRemote, tmp_path):
    """The manifest is in the first message, but a file that describes the
    project still needs the model to open something on its own."""
    checkout = checkout_of(remote, tmp_path, {"package.json": '{"name": "demo"}\n', "src/app.js": "export const a = 1;\n"})
    llm = ScriptedLLM(
        [
            [("write_file", {"path": "spec.md", "content": "# Spec\n"})],
            [("read_file", {"path": "src/app.js"})],
            [("write_file", {"path": "spec.md", "content": "# Spec\n\nExports a.\n"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent, key_files=checkout.key_files())
    assert llm.tool_results()[0].startswith("Error: spec.md was not written")
    assert outcome.finished and outcome.files_read == ["package.json", "src/app.js"]


def test_system_prompt_states_the_rules_the_first_run_broke():
    for fragment in (
        "Look before you write",
        "Never invent dependencies",
        'npm\'s placeholder that only prints "Error: no test specified"',
        "A change that only touches documentation needs no install, build or test run",
        "Reply only by calling the provided tools; no other tools exist",
    ):
        assert fragment in " ".join(SYSTEM_PROMPT.split()), fragment
    assert TOOL_NAMES == tuple(tool["function"]["name"] for tool in AGENT_TOOLS) and "finish" in TOOL_NAMES


# --------------------------------------------------------------------------- #
# Loop control
# --------------------------------------------------------------------------- #
def test_text_only_replies_are_nudged_then_accepted_as_done(checkout: RepoCheckout):
    llm = ScriptedLLM(["I will look at the code.", "Still thinking.", "All done: nothing needed changing."])
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.summary == "All done: nothing needed changing."
    nudges = [m for m in llm.requests[-1] if m["role"] == "user" and "calling a tool" in m["content"]]
    assert len(nudges) == 2


def test_agent_can_give_up(checkout: RepoCheckout):
    llm = ScriptedLLM([[("finish", {"summary": "The repository has no such module.", "success": False})]])
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.stop_reason == "gave_up" and not outcome.finished
    assert outcome.error == "The repository has no such module."


def test_max_turns_and_deadline_stop_the_loop(checkout: RepoCheckout):
    endless = ScriptedLLM([[("list_files", {})]] * 10)
    agent, _events = make_agent(checkout, endless, max_turns=3)
    outcome = run(agent)
    assert outcome.stop_reason == "max_turns" and outcome.turns == 3 and "3 turns" in (outcome.error or "")

    agent, _events = make_agent(checkout, ScriptedLLM([[("list_files", {})]]), deadline=Deadline(20), reserve_seconds=25)
    outcome = run(agent)
    assert outcome.stop_reason == "timeout" and outcome.turns == 0


def test_malformed_tool_call_is_fed_back(checkout: RepoCheckout):
    from apps.api.services.llm import LLMResponse, ToolCall

    def broken(_messages):
        return None

    class OneBadCall(ScriptedLLM):
        def complete(self, messages, tools, *, max_output_tokens, timeout):
            if len(self.requests) == 0:
                self.requests.append(json.loads(json.dumps(messages)))
                wire = [{"id": "c0", "type": "function", "function": {"name": "read_file", "arguments": "{oops"}}]
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="c0", name="read_file", arguments={}, error="arguments were not valid JSON (x)")],
                    finish_reason="tool_calls",
                    model=self.model,
                    assistant_message={"role": "assistant", "content": None, "tool_calls": wire},
                )
            return super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)

    llm = OneBadCall([[("finish", {"summary": "ok"})]])
    agent, events = make_agent(checkout, llm)
    assert run(agent).finished
    assert "Call the tool again with a valid JSON object" in llm.tool_results()[0]
    # The unreadable arguments are not replayed to the provider (it would refuse
    # every later request); the call itself stays, answered by the error above.
    (stored,) = [m for m in llm.requests[-1] if m["role"] == "assistant"]
    assert stored["tool_calls"][0]["function"] == {"name": "read_file", "arguments": "{}"}
    assert any(level == "warn" and "read_file call was unreadable" in message for level, message in events)


# What Groq answered in production (2026-11-11) when openai/gpt-oss-120b tried to
# "call" the channels of its own output format; see test_llm.py for the wire form.
def refused_reply(tool: str, generation: str = "") -> LLMError:
    return LLMError(
        f"groq refused the model's reply because it was not a valid tool call (it tried to call a tool named '{tool}', which does not exist).",
        kind="tool_format",
        status_code=400,
        rejected_tool=tool,
        failed_generation=generation or None,
    )


def test_refused_reply_is_explained_to_the_model_and_asked_again(checkout: RepoCheckout):
    """Regression (first production run): two such replies in a row used up the
    whole retry budget; a third would have failed the task."""
    waits: List[float] = []
    generation = '{"name": "json", "arguments": {"summary": "Added spec.md"}}'
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"})],
            refused_reply("commentary"),
            refused_reply("json", generation),
            refused_reply("commentary"),
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm, sleep=waits.append)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 2
    assert MAX_FORMAT_RETRIES == 3 and waits == [0.25, 0.5, 0.75]

    first_try, second_try, third_try, fourth_try = llm.requests[1:]
    assert first_try[-1]["role"] == "tool"  # the original request: no correction yet
    correction = second_try[-1]
    assert correction["role"] == "user" and second_try[:-1] == first_try
    assert "nothing was executed: it called a tool named 'commentary', which does not exist." in correction["content"]
    assert f"Reply again with a call to one of these tools: {', '.join(TOOL_NAMES)}." in correction["content"]
    assert "summary argument of finish" in correction["content"]
    # The next correction quotes what the model wrote, so it can send it again properly.
    assert "'json'" in third_try[-1]["content"] and generation in third_try[-1]["content"]
    # One correction at a time, and none of them stays in the history.
    assert third_try[:-1] == first_try and fourth_try[:-1] == first_try
    assert [level for level, _message in events if level == "warn"] == ["warn"] * 3
    assert (
        "warn",
        "The model's reply was not a valid tool call (it called 'commentary', which is not one of its tools); asking it again (1/3)",
    ) in events
    assert events[-1][0] != "error"


def test_refused_replies_do_not_spend_the_budget_for_outages(checkout: RepoCheckout):
    busy = LLMError("groq returned HTTP 503: busy", kind="server", status_code=503)
    llm = ScriptedLLM([busy, refused_reply("json"), busy, refused_reply("json"), refused_reply("json"), [("finish", {"summary": "ok"})]])
    agent, _events = make_agent(checkout, llm)
    assert run(agent).finished and len(llm.requests) == 6


def test_model_that_only_produces_refused_replies_fails_with_advice(checkout: RepoCheckout):
    llm = ScriptedLLM([refused_reply("json")] * (MAX_FORMAT_RETRIES + 1), model="openai/gpt-oss-120b")
    llm.model_env_var = "GROQ_MODEL"
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.stop_reason == "llm_error" and len(llm.requests) == MAX_FORMAT_RETRIES + 1
    assert outcome.error == (
        "The model (openai/gpt-oss-120b) answered 4 times in a row with an invalid tool call (it called "
        "'json', which is not one of its tools), so the task could not continue. Nothing was wrong with "
        "the task itself: run it again, or choose another model with GROQ_MODEL. With a second "
        "provider configured (Gemini or OpenRouter, in Settings) the agent switches to it when a "
        "model keeps doing this."
    )


def test_refusal_without_a_made_up_tool_passes_the_providers_reason_on(checkout: RepoCheckout):
    reason = "groq refused the model's reply because it was not a valid tool call (parameters for tool write_file did not match schema)."
    llm = ScriptedLLM([LLMError(reason, kind="tool_format", status_code=400), [("finish", {"summary": "ok"})]])
    agent, events = make_agent(checkout, llm)
    assert run(agent).finished
    assert "it was not a valid tool call. The provider reported: " + reason in llm.requests[1][-1]["content"]
    assert events[-1] == ("warn", f"The model's reply was not a valid tool call ({reason[:-1]}); asking it again (1/3)")


def test_call_to_a_tool_that_does_not_exist_is_not_kept_in_the_history(checkout: RepoCheckout):
    """Providers validate the history too: one made-up tool call left in it and
    every later request of the run is refused."""
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "calc.py"}), ("commentary", {"text": "looks wrong"})],
            [("browser.search", {"query": "python add"})],
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 3

    history = llm.requests[-1]
    called = [call["function"]["name"] for m in history if m["role"] == "assistant" for call in m.get("tool_calls") or []]
    assert called == ["read_file"]  # the real call stays, with its result
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool", "user", "user"]
    told = [m["content"] for m in history[4:]]
    assert told[0] == f"Error: unknown tool 'commentary'. Nothing was executed. Available tools: {', '.join(TOOL_NAMES)}."
    assert told[1].startswith("Error: unknown tool 'browser.search'.")
    assert ("warn", "The model called a tool that does not exist (commentary); it was told which tools there are") in events


def test_text_next_to_a_made_up_tool_call_is_kept(checkout: RepoCheckout):
    class TalksAndInvents(ScriptedLLM):
        def complete(self, messages, tools, *, max_output_tokens, timeout):
            if not self.requests:
                self.requests.append(json.loads(json.dumps(messages)))
                wire = [{"id": "c0", "type": "function", "function": {"name": "json", "arguments": "{}"}}]
                return LLMResponse(
                    content="I will answer in JSON.",
                    tool_calls=[ToolCall(id="c0", name="json", arguments={})],
                    finish_reason="tool_calls",
                    model=self.model,
                    assistant_message={
                        "role": "assistant",
                        "content": "I will answer in JSON.",
                        "tool_calls": wire,
                        "_anthropic_content": [{"type": "tool_use", "id": "c0", "name": "json", "input": {}}],
                    },
                )
            return super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)

    llm = TalksAndInvents([[("finish", {"summary": "done"})]])
    agent, _events = make_agent(checkout, llm)
    assert run(agent).finished
    assert llm.requests[-1][2] == {"role": "assistant", "content": "I will answer in JSON."}


def test_empty_reply_is_nudged_but_not_stored(checkout: RepoCheckout):
    llm = ScriptedLLM(["", [("finish", {"summary": "done"})]])
    agent, _events = make_agent(checkout, llm)
    assert run(agent).finished
    assert [m["role"] for m in llm.requests[-1]] == ["system", "user", "user"]
    assert "A reply without a tool call is not acted on." in llm.requests[-1][-1]["content"]


def test_model_prose_is_logged_without_markdown_markers(checkout: RepoCheckout):
    said = "## Done\nAdded a new **spec.md** file; `__init__.py` and 2 ** 3 are left alone."
    llm = ScriptedLLM([said, [("finish", {"summary": "done"})]])
    agent, events = make_agent(checkout, llm)
    run(agent)
    assert ("agent", "Done\nAdded a new spec.md file; `__init__.py` and 2 ** 3 are left alone.") in events
    assert plain_text("**a** and **b c**, not ** this **") == "a and b c, not ** this **"


def test_rate_limit_pause_names_the_limit(checkout: RepoCheckout):
    message = (
        "groq returned HTTP 429: Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service "
        "tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7421, Requested 1630. Please try again in 3.5s."
    )
    llm = ScriptedLLM([LLMError(message, kind="rate_limit", status_code=429, retry_after=3.5), [("finish", {"summary": "ok"})]])
    agent, events = make_agent(checkout, llm)
    assert run(agent).finished
    assert events[-1] == ("info", "Waiting 4s for scripted's rate limit (8,000 tokens per minute)")

    # Without details in the message, the limit the provider reported earlier is used.
    llm = ScriptedLLM(
        [
            [("list_files", {})],
            LLMError("groq returned HTTP 429: slow down", kind="rate_limit", status_code=429, retry_after=1.0),
            [("finish", {"summary": "ok"})],
        ],
        tokens_per_minute=6000,
    )
    agent, events = make_agent(checkout, llm)
    assert run(agent).finished
    assert events[-1] == ("info", "Waiting 2s for scripted's rate limit (6,000 tokens per minute)")


def test_rate_limits_are_waited_out(checkout: RepoCheckout):
    waits: List[float] = []
    llm = ScriptedLLM(
        [
            LLMError("groq returned HTTP 429: slow down", kind="rate_limit", status_code=429, retry_after=12.0),
            LLMError("groq returned HTTP 503: busy", kind="server", status_code=503),
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm, sleep=waits.append)
    assert run(agent).finished
    assert waits == [12.5, 1.5]
    # A pause for the provider's rate limit is normal operation, not a warning.
    assert ("info", "Waiting 12s for scripted's rate limit") in events
    assert [message for level, message in events if level == "warn"] == [
        "groq returned HTTP 503: busy - retrying (1/2)"
    ]


def test_rate_limit_longer_than_the_budget_fails_honestly(checkout: RepoCheckout):
    llm = ScriptedLLM([LLMError("groq returned HTTP 429: slow down", kind="rate_limit", retry_after=600.0)])
    agent, _events = make_agent(checkout, llm, deadline=Deadline(80), reserve_seconds=25)
    outcome = run(agent)
    assert outcome.stop_reason == "llm_error"
    assert "rate limit" in (outcome.error or "").lower() and "time budget" in (outcome.error or "")


def test_auth_errors_are_not_retried(checkout: RepoCheckout):
    llm = ScriptedLLM([LLMError("groq returned HTTP 401: Invalid API Key", kind="auth", status_code=401)])
    agent, _events = make_agent(checkout, llm)
    outcome = run(agent)
    assert outcome.stop_reason == "llm_error" and "Invalid API Key" in (outcome.error or "")
    assert len(llm.requests) == 1


def test_budgets_follow_the_providers_token_limit(checkout: RepoCheckout):
    """Groq's free tier allows 8k tokens/minute; a paid tier allows far more."""
    free_tier = ScriptedLLM([[("list_files", {})], [("finish", {"summary": "done"})]], tokens_per_minute=8000)
    agent, _events = make_agent(checkout, free_tier, max_output_tokens=8192)
    run(agent)
    assert free_tier.output_token_limits == [4096, 2400]  # conservative first, then 30% of the limit
    assert agent.context_budget == 14_400  # ~3.6k tokens of history
    # prompt + tools + reply must fit in one minute's allowance
    tools_tokens = len(json.dumps(AGENT_TOOLS)) / 4
    assert agent.context_budget / 4 + tools_tokens + agent.max_output_tokens < 8000

    paid = ScriptedLLM([[("list_files", {})], [("finish", {"summary": "done"})]], tokens_per_minute=250_000)
    agent, _events = make_agent(checkout, paid, max_output_tokens=8192)
    run(agent)
    assert paid.output_token_limits == [4096, 8192] and agent.context_budget == 160_000


def test_request_too_large_shrinks_context_and_retries(checkout: RepoCheckout):
    (Path(checkout.directory) / "big.txt").write_text("x" * 60 + "\n" + ("data " * 12 + "\n") * 300)
    too_large = LLMError(
        "groq returned HTTP 413: Request too large on tokens per minute (TPM): Limit 8000, Requested 9920",
        kind="too_large",
        status_code=413,
    )
    llm = ScriptedLLM(
        [
            [("read_file", {"path": "big.txt"})],
            [("read_file", {"path": "big.txt", "start_line": 100})],
            [("read_file", {"path": "big.txt", "start_line": 200})],
            too_large,
            [("finish", {"summary": "done"})],
        ]
    )
    agent, events = make_agent(checkout, llm)
    before = agent.context_budget
    assert run(agent).finished
    assert agent.context_budget < before
    assert any(level == "warn" and "too large" in message for level, message in events)
    # The retried request really is smaller than the one that was rejected.
    assert history_size(llm.requests[-1]) < history_size(llm.requests[-2])


def test_compact_history_keeps_task_and_recent_messages():
    big = "A" * 5000
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "the task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "a.py", "content": big})}}]},
        {"role": "tool", "tool_call_id": "1", "content": big},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}}]},
        {"role": "tool", "tool_call_id": "2", "content": big},
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "recent user"},
    ]
    assert compact_history(messages, budget=100_000) is False
    assert compact_history(messages, budget=7000, keep_recent=4) is True
    assert messages[0]["content"] == "system" and messages[1]["content"] == "the task"
    assert len(messages[3]["content"]) < 300 and "dropped to save context" in messages[3]["content"]
    shortened = json.loads(messages[2]["tool_calls"][0]["function"]["arguments"])
    assert shortened["path"] == "a.py" and "characters omitted" in shortened["content"]
    assert messages[5]["content"] == big  # inside the protected recent window
    assert history_size(messages) <= 7000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_redactor_masks_credentials():
    secret = "super-secret-value-123"
    redact = Redactor([secret, "short"])
    text = (
        f"token={secret} ghp_{'a' * 36} github_pat_{'b' * 30} "
        "https://x-access-token:abc123@github.com/o/r.git Authorization: Bearer abcdefgh12345678 short"
    )
    cleaned = redact(text)
    assert secret not in cleaned and "ghp_" not in cleaned and "github_pat_" not in cleaned
    assert "abc123" not in cleaned and "abcdefgh12345678" not in cleaned
    assert "x-access-token:***@github.com" in cleaned
    assert "short" in cleaned  # too short to be treated as a secret


def test_clip_keeps_head_and_tail():
    text = "".join(str(n % 10) for n in range(1000))
    clipped = clip(text, 100)
    assert clipped.startswith(text[:30]) and clipped.endswith(text[-70:]) and "omitted" in clipped
    assert clip("short", 100) == "short"
