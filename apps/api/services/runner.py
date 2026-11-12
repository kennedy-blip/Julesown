"""Runs one task end to end.

    configuration check -> GitHub access -> sandbox -> clone -> agent
        -> collect changes -> commit (GitHub API) -> pull request

Every step either really happens or the task fails with the reason. A task is
only ever reported as having a pull request when GitHub returned the pull
request that was just created.

A repository without commits has no branch a pull request could be opened
against. There the agent builds the project from scratch and the result becomes
the repository's first commit on its default branch (outcome
``initial_commit``). That is the only case in which a default branch is written
to; in a repository that has commits the base branch is never touched.

A task can also be more work on a pull request that already exists (fixing its
failed checks, finishing what an earlier run left unfinished, or following
instructions): then the agent starts from that pull request's branch and its
commit is added to it (outcome ``pr_updated``). The branch only ever moves
forward; if somebody pushed to it in the meantime, nothing is pushed over that.

When the agent stops before the task is done - the clock, its model provider,
the turn limit, or its own verdict that it cannot complete the task - after it
has changed files, that work is delivered all the same, marked as unfinished.
It is not thrown away. The task record then carries a ``handoff``: where the
work stands, for the follow-up task that finishes it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..config import Settings
from .agent import AgentOutcome, CodingAgent
from .ci import (
    CONTINUE_NOTE,
    GAVE_UP,
    UNFINISHED_PREFIX,
    build_fix_prompt,
    build_follow_up_prompt,
    build_handoff,
    continue_note,
    describe_failed_checks,
    failure_note,
    handoff_from_commit,
    names_of,
    standing_instructions,
)
from .common import Deadline, Redactor, clip, first_line, listed
from .github import GitHubClient, GitHubError, PullRequest, compare_url, noreply_address
from .llm import ChatClient, LLMError, create_llm_client
from .sandbox import ChangeSet, E2BWorkspace, RepoCheckout, SandboxError, Workspace
from .store import TaskStore, branch_for_task
from .verification import VerificationRun, has_no_code, is_docs_only

logger = logging.getLogger("jules.runner")

# Time kept back from the agent so the commit and pull request can still be made.
COMMIT_RESERVE_SECONDS = 25.0
# Why the agent stopped without finishing. In each of these cases what it had
# changed is kept: interrupted work can be continued, and work the agent gave
# up on is for its owner to judge - with the agent's explanation next to it.
SALVAGEABLE_STOPS = ("timeout", "llm_error", "max_turns", "gave_up")
# Below this much time left nothing is pushed any more: a half-made push is
# worse than none.
MIN_SECONDS_TO_DELIVER = 8.0
FILE_LISTING_CHARS = 2000
STATUS_LABELS = {"A": "added", "M": "modified", "D": "deleted", "T": "type changed"}

EventSink = Callable[[Dict[str, Any]], None]


class TaskFailed(Exception):
    """The task cannot continue. The message is shown to the user."""


@dataclass
class RunnerDeps:
    """Factories for the external systems, replaceable in tests."""

    github_factory: Callable[[Settings], GitHubClient]
    workspace_factory: Callable[[Settings, str, int], Workspace]
    llm_factory: Callable[[Settings, Callable[[str], None]], ChatClient]
    clone_url_for: Callable[[str], str]


def github_client_from(settings: Settings) -> GitHubClient:
    return GitHubClient(
        token=settings.github_token,
        app_id=settings.github_app_id,
        app_private_key=settings.github_app_private_key,
        api_url=settings.github_api_url,
    )


def _e2b_workspace(settings: Settings, task_id: str, lifetime_seconds: int) -> Workspace:
    return E2BWorkspace.create(
        api_key=settings.e2b_api_key or "",
        template=settings.e2b_template,
        lifetime_seconds=lifetime_seconds,
        metadata={"app": "jules-clone", "task": task_id},
    )


def default_deps() -> RunnerDeps:
    return RunnerDeps(
        github_factory=github_client_from,
        workspace_factory=_e2b_workspace,
        llm_factory=lambda settings, notice: create_llm_client(settings, on_notice=notice),
        clone_url_for=lambda slug: f"https://github.com/{slug}.git",
    )


# --------------------------------------------------------------------------- #
# Text builders
# --------------------------------------------------------------------------- #
def format_file_listing(files: List[str], total: int) -> str:
    if not files:
        return ""
    shown: List[str] = []
    used = 0
    for path in files:
        if used + len(path) + 1 > FILE_LISTING_CHARS:
            break
        shown.append(path)
        used += len(path) + 1
    note = f", first {len(shown)} shown" if len(shown) < total else ""
    return f"Repository files ({total} tracked{note}):\n" + "\n".join(shown)


def _verification_result(run: VerificationRun) -> str:
    if run.timed_out:
        return "⏱️ timed out"
    if run.no_tests:
        return "➖ no tests to run"
    return "✅ passed" if run.exit_code == 0 else f"❌ failed (exit {run.exit_code})"


def _code(text: str) -> str:
    return "`" + text.replace("`", "'").replace("\n", " ") + "`"


def _left_out_item(item: str) -> str:
    """``path (reason)`` from ``ChangeSet.left_out``, with the path as code."""
    path, _, reason = item.rpartition(" (")
    return f"{_code(path)} ({reason}" if path else _code(item)


def build_pr_body(
    *,
    task: Dict[str, Any],
    outcome: AgentOutcome,
    changeset: ChangeSet,
    base_branch: str,
    base_sha: str,
    llm_label: str,
    new_project: bool = False,
    unfinished: Optional[str] = None,
    continue_by_comment: bool = False,
) -> str:
    docs_only = is_docs_only(change.path for change in changeset.changes)
    lines: List[str] = []
    if unfinished:
        # Where GitHub reaches this server, a comment is all it takes.
        finish = (
            "Comment `/jules continue` on this pull request to have it finished"
            if continue_by_comment
            else "A follow-up task can finish it"
        )
        lines += [
            "> [!WARNING]",
            f"> **Unfinished.** {_unfinished_sentence(unfinished)}",
            "> What it had changed until then is in this pull request so that the work is not lost. "
            f"It may not build or pass its tests. {finish}; do not merge it as it is.",
            "",
        ]
    lines += ["## Summary", "", outcome.summary.strip() or "_The agent did not write a summary._", ""]
    lines += ["## Task", ""]
    lines += [f"> {line}" for line in task["prompt"].strip().splitlines()[:40]]
    lines += ["", "## Changes", ""]
    for change in changeset.changes[:60]:
        label = STATUS_LABELS.get(change.status, change.status)
        counts = ""
        if change.additions is not None and change.deletions is not None:
            counts = f", +{change.additions} / -{change.deletions}"
        lines.append(f"- `{change.path}` ({label}{counts})")
    if len(changeset.changes) > 60:
        lines.append(f"- ... and {len(changeset.changes) - 60} more files")
    if changeset.left_out:
        # The work tree held more than this pull request does: say what and why.
        lines += [
            "",
            "Not included, because it is build output or too large to commit: "
            + listed([_left_out_item(item) for item in changeset.left_out], 12),
        ]

    # What the change is based on: a reviewer can see at a glance whether the
    # agent looked at the code it describes or changes.
    lines.append("")
    if new_project:
        lines.append("The repository had no code yet: this pull request creates the project from scratch.")
    elif outcome.files_read:
        shown = ", ".join(_code(path) for path in outcome.files_read[:20])
        more = len(outcome.files_read) - 20
        lines.append(f"Files the agent looked at: {shown}" + (f" and {more} more" if more > 0 else ""))
    else:
        lines.append(
            "⚠️ The agent did not open any file of the repository before making this change. "
            "Check what it wrote against the code."
        )

    lines += ["", "## Verification", ""]
    runs = outcome.verification
    tested = [run for run in runs if not run.no_tests]
    if runs:
        lines += ["| Command | Result |", "| --- | --- |"]
        for run in runs[-10:]:
            command = run.command.replace("|", "\\|").replace("`", "'").replace("\n", " ")
            lines.append(f"| `{clip(command, 160, tail_share=0.0)}` | {_verification_result(run)} |")
    if tested:
        last = tested[-1]
        if last.output_tail.strip():
            output = last.output_tail.replace("```", "'''")
            lines += [
                "",
                "<details><summary>Output of the last run</summary>",
                "",
                "```text",
                clip(output, 2500),
                "```",
                "",
                "</details>",
            ]
        if not last.passed:
            lines += ["", "> ⚠️ The last verification run did not pass. Review before merging."]
    elif runs:
        # Test commands were run, but there was nothing for them to test.
        note = f"This repository has no automated tests to run ({runs[-1].no_tests}), so the change is untested."
        lines += ["", note + (" It only touches documentation." if docs_only else " Review before merging.")]
    elif docs_only:
        lines.append("No tests were run: this pull request only changes documentation.")
    else:
        lines.append("⚠️ The agent did not run any tests for this change. Review before merging.")

    lines += ["", "---"]
    footer = (
        f"Opened by Jules Clone · task `{task['task_id']}` · base `{base_branch}` @ "
        f"`{base_sha[:10]}` · model `{llm_label}`"
    )
    lines.append(footer)
    issue_number = task.get("issue_number")
    if issue_number:
        lines += ["", f"Refs #{issue_number}"]
    return clip("\n".join(lines), 60_000, tail_share=0.1)


def build_commit_message(
    title: str,
    outcome: AgentOutcome,
    task_id: str,
    trailers: Optional[List[str]] = None,
    *,
    unfinished: Optional[str] = None,
) -> str:
    parts = [title]
    summary = outcome.summary.strip()
    if summary and summary != title:
        parts += ["", clip(summary, 2000, tail_share=0.0)]
    if unfinished:
        lead = "" if _gave_up(unfinished) else "the agent was stopped before it was done. "
        parts += ["", "Unfinished: " + lead + clip(unfinished, 500, tail_share=0.0)]
        # What still has to be fixed: the run that continues this reads it here.
        still_failing = failure_note(outcome.last_failure)
        if still_failing:
            parts += ["", still_failing]
    # Trailers form the last paragraph: that is where git and GitHub look for
    # them ("Co-authored-by" is how GitHub credits a second author).
    parts += ["", f"Task-Id: {task_id}"] + list(trailers or [])
    return "\n".join(parts)


def _gave_up(unfinished: str) -> bool:
    """The agent stopped by itself (as opposed to being interrupted)."""
    return unfinished.startswith(GAVE_UP)


def _unfinished_sentence(unfinished: str) -> str:
    """Why the work is unfinished, as a sentence for a pull request."""
    text = unfinished.strip()
    return text if _gave_up(text) else f"The agent was stopped before it was done: {text}"


def resolve_commit_identity(settings: Settings, task: Dict[str, Any]) -> Dict[str, Any]:
    """Who the commits of this task are attributed to.

    Returns ``author`` (for the commit request, or None to leave it to GitHub),
    ``trailers`` (lines for the commit message), ``mode`` as it is applied and
    a ``note`` for the log. A person can only be named when a name and an
    e-mail address are known: from the settings, or - for a task started from
    GitHub - the account that asked for it, with its ``noreply`` address.
    """
    identity = settings.commit_identity
    mode, name, email = identity["mode"], identity["name"], identity["email"]
    if mode != "bot" and not (name and email):
        requester = task.get("requested_by") if isinstance(task.get("requested_by"), dict) else {}
        login = str(requester.get("login") or "")
        address = noreply_address(login, requester.get("id"))
        if login and address:
            name, email = login, address
    if mode == "bot":
        return {"author": None, "trailers": [], "mode": "bot", "note": ""}
    if not (name and email):
        return {
            "author": None,
            "trailers": [],
            "mode": "bot",
            "note": "Commits are set to carry a person's name, but no name and e-mail address are "
            "configured for it: committing as the GitHub credentials instead",
        }
    if mode == "user":
        return {
            "author": {"name": name, "email": email},
            "trailers": [],
            "mode": "user",
            "note": f"The commit is authored as {name} <{email}>",
        }
    return {
        "author": None,
        "trailers": [f"Co-authored-by: {name} <{email}>"],
        "mode": "coauthor",
        "note": f"The commit names {name} <{email}> as co-author",
    }


def models_used(llm: ChatClient, served: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Which models answered during the run and how often.

    A pool of models keeps the count itself. A single client does not: for it
    ``served`` - the models the agent saw answer under a router's or an alias's
    name - is all there is to add.
    """
    usage = getattr(llm, "usage", None)
    if callable(usage):
        return [item for item in usage() if item.get("requests") or item.get("failures")]
    record: Dict[str, Any] = {"provider": llm.provider, "model": llm.model}
    if served:
        record["served"] = dict(served)
    return [record]


def build_follow_up_comment(
    *,
    task: Dict[str, Any],
    outcome: AgentOutcome,
    commit_sha: str,
    commit_url: Optional[str],
    changeset: ChangeSet,
    unfinished: Optional[str],
    fixed_checks: Optional[str],
    continue_by_comment: bool = False,
) -> str:
    """The comment that tells a pull request what was just pushed to it."""
    where = f"[`{commit_sha[:10]}`]({commit_url})" if commit_url else f"`{commit_sha[:10]}`"
    if unfinished and _gave_up(unfinished):
        head = f"⚠️ I pushed {where}, but I could not complete the task" + unfinished.strip()[len(GAVE_UP) :]
    elif unfinished:
        head = f"⚠️ I pushed {where}, but I was stopped before I was done: {unfinished.strip()}"
    elif fixed_checks:
        head = f"🔧 I pushed {where} to fix the failed checks ({fixed_checks})."
    else:
        head = f"✅ I pushed {where}."
    lines = [head]
    summary = outcome.summary.strip()
    if summary:
        lines += ["", summary]
    files = ", ".join(f"`{change.path}`" for change in changeset.changes[:12])
    more = len(changeset.changes) - 12
    lines += ["", f"Changed: {files}" + (f" and {more} more" if more > 0 else "")]
    tested = [run for run in outcome.verification if not run.no_tests]
    if tested:
        last = tested[-1]
        lines += ["", f"Last test run in the sandbox: `{clip(last.command, 120, tail_share=0.0)}` - {_verification_result(last)}"]
    if unfinished and continue_by_comment:
        lines += ["", "Comment `/jules continue` to have it finished."]
    lines += ["", f"<sub>Jules Clone · task `{task['task_id']}`</sub>"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class TaskRunner:
    def __init__(
        self, settings: Settings, store: TaskStore, deps: Optional[RunnerDeps] = None
    ) -> None:
        self.settings = settings
        self.store = store
        self.deps = deps or default_deps()

    def run(
        self,
        task_id: str,
        *,
        budget_seconds: Optional[float] = None,
        on_event: Optional[EventSink] = None,
    ) -> Dict[str, Any]:
        """Execute the task and return its final record. Never raises."""
        settings, store = self.settings, self.store
        task = store.get(task_id)
        if task is None:
            raise KeyError(task_id)

        budget = float(budget_seconds or settings.task_budget_seconds)
        deadline = Deadline(budget)
        # Every key this run can use - the server's and the ones sent along with
        # the task - is kept out of the log and of error messages.
        redact = Redactor(settings.secret_values())

        def publish(event: Dict[str, Any]) -> None:
            if on_event is None:
                return
            try:
                on_event(event)
            except Exception:  # a broken listener must not abort the run
                logger.exception("task event listener failed")

        def emit(level: str, message: str) -> None:
            publish({"type": "log", "entry": store.log(task_id, level, redact(message))})

        def update(**fields: Any) -> None:
            store.update(task_id, **fields)
            publish({"type": "update", "fields": fields})

        def stage(name: str, message: str) -> None:
            update(stage=name)
            emit("stage", message)

        github: Optional[GitHubClient] = None
        workspace: Optional[Workspace] = None
        llm: Optional[ChatClient] = None
        checkout: Optional[RepoCheckout] = None

        try:
            update(status="running", stage="configuration", started_at=time.time())
            emit(
                "info",
                f"Task {task_id[:12]} started for {task['repository']} (time budget {int(budget)} s)",
            )

            # 1. Configuration ------------------------------------------------
            missing = settings.missing_requirements()
            if missing:
                raise TaskFailed("The server is not ready to run tasks. Missing: " + "; ".join(missing) + ".")

            # 2. GitHub access ------------------------------------------------
            stage("github", "Checking GitHub access")
            github = self.deps.github_factory(settings)
            repo = github.get_repository(task["repository"])
            access = github.access_for(task["repository"])
            redact.add(*github.secrets())
            slug = repo.slug
            emit(
                "info",
                f"GitHub access confirmed for {slug} via "
                + ("GitHub App installation" if access.mode == "app" else "GITHUB_TOKEN"),
            )
            if repo.archived:
                raise TaskFailed(f"{slug} is archived, so it cannot receive pull requests.")
            if repo.can_push is False:
                raise TaskFailed(
                    f"The configured GitHub token cannot push to {slug}. Grant it write access "
                    "(contents and pull requests) or install the GitHub App on the repository."
                )
            pull_number = task.get("pull_request")
            fix_checks = bool(task.get("fix_checks"))
            continuing = bool(task.get("continue_work"))
            # Where the run this task continues had got to, if it said (see build_handoff).
            previous_handoff = str(task.get("previous_handoff") or "") if continuing else ""
            # The pull request this task adds to, if it is more work on one that exists.
            target: Optional[PullRequest] = None
            empty_repository = False
            head_branch: Optional[str]
            if pull_number:
                target = github.get_pull_request(slug, int(pull_number))
                if target.merged or target.state != "open":
                    raise TaskFailed(
                        f"Pull request #{target.number} is {'merged' if target.merged else 'closed'}, "
                        "so nothing can be added to it."
                    )
                if target.head_repository and target.head_repository.lower() != slug.lower():
                    raise TaskFailed(
                        f"Pull request #{target.number} comes from a fork ({target.head_repository}). "
                        f"The agent can only push to branches of {slug} itself."
                    )
                base_branch = head_branch = target.head_branch
                if github.get_branch_sha(slug, head_branch) is None:
                    raise TaskFailed(
                        f"The branch of pull request #{target.number} ('{head_branch}') no longer exists."
                    )
                emit(
                    "info",
                    f"Working on pull request #{target.number} \"{target.title}\": the result is added "
                    f"to its branch {head_branch}",
                )
            else:
                base_branch = task.get("branch") or repo.default_branch
                try:
                    branch_exists = github.get_branch_sha(slug, base_branch) is not None
                except GitHubError as exc:
                    if exc.code != "repo_empty":
                        raise
                    branch_exists, empty_repository = False, True
                if empty_repository:
                    # No commits, so no branch: the result becomes the first commit
                    # on the default branch instead of a pull request.
                    if fix_checks:
                        raise TaskFailed(f"{slug} is empty: it has no commits and therefore no checks to fix.")
                    if base_branch != repo.default_branch:
                        raise TaskFailed(
                            f"{slug} is empty, so it has no branch '{base_branch}' to start from. Leave the "
                            "base branch empty: the first commit is made on the default branch "
                            f"'{repo.default_branch}'."
                        )
                    head_branch = None
                    emit(
                        "info",
                        f"{slug} is empty: the agent builds the project from scratch and the result "
                        f"becomes the first commit on {base_branch}",
                    )
                else:
                    if not branch_exists:
                        raise TaskFailed(
                            f"Branch '{base_branch}' does not exist in {slug}. "
                            f"The default branch is '{repo.default_branch}'."
                        )
                    head_branch = branch_for_task(task_id)
            update(
                repository=slug,
                repository_url=repo.html_url,
                branch=base_branch,
                base_branch=target.base_branch if target else base_branch,
                head_branch=head_branch,
                empty_repository=empty_repository,
                **({"pr_number": target.number, "pr_url": target.html_url} if target else {}),
            )

            # What CI says about the commit the work starts from. For a task that
            # is about the failed checks this is the task; for other work on a
            # pull request it is context.
            agent_prompt = task["prompt"]
            failed_names = ""
            if fix_checks or target is not None:
                checked_sha = (target.head_sha if target else "") or github.get_branch_sha(slug, base_branch) or ""
                checks = github.checks_for(slug, checked_sha)
                for note in checks["notes"]:
                    emit("warn", note[0].upper() + note[1:])
                checks_text: Optional[str] = None
                if checks["state"] == "failure":
                    failed_names = names_of(checks)
                    emit("info", f"Failed checks on {checked_sha[:10]}: {failed_names}")
                    checks_text = describe_failed_checks(github, slug, checks)
                    update(
                        failed_checks=[
                            {"name": check["name"], "kind": check["kind"], "url": check.get("url")}
                            for check in checks["failing"]
                        ]
                    )
                elif fix_checks:
                    if checks["state"] == "unknown":
                        raise TaskFailed(
                            f"The checks of {checked_sha[:10]} could not be read: "
                            + "; ".join(checks["notes"])
                            + ". Add those permissions to the GitHub App and approve them on its installation."
                        )
                    counted = (
                        "it has no checks"
                        if checks["state"] == "none"
                        else f"{checks['passed']} passed, {checks['pending']} still running"
                    )
                    update(status="completed", stage="done", outcome="no_changes", completed_at=time.time())
                    emit("success", f"Nothing to fix: no check has failed on {checked_sha[:10]} ({counted}).")
                    return store.get(task_id) or task
                if fix_checks and checks_text:
                    where = (
                        f"pull request #{target.number} \"{target.title}\" (branch `{head_branch}`)"
                        if target
                        else f"branch `{base_branch}`"
                    )
                    agent_prompt = build_fix_prompt(
                        target=where,
                        sha=checked_sha,
                        checks_text=checks_text,
                        instructions=str(task.get("instructions") or ""),
                    )
                elif target is not None:
                    agent_prompt = build_follow_up_prompt(
                        instructions=task["prompt"],
                        number=target.number,
                        title=target.title,
                        head=target.head_branch,
                        base=target.base_branch,
                        checks_text=checks_text,
                        continuing=continuing,
                        handoff=previous_handoff,
                    )
            elif continuing:
                agent_prompt = f"{task['prompt']}\n\n{continue_note(previous_handoff)}"
            if continuing and previous_handoff:
                emit("info", "Continuing from what the previous run reported when it stopped")

            # 3. Sandbox ------------------------------------------------------
            stage("sandbox", "Starting sandbox")
            workspace = self.deps.workspace_factory(settings, task_id, int(budget) + 60)
            update(sandbox=workspace.description)
            emit("info", f"Sandbox ready: {workspace.description}")

            # 4. Clone --------------------------------------------------------
            files: List[str] = []
            total_files = 0
            key_files: List[Any] = []
            if empty_repository:
                stage("clone", f"Setting up an empty work tree for {slug}")
                checkout = RepoCheckout.init_empty(
                    workspace, branch=base_branch, clone_url=self.deps.clone_url_for(slug)
                )
                emit("info", f"{slug} has no commits to clone: starting from an empty work tree on {base_branch}")
            else:
                stage("clone", f"Cloning {slug} ({base_branch})")
                clone_token = github.clone_token(slug) if repo.private else None
                redact.add(*github.secrets())
                checkout = RepoCheckout.clone(
                    workspace,
                    clone_url=self.deps.clone_url_for(slug),
                    branch=base_branch,
                    token=clone_token,
                    timeout=max(20.0, min(120.0, deadline.remaining() - COMMIT_RESERVE_SECONDS)),
                )
                files, total_files = checkout.list_files(limit=200)
                key_files = checkout.key_files()
                emit("info", f"Cloned {slug}@{base_branch} at {checkout.base_sha[:10]} ({total_files} files)")
                if continuing and not previous_handoff and CONTINUE_NOTE in agent_prompt:
                    # Nobody passed on where the earlier run had got to (the task was
                    # started on GitHub, or from another browser): its commit says.
                    previous_handoff = redact(handoff_from_commit(checkout.head_message()))
                    if previous_handoff:
                        agent_prompt = agent_prompt.replace(CONTINUE_NOTE, continue_note(previous_handoff), 1)
                        emit("info", "Continuing from what the commit of the previous run says about where it stopped")
            # Nothing but a README, a licence or a .gitignore is not a project yet
            # either: the task is to build one, not to change one.
            new_project = empty_repository or has_no_code(files, total_files)
            update(new_project=new_project)
            if new_project and not empty_repository:
                emit(
                    "info",
                    f"{slug} has no code yet (only {', '.join(files) or 'no files'}): "
                    "the agent builds the project from scratch",
                )

            # The owner's own rules for every task, after the task itself.
            guidelines = standing_instructions(settings.task_guidelines, str(task.get("guidelines") or ""))
            if guidelines:
                agent_prompt = f"{agent_prompt}\n\n{redact(guidelines)}"
                emit("info", "Standing instructions are added to the task")

            # 5. Agent --------------------------------------------------------
            llm = self.deps.llm_factory(settings, lambda notice: emit("warn", notice))
            chain = list(getattr(llm, "chain", None) or [f"{llm.provider} / {llm.model}"])
            update(llm_provider=llm.provider, llm_model=llm.model, llm_chain=chain)
            stage("agent", f"Agent working with {llm.provider} / {llm.model}")
            if len(chain) > 1:
                emit("info", "If that provider reaches a limit or fails, the agent carries on with: " + ", ".join(chain[1:]))
            agent = CodingAgent(
                llm=llm,
                checkout=checkout,
                emit=emit,
                deadline=deadline,
                max_turns=settings.max_agent_turns,
                command_timeout=float(settings.command_timeout_seconds),
                max_output_tokens=settings.max_output_tokens,
                reserve_seconds=COMMIT_RESERVE_SECONDS,
                new_project=new_project,
                update=target is not None,
            )
            outcome = agent.run(
                prompt=agent_prompt,
                repository=slug,
                branch=base_branch,
                file_listing=format_file_listing(files, total_files),
                key_files=key_files,
            )
            tests_status = outcome.tests_status
            used = models_used(llm, outcome.served)
            answered = [f"{item['provider']}/{item['model']}" for item in used if item.get("requests", 1)]
            update(
                llm_provider=llm.provider,
                llm_model=llm.model,
                llm_usage=used,
                summary=redact(outcome.summary) or None,
                tests_status=tests_status,
                tests_passed={"passed": True, "failed": False}.get(tests_status),
                verification=[run.as_record(redact) for run in outcome.verification],
                files_read=[redact(path) for path in outcome.files_read],
            )
            # Set when the agent was stopped but what it changed is delivered anyway.
            unfinished: Optional[str] = None
            changeset: Optional[ChangeSet] = None
            if not outcome.finished:
                reason = outcome.error or "The agent stopped before finishing."
                if outcome.stop_reason == "gave_up":
                    reason = f"{GAVE_UP}: {reason}"
                if (
                    settings.keep_unfinished_work
                    and outcome.stop_reason in SALVAGEABLE_STOPS
                    and deadline.remaining() >= MIN_SECONDS_TO_DELIVER
                ):
                    try:
                        # Nobody can tidy up any more: build output and files too
                        # large to commit are left out rather than costing the rest.
                        changeset = checkout.collect_changes(salvage=True, written=outcome.files_written)
                    except SandboxError as exc:
                        emit("warn", f"The unfinished work cannot be kept: {exc}")
                        changeset = None
                if changeset is None or changeset.empty:
                    if changeset is not None and changeset.left_out:
                        emit(
                            "warn",
                            "Nothing of the unfinished work can be kept. All of it is build output or too large "
                            "to commit: " + listed(changeset.left_out),
                        )
                    self._note_discarded_work(checkout, emit)
                    raise TaskFailed(reason)
                unfinished = redact(reason)
                update(
                    unfinished=True,
                    stop_reason=outcome.stop_reason,
                    unfinished_reason=unfinished,
                    handoff=redact(
                        build_handoff(
                            summary=outcome.summary,
                            reason=reason,
                            last_failure=outcome.last_failure,
                            left_out=changeset.left_out,
                        )
                    ),
                )
                if changeset.left_out:
                    emit(
                        "warn",
                        "Not kept, because it is no part of the work or cannot be committed: "
                        + listed(changeset.left_out),
                    )
                emit(
                    "warn",
                    (
                        f"{unfinished.rstrip('.')}. "
                        if outcome.stop_reason == "gave_up"
                        else f"The agent was stopped before it was done ({unfinished.rstrip('.')}). "
                    )
                    + "What it changed until then is kept and delivered marked as unfinished.",
                )
            else:
                emit("info", f"Agent finished after {outcome.turns} model turns")

            # 6. Changes ------------------------------------------------------
            stage("commit", "Collecting changes")
            if changeset is None:
                try:
                    changeset = checkout.collect_changes()
                except SandboxError as exc:
                    # The agent has finished and cannot tidy up any more. What
                    # cannot be committed (a bundle, a dump) is left out and
                    # named, rather than costing the whole result.
                    changeset = checkout.collect_changes(salvage=True, written=outcome.files_written)
                    if changeset.empty or not changeset.left_out:
                        raise exc  # there is no rest to deliver: the task fails for the reason it gives
                    emit(
                        "warn",
                        f"{str(exc).rstrip('.')}. Left out so that the rest can be delivered: "
                        + listed(changeset.left_out),
                    )
            if changeset.empty:
                update(
                    status="completed",
                    stage="done",
                    outcome="no_changes",
                    completed_at=time.time(),
                )
                emit(
                    "success",
                    "Finished without creating any files, so nothing was pushed."
                    if empty_repository
                    else f"Finished without changing any files, so nothing was added to pull request #{target.number}."
                    if target
                    else "Finished without changing any files, so no pull request was opened.",
                )
                if target is not None and continuing and target.title.startswith(UNFINISHED_PREFIX):
                    # Asked to finish the work, the agent found nothing left to
                    # do: the pull request is complete as it is.
                    self._clear_unfinished_mark(github, slug, target, redact(outcome.summary), task_id, emit)
                return store.get(task_id) or task
            docs_only = is_docs_only(change.path for change in changeset.changes)
            update(changed_files=[change.summary() for change in changeset.changes], docs_only=docs_only)
            emit(
                "info",
                f"{len(changeset.changes)} file(s) changed: "
                + ", ".join(change.path for change in changeset.changes[:8])
                + (" ..." if len(changeset.changes) > 8 else ""),
            )

            # 7. Commit + pull request ---------------------------------------
            if unfinished:
                # What the agent wrote when it stopped is an account of where the
                # work stands, not a title: without one, the task names the work.
                title = outcome.pr_title or first_line(task["prompt"]) or first_line(outcome.summary)
            else:
                title = outcome.pr_title or first_line(outcome.summary) or first_line(task["prompt"])
            title = redact(title) or "Changes by Jules"
            if unfinished:
                title = UNFINISHED_PREFIX + title
            identity = resolve_commit_identity(settings, task)
            # A commit with another author still names who made it: the App.
            committer = github.committer_identity(slug) if identity["author"] else None
            if identity["note"]:
                emit("info", identity["note"] + (f", committed by {committer['name']}" if committer else ""))
            update(commit_authorship=identity["mode"])
            message = redact(
                build_commit_message(title, outcome, task_id, identity["trailers"], unfinished=unfinished)
            )
            if empty_repository:
                self._push_first_commit(
                    github=github,
                    slug=slug,
                    branch=base_branch,
                    changeset=changeset,
                    message=message,
                    tests_status=tests_status,
                    docs_only=docs_only,
                    stage=stage,
                    update=update,
                    emit=emit,
                    author=identity["author"],
                    committer=committer,
                    unfinished=unfinished,
                )
                return store.get(task_id) or task
            assert head_branch is not None
            commit = github.create_commit(
                slug,
                parent_sha=checkout.base_sha,
                base_tree_sha=checkout.base_tree_sha,
                changes=changeset.changes,
                message=message,
                expected_tree_sha=changeset.tree_sha,
                author=identity["author"],
                committer=committer,
            )
            if target is not None:
                self._add_to_pull_request(
                    github=github,
                    slug=slug,
                    target=target,
                    commit=commit,
                    task=store.get(task_id) or task,
                    outcome=outcome,
                    changeset=changeset,
                    unfinished=unfinished,
                    fixed_checks=failed_names if fix_checks else None,
                    redact=redact,
                    stage=stage,
                    update=update,
                    emit=emit,
                    continue_by_comment=bool(settings.github_webhook_secret),
                )
                return store.get(task_id) or task
            github.create_branch(slug, head_branch, commit["sha"])
            update(commit_sha=commit["sha"])
            emit("info", f"Pushed commit {commit['sha'][:10]} to branch {head_branch}")

            stage("pull_request", "Opening pull request")
            try:
                pull_request = github.create_pull_request(
                    slug,
                    title=title,
                    head=head_branch,
                    base=base_branch,
                    body=redact(
                        build_pr_body(
                            task=store.get(task_id) or task,
                            outcome=outcome,
                            changeset=changeset,
                            base_branch=base_branch,
                            base_sha=checkout.base_sha,
                            llm_label=", ".join(answered) or f"{llm.provider}/{llm.model}",
                            new_project=new_project,
                            unfinished=unfinished,
                            continue_by_comment=bool(settings.github_webhook_secret),
                        )
                    ),
                )
            except GitHubError as exc:
                raise TaskFailed(
                    f"The changes were pushed to branch {head_branch}, but GitHub refused to open the "
                    f"pull request: {exc.describe()} You can open it yourself: "
                    f"{compare_url(slug, base_branch, head_branch)}"
                ) from exc

            update(
                status="completed",
                stage="done",
                outcome="pull_request",
                pr_url=pull_request.html_url,
                pr_number=pull_request.number,
                completed_at=time.time(),
            )
            emit("success", f"Pull request #{pull_request.number} opened: {pull_request.html_url}")
            if unfinished:
                emit(
                    "warn",
                    "The pull request is marked as unfinished. Start a follow-up on it to have the work "
                    "completed; do not merge it as it is.",
                )
            elif tests_status == "failed":
                emit("warn", "The last test run failed - the pull request says so. Review before merging.")
            elif docs_only:
                pass  # prose only: there is nothing a test run could have told us
            elif tests_status == "no_tests":
                emit(
                    "warn",
                    "This repository has no automated tests, so the change is untested - "
                    "the pull request says so. Review before merging.",
                )
            elif tests_status == "not_run":
                emit("warn", "The agent did not run tests - the pull request says so.")
        except TaskFailed as exc:
            self._fail(task_id, str(exc), update, emit, redact)
        except GitHubError as exc:
            self._fail(task_id, exc.describe(), update, emit, redact)
        except (SandboxError, LLMError) as exc:
            self._fail(task_id, str(exc), update, emit, redact)
        except Exception as exc:  # last line of defence: report, never fake success
            logger.exception("task %s crashed", task_id)
            self._fail(
                task_id,
                f"Unexpected error ({exc.__class__.__name__}): {exc}",
                update,
                emit,
                redact,
            )
        finally:
            for resource in (workspace, llm, github):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        logger.debug("closing %r failed", resource, exc_info=True)
            self._log_outcome(store.get(task_id) or task, deadline.elapsed())
        return store.get(task_id) or task

    @staticmethod
    def _push_first_commit(
        *,
        github: GitHubClient,
        slug: str,
        branch: str,
        changeset: ChangeSet,
        message: str,
        tests_status: str,
        docs_only: bool,
        stage: Callable[[str, str], None],
        update: Callable[..., None],
        emit: Callable[[str, str], None],
        author: Optional[Dict[str, str]] = None,
        committer: Optional[Dict[str, str]] = None,
        unfinished: Optional[str] = None,
    ) -> None:
        """Deliver the work to a repository that was empty: as its first commit."""
        stage("push", f"Pushing the first commit to {branch}")
        if not github.is_empty(slug, branch):
            # Somebody committed while the agent worked. Their commit is not overwritten.
            raise TaskFailed(
                f"{slug} received its first commit while the task was running, so the result was not "
                "pushed over it. Run the task again: it will now arrive as a pull request."
            )
        try:
            commit = github.create_initial_commit(
                slug,
                branch=branch,
                changes=changeset.changes,
                message=message,
                expected_tree_sha=changeset.tree_sha,
                author=author,
                committer=committer,
            )
        except GitHubError as exc:
            if exc.code == "initial_commit_incomplete":
                others = len(changeset.changes) - 1
                raise TaskFailed(
                    f"The first commit was created in {slug} with {exc.details.get('first_path')}, but "
                    f"the other {others} file{'s' if others != 1 else ''} could not be pushed: "
                    f"{exc.describe()} Run the task again - the repository is no longer empty, so the "
                    "result will arrive as a pull request."
                ) from exc
            if exc.code == "repo_not_empty":
                raise TaskFailed(
                    f"{exc.message} Run the task again: it will now arrive as a pull request."
                ) from exc
            raise
        commit_url = commit.get("html_url")
        update(
            status="completed",
            stage="done",
            outcome="initial_commit",
            commit_sha=commit["sha"],
            commit_url=commit_url if isinstance(commit_url, str) and commit_url.startswith("http") else None,
            branch=commit["branch"],
            base_branch=commit["branch"],
            completed_at=time.time(),
        )
        count = len(changeset.changes)
        emit(
            "success",
            f"First commit {commit['sha'][:10]} pushed to {commit['branch']} "
            f"({count} file{'s' if count != 1 else ''})" + (f": {commit_url}" if commit_url else ""),
        )
        if unfinished:
            emit(
                "warn",
                "The project is unfinished: what the agent had built when it stopped was pushed so "
                "that the work is not lost. It may not build yet. Run another task on this repository "
                "to have it completed - that one arrives as a pull request.",
            )
        elif tests_status == "failed":
            emit(
                "warn",
                "The last test run failed, and the files were pushed anyway: an empty repository has "
                "no pull request that could hold them back. Check the project before you rely on it.",
            )
        elif not docs_only and tests_status in ("no_tests", "not_run"):
            emit("info", "The new project has no automated tests. Try it before you deploy it.")

    @staticmethod
    def _add_to_pull_request(
        *,
        github: GitHubClient,
        slug: str,
        target: PullRequest,
        commit: Dict[str, Any],
        task: Dict[str, Any],
        outcome: AgentOutcome,
        changeset: ChangeSet,
        unfinished: Optional[str],
        fixed_checks: Optional[str],
        redact: Redactor,
        stage: Callable[[str, str], None],
        update: Callable[..., None],
        emit: Callable[[str, str], None],
        continue_by_comment: bool = False,
    ) -> None:
        """Deliver the work as a new commit on the branch of an existing pull request."""
        sha = commit["sha"]
        stage("pull_request", f"Adding the commit to pull request #{target.number}")
        try:
            # Never forced: GitHub refuses unless the commit descends from the branch's head.
            github.update_branch(slug, target.head_branch, sha)
        except GitHubError as exc:
            if exc.status_code == 422:
                raise TaskFailed(
                    f"The branch of pull request #{target.number} ({target.head_branch}) changed while "
                    "the agent was working, so its commit was not pushed over that. Run the task again."
                ) from exc
            raise
        commit_url = commit.get("html_url") or f"https://github.com/{slug}/commit/{sha}"
        update(commit_sha=sha, commit_url=commit_url)
        emit("info", f"Pushed commit {sha[:10]} to branch {target.head_branch}")

        # The title says whether the pull request is still unfinished.
        marked = target.title.startswith(UNFINISHED_PREFIX)
        new_title: Optional[str] = None
        if unfinished and not marked:
            new_title = UNFINISHED_PREFIX + target.title
        elif not unfinished and marked:
            new_title = target.title[len(UNFINISHED_PREFIX) :]
        try:
            if new_title:
                github.update_pull_request(slug, target.number, title=new_title)
                if not unfinished:
                    emit("info", f"Pull request #{target.number} is no longer marked as unfinished")
            github.create_issue_comment(
                slug,
                target.number,
                redact(
                    build_follow_up_comment(
                        task=task,
                        outcome=outcome,
                        commit_sha=sha,
                        commit_url=commit_url,
                        changeset=changeset,
                        unfinished=unfinished,
                        fixed_checks=fixed_checks,
                        continue_by_comment=continue_by_comment,
                    )
                ),
            )
        except GitHubError as exc:
            # The commit is on the branch; only the note about it is missing.
            emit("warn", f"The commit was pushed, but the pull request could not be annotated: {exc.describe()}")
        update(
            status="completed",
            stage="done",
            outcome="pr_updated",
            pr_url=target.html_url,
            pr_number=target.number,
            completed_at=time.time(),
        )
        emit("success", f"Pull request #{target.number} updated with commit {sha[:10]}: {target.html_url}")
        if unfinished:
            emit("warn", "The work is still unfinished. Start another follow-up on the pull request to have it completed.")
        elif fixed_checks:
            emit("info", "GitHub runs the checks again for the new commit; their result shows on the pull request.")

    @staticmethod
    def _clear_unfinished_mark(
        github: GitHubClient,
        slug: str,
        target: PullRequest,
        summary: str,
        task_id: str,
        emit: Callable[[str, str], None],
    ) -> None:
        """Take the "[Unfinished]" mark off a pull request that a follow-up task
        looked at and found complete."""
        try:
            github.update_pull_request(slug, target.number, title=target.title[len(UNFINISHED_PREFIX) :])
            lines = ["✅ I went over this pull request to finish it and found nothing left to do: it is no longer marked as unfinished."]
            if summary.strip():
                lines += ["", summary.strip()]
            lines += ["", f"<sub>Jules Clone · task `{task_id}`</sub>"]
            github.create_issue_comment(slug, target.number, "\n".join(lines))
        except GitHubError as exc:
            emit("warn", f"The pull request could not be updated: {exc.describe()}")
            return
        emit("info", f"Pull request #{target.number} is no longer marked as unfinished")

    @staticmethod
    def _log_outcome(record: Dict[str, Any], elapsed: float) -> None:
        """One line per task in the server log (on Vercel: the function log), so a
        run can be diagnosed there even when nobody was watching the dashboard.
        The error text is the already redacted one from the task record."""
        try:
            completed = record.get("status") == "completed"
            if completed:
                result = (
                    record.get("pr_url")
                    or (record.get("outcome") == "initial_commit" and f"first commit {record.get('commit_sha')}")
                    or "no changes, no pull request"
                )
                if record.get("unfinished"):
                    result = f"UNFINISHED ({record.get('stop_reason')}): {result}"
            else:
                result = f"FAILED at stage '{record.get('stage')}': {record.get('error')}"
            logger.log(
                logging.INFO if completed else logging.WARNING,
                "task %s on %s finished after %.0fs: %s",
                str(record.get("task_id", ""))[:12],
                record.get("repository"),
                elapsed,
                result,
            )
        except Exception:  # logging must never change the outcome of a run
            pass

    @staticmethod
    def _note_discarded_work(checkout: RepoCheckout, emit: Callable[[str, str], None]) -> None:
        try:
            stat = checkout.diff_stat()
        except Exception:
            return
        if stat:
            last = stat.splitlines()[-1].strip()
            emit("warn", f"Unfinished work in the sandbox was discarded ({last}).")

    @staticmethod
    def _fail(
        task_id: str,
        message: str,
        update: Callable[..., None],
        emit: Callable[[str, str], None],
        redact: Redactor,
    ) -> None:
        clean = redact(message)
        update(status="failed", error=clean, completed_at=time.time())
        emit("error", clean)


# --------------------------------------------------------------------------- #
# Recovering a task's outcome from GitHub
# --------------------------------------------------------------------------- #
def pull_request_for_task(
    github: GitHubClient, repository: str, task_id: str
) -> Optional[PullRequest]:
    """Find the pull request a task opened, using its deterministic branch name."""
    return github.find_pull_request(repository, branch_for_task(task_id))
