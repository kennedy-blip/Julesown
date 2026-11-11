"""Runs one task end to end.

    configuration check -> GitHub access -> sandbox -> clone -> agent
        -> collect changes -> commit (GitHub API) -> pull request

Every step either really happens or the task fails with the reason. A task is
only ever reported as having a pull request when GitHub returned the pull
request that was just created.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..config import Settings
from .agent import AgentOutcome, CodingAgent
from .common import Deadline, Redactor, clip, first_line
from .github import GitHubClient, GitHubError, PullRequest, compare_url
from .llm import ChatClient, LLMError, create_llm_client
from .sandbox import ChangeSet, E2BWorkspace, RepoCheckout, SandboxError, Workspace
from .store import TaskStore, branch_for_task

logger = logging.getLogger("jules.runner")

# Time kept back from the agent so the commit and pull request can still be made.
COMMIT_RESERVE_SECONDS = 25.0
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


def _verification_result(run: Any) -> str:
    if run.timed_out:
        return "timed out"
    return "passed" if run.exit_code == 0 else f"failed (exit {run.exit_code})"


def build_pr_body(
    *,
    task: Dict[str, Any],
    outcome: AgentOutcome,
    changeset: ChangeSet,
    base_branch: str,
    base_sha: str,
    llm_label: str,
) -> str:
    lines: List[str] = ["## Summary", "", outcome.summary.strip() or "_The agent did not write a summary._", ""]
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

    lines += ["", "## Verification", ""]
    if outcome.verification:
        lines += ["| Command | Result |", "| --- | --- |"]
        for run in outcome.verification[-10:]:
            command = run.command.replace("|", "\\|").replace("`", "'").replace("\n", " ")
            icon = "✅" if run.passed else "❌"
            lines.append(f"| `{clip(command, 160, tail_share=0.0)}` | {icon} {_verification_result(run)} |")
        last = outcome.verification[-1]
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


def build_commit_message(title: str, outcome: AgentOutcome, task_id: str) -> str:
    parts = [title]
    summary = outcome.summary.strip()
    if summary and summary != title:
        parts += ["", clip(summary, 2000, tail_share=0.0)]
    parts += ["", f"Task-Id: {task_id}"]
    return "\n".join(parts)


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
        redact = Redactor(
            [
                settings.github_token or "",
                settings.e2b_api_key or "",
                settings.groq_api_key or "",
                settings.openai_api_key or "",
                settings.anthropic_api_key or "",
                settings.access_key or "",
                settings.github_webhook_secret or "",
            ]
        )

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
            base_branch = task.get("branch") or repo.default_branch
            if github.get_branch_sha(slug, base_branch) is None:
                raise TaskFailed(
                    f"Branch '{base_branch}' does not exist in {slug}. "
                    f"The default branch is '{repo.default_branch}'."
                )
            head_branch = branch_for_task(task_id)
            update(repository=slug, branch=base_branch, base_branch=base_branch, head_branch=head_branch)

            # 3. Sandbox ------------------------------------------------------
            stage("sandbox", "Starting sandbox")
            workspace = self.deps.workspace_factory(settings, task_id, int(budget) + 60)
            update(sandbox=workspace.description)
            emit("info", f"Sandbox ready: {workspace.description}")

            # 4. Clone --------------------------------------------------------
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
            emit("info", f"Cloned {slug}@{base_branch} at {checkout.base_sha[:10]} ({total_files} files)")

            # 5. Agent --------------------------------------------------------
            llm = self.deps.llm_factory(settings, lambda notice: emit("warn", notice))
            update(llm_provider=llm.provider, llm_model=llm.model)
            stage("agent", f"Agent working with {llm.provider} / {llm.model}")
            agent = CodingAgent(
                llm=llm,
                checkout=checkout,
                emit=emit,
                deadline=deadline,
                max_turns=settings.max_agent_turns,
                command_timeout=float(settings.command_timeout_seconds),
                max_output_tokens=settings.max_output_tokens,
                reserve_seconds=COMMIT_RESERVE_SECONDS,
            )
            outcome = agent.run(
                prompt=task["prompt"],
                repository=slug,
                branch=base_branch,
                file_listing=format_file_listing(files, total_files),
            )
            tests_status = outcome.tests_status
            update(
                llm_model=llm.model,
                summary=redact(outcome.summary) or None,
                tests_status=tests_status,
                tests_passed={"passed": True, "failed": False}.get(tests_status),
                verification=[
                    {
                        "command": redact(run.command),
                        "exit_code": run.exit_code,
                        "passed": run.passed,
                        "timed_out": run.timed_out,
                        "duration": round(run.duration, 1),
                    }
                    for run in outcome.verification
                ],
            )
            if not outcome.finished:
                self._note_discarded_work(checkout, emit)
                reason = outcome.error or "The agent stopped before finishing."
                if outcome.stop_reason == "gave_up":
                    reason = f"The agent reported that it could not complete the task: {reason}"
                raise TaskFailed(reason)
            emit("info", f"Agent finished after {outcome.turns} model turns")

            # 6. Changes ------------------------------------------------------
            stage("commit", "Collecting changes")
            changeset = checkout.collect_changes()
            if changeset.empty:
                update(
                    status="completed",
                    stage="done",
                    outcome="no_changes",
                    completed_at=time.time(),
                )
                emit(
                    "success",
                    "Finished without changing any files, so no pull request was opened.",
                )
                return store.get(task_id) or task
            update(changed_files=[change.summary() for change in changeset.changes])
            emit(
                "info",
                f"{len(changeset.changes)} file(s) changed: "
                + ", ".join(change.path for change in changeset.changes[:8])
                + (" ..." if len(changeset.changes) > 8 else ""),
            )

            # 7. Commit + pull request ---------------------------------------
            title = outcome.pr_title or first_line(outcome.summary) or first_line(task["prompt"])
            title = redact(title) or "Changes by Jules"
            commit = github.create_commit(
                slug,
                parent_sha=checkout.base_sha,
                base_tree_sha=checkout.base_tree_sha,
                changes=changeset.changes,
                message=redact(build_commit_message(title, outcome, task_id)),
                expected_tree_sha=changeset.tree_sha,
            )
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
                            llm_label=f"{llm.provider}/{llm.model}",
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
            if tests_status == "failed":
                emit("warn", "The last test run failed - the pull request says so. Review before merging.")
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
    def _log_outcome(record: Dict[str, Any], elapsed: float) -> None:
        """One line per task in the server log (on Vercel: the function log), so a
        run can be diagnosed there even when nobody was watching the dashboard.
        The error text is the already redacted one from the task record."""
        try:
            completed = record.get("status") == "completed"
            if completed:
                result = record.get("pr_url") or "no changes, no pull request"
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
