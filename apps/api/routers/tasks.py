"""Task endpoints.

``POST /stream`` is what the dashboard uses: the task runs inside that request
and its progress is streamed back as newline-delimited JSON. That keeps the
work alive on serverless platforms and needs no shared storage - the client
that started the task receives every log line and the final result directly.

``POST ""`` starts the same work in the background and returns immediately;
progress is then read with ``GET /{task_id}``. Task records are kept in
process memory, so on a serverless platform a poll can reach a different
instance. In that case the result can still be recovered from GitHub by
passing ``?repository=owner/name``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from ..config import Settings, get_settings, platform_seconds_remaining
from ..dependencies import get_runner_deps, get_task_runner
from ..security import require_access, require_json_body
from ..services.github import GitHubError, PullRequest, parse_repository, validate_branch_name
from ..services.runner import RunnerDeps, TaskRunner, pull_request_for_task
from ..services.store import branch_for_task, normalize_task_id, task_store

logger = logging.getLogger("jules.tasks")

router = APIRouter(tags=["tasks"], dependencies=[Depends(require_access)])

HEARTBEAT_SECONDS = 10.0


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class TaskCreateRequest(BaseModel):
    repository: str = Field(
        ...,
        description="Target repository as owner/name or a GitHub URL.",
        json_schema_extra={"example": "octocat/hello-world"},
    )
    branch: Optional[str] = Field(
        None,
        description="Branch to base the change on. Defaults to the repository's default branch.",
        json_schema_extra={"example": "main"},
    )
    prompt: str = Field(
        ...,
        max_length=8000,
        description="What the agent should do.",
        json_schema_extra={"example": "Fix the failing date parsing test and add a regression test"},
    )

    @field_validator("repository")
    @classmethod
    def _check_repository(cls, value: str) -> str:
        return parse_repository(value)

    @field_validator("branch")
    @classmethod
    def _check_branch(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        return validate_branch_name(value)

    @field_validator("prompt")
    @classmethod
    def _check_prompt(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Describe the task for the agent.")
        return cleaned


class LogEntry(BaseModel):
    ts: float
    level: str
    message: str


class TaskResponse(BaseModel):
    task_id: str
    repository: str
    branch: str = ""
    prompt: str = ""
    status: str  # pending | running | completed | failed
    stage: Optional[str] = None
    logs: List[str] = []
    log_entries: List[LogEntry] = []
    # pull_request | no_changes | initial_commit (the first commit of a repository that was empty)
    outcome: Optional[str] = None
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None
    head_branch: Optional[str] = None
    base_branch: Optional[str] = None
    commit_sha: Optional[str] = None
    commit_url: Optional[str] = None  # the commit on GitHub, when the result was committed directly
    repository_url: Optional[str] = None
    empty_repository: Optional[bool] = None  # the repository had no commits when the task started
    new_project: Optional[bool] = None  # it had no code: the agent built a project from scratch
    changed_files: List[Dict[str, Any]] = []
    docs_only: Optional[bool] = None  # the change only touches documentation files
    files_read: List[str] = []  # repository files the agent looked at
    verification: List[Dict[str, Any]] = []
    # passed | failed (the last run that really tested something) | no_tests (a test
    # command was run but the project has no tests) | not_run
    tests_status: Optional[str] = None
    tests_passed: Optional[bool] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    sandbox: Optional[str] = None
    source: str = "api"
    recovered: bool = False
    issue_number: Optional[int] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_budget(settings: Settings) -> float:
    """Seconds this run may take: the configured budget, capped by what the
    hosting platform still allows for the current invocation."""
    budget = float(settings.task_budget_seconds)
    platform_remaining = platform_seconds_remaining()
    if platform_remaining is not None:
        budget = min(budget, platform_remaining - 15.0)
    return max(20.0, budget)


# --------------------------------------------------------------------------- #
# Streaming response
# --------------------------------------------------------------------------- #
class TaskStreamResponse(Response):
    """Streams events produced by a blocking worker as NDJSON.

    The worker runs in a thread. If the client disconnects the worker keeps
    going - the request stays open until it finishes - so a task that is about
    to open its pull request is not thrown away when a browser tab closes.
    """

    media_type = "application/x-ndjson"

    def __init__(self, producer: Callable[[Callable[[Dict[str, Any]], None]], None]) -> None:
        self.producer = producer
        self.status_code = 200
        self.background = None
        self.init_headers(
            {
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Content-Type": "application/x-ndjson; charset=utf-8",
            }
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        finished = object()

        def emit(event: Dict[str, Any]) -> None:  # called from the worker thread
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def work() -> None:
            try:
                self.producer(emit)
            except Exception:
                logger.exception("task stream producer crashed")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, finished)

        async def heartbeat() -> None:
            # Keeps proxies from closing a quiet connection during long steps.
            while True:
                await asyncio.sleep(HEARTBEAT_SECONDS)
                queue.put_nowait({"type": "ping", "ts": time.time()})

        await send(
            {"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}
        )
        worker = loop.run_in_executor(None, work)
        pinger = asyncio.ensure_future(heartbeat())
        client_connected = True
        try:
            while True:
                item = await queue.get()
                if item is finished:
                    break
                if not client_connected:
                    continue
                line = json.dumps(item, separators=(",", ":"), ensure_ascii=False) + "\n"
                try:
                    await send(
                        {"type": "http.response.body", "body": line.encode("utf-8"), "more_body": True}
                    )
                except OSError:
                    client_connected = False
        finally:
            pinger.cancel()
            await asyncio.shield(worker)
        if client_connected:
            try:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            except OSError:
                pass


class PullRequestStatus(BaseModel):
    """A task's pull request as it is on GitHub right now."""

    task_id: str
    repository: str
    number: int
    url: str
    state: str  # open | merged | closed
    title: str = ""
    head_branch: str = ""
    base_branch: str = ""


def _without_logs(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if key not in ("logs", "log_entries")}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.post(
    "",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_json_body)],
)
async def create_task(
    req: TaskCreateRequest,
    background_tasks: BackgroundTasks,
    runner: TaskRunner = Depends(get_task_runner),
):
    """Create a task and run it in the background. Poll ``GET /{task_id}`` for progress."""
    record = task_store.create(repository=req.repository, branch=req.branch, prompt=req.prompt)
    task_store.log(record["task_id"], "info", f"Task queued for {req.repository}")
    background_tasks.add_task(
        runner.run, record["task_id"], budget_seconds=resolve_budget(runner.settings)
    )
    return task_store.get(record["task_id"])


@router.post(
    "/stream",
    response_class=Response,
    dependencies=[Depends(require_json_body)],
    responses={
        200: {
            "description": (
                "Newline-delimited JSON events: `task` (initial record), `log`, `update`, `ping`, "
                "and finally `done` with the complete task record."
            ),
            "content": {"application/x-ndjson": {}},
        }
    },
)
async def stream_task(req: TaskCreateRequest, runner: TaskRunner = Depends(get_task_runner)):
    """Create a task, run it within this request and stream its progress."""
    record = task_store.create(repository=req.repository, branch=req.branch, prompt=req.prompt)
    task_id = record["task_id"]
    budget = resolve_budget(runner.settings)

    def produce(emit: Callable[[Dict[str, Any]], None]) -> None:
        emit({"type": "task", "task": _without_logs(record)})
        try:
            final = runner.run(task_id, budget_seconds=budget, on_event=emit)
        except Exception as exc:  # runner.run reports its own failures; this is a safety net
            logger.exception("task %s failed outside the runner", task_id)
            task_store.update(
                task_id,
                status="failed",
                error=f"Unexpected server error ({exc.__class__.__name__}).",
                completed_at=time.time(),
            )
            final = task_store.get(task_id) or record
        emit({"type": "done", "task": final})

    return TaskStreamResponse(produce)


@router.get("", response_model=List[TaskResponse])
async def list_tasks():
    """Tasks known to this server instance, newest first."""
    return task_store.list()


def _pull_request_state(pull_request: PullRequest) -> str:
    """``open``, ``merged`` or ``closed`` (closed without being merged)."""
    return "merged" if pull_request.merged else pull_request.state


def _find_pull_request(task_id: str, repository: str, deps: RunnerDeps) -> "tuple[str, Optional[PullRequest]]":
    """The repository slug and the pull request the task opened there, if any.

    Raises ValueError for an unusable repository name and GitHubError when
    GitHub could not be asked.
    """
    slug = parse_repository(repository)
    settings = get_settings()
    if settings.github_auth_mode == "none":
        return slug, None
    github = deps.github_factory(settings)
    try:
        return slug, pull_request_for_task(github, slug, task_id)
    finally:
        github.close()


def _find_first_commit(task_id: str, slug: str, deps: RunnerDeps) -> Optional[Dict[str, Any]]:
    """The commit a task made directly to a repository that was empty, if any."""
    settings = get_settings()
    if settings.github_auth_mode == "none":
        return None
    github = deps.github_factory(settings)
    try:
        return github.find_task_commit(slug, task_id)
    finally:
        github.close()


def _recovered_first_commit(task_id: str, slug: str, commit: Dict[str, Any]) -> Dict[str, Any]:
    message = f"Recovered from GitHub: first commit {str(commit['sha'])[:10]}"
    now = time.time()
    return {
        "task_id": task_id,
        "repository": slug,
        "branch": "",
        "prompt": "",
        "status": "completed",
        "stage": "done",
        "logs": [message],
        "log_entries": [{"ts": now, "level": "success", "message": message}],
        "outcome": "initial_commit",
        "commit_sha": commit["sha"],
        "commit_url": commit.get("html_url"),
        "repository_url": f"https://github.com/{slug}",
        "empty_repository": True,
        "new_project": True,
        "summary": commit.get("title") or None,
        "recovered": True,
        "created_at": now,
        "completed_at": now,
    }


def _recover_from_github(task_id: str, repository: str, deps: RunnerDeps) -> Optional[Dict[str, Any]]:
    slug, pull_request = _find_pull_request(task_id, repository, deps)
    if pull_request is None:
        # No pull request: the task may have made the first commit of a
        # repository that was empty.
        commit = _find_first_commit(task_id, slug, deps)
        return _recovered_first_commit(task_id, slug, commit) if commit else None
    state = _pull_request_state(pull_request)
    message = f"Recovered from GitHub: pull request #{pull_request.number} ({state})"
    now = time.time()
    return {
        "task_id": task_id,
        "repository": slug,
        "branch": pull_request.base_branch,
        "prompt": "",
        "status": "completed",
        "stage": "done",
        "logs": [message],
        "log_entries": [{"ts": now, "level": "success", "message": message}],
        "outcome": "pull_request",
        "pr_url": pull_request.html_url,
        "pr_number": pull_request.number,
        "head_branch": pull_request.head_branch or branch_for_task(task_id),
        "base_branch": pull_request.base_branch,
        "summary": pull_request.title or None,
        "recovered": True,
        "created_at": now,
        "completed_at": now,
    }


@router.get("/{task_id}", response_model=TaskResponse)
def get_task_status(
    task_id: str,
    repository: Optional[str] = Query(
        None,
        description=(
            "owner/name of the task's repository. When the task is unknown to this server "
            "instance, its pull request is looked up on GitHub instead."
        ),
    ),
    deps: RunnerDeps = Depends(get_runner_deps),
):
    """Status and logs of a task."""
    normalized = normalize_task_id(task_id)
    record = task_store.get(normalized or task_id)
    if record is not None:
        return record

    if normalized and repository:
        try:
            recovered = _recover_from_github(normalized, repository, deps)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_request", "message": str(exc)})
        except GitHubError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "github_error", "message": exc.describe()},
            )
        if recovered is not None:
            return recovered

    raise HTTPException(
        status_code=404,
        detail={
            "code": "task_not_found",
            "message": (
                "This task is not known to the server instance that answered"
                + (
                    ", and no pull request or commit from it exists on GitHub."
                    if repository
                    else ". Pass ?repository=owner/name to look for its pull request on GitHub."
                )
            ),
        },
    )


@router.get("/{task_id}/pull-request", response_model=PullRequestStatus)
def get_task_pull_request(
    task_id: str,
    repository: str = Query(..., description="owner/name of the task's repository."),
    deps: RunnerDeps = Depends(get_runner_deps),
):
    """The pull request a task opened, as it is on GitHub right now.

    A task record says "pull request opened" forever; whether it has been
    merged or closed since is only known to GitHub. The pull request is found
    through the task's branch name, so this works on any server instance.
    """
    normalized = normalize_task_id(task_id)
    pull_request: Optional[PullRequest] = None
    slug = repository
    if normalized:
        try:
            slug, pull_request = _find_pull_request(normalized, repository, deps)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_request", "message": str(exc)})
        except GitHubError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "github_error", "message": exc.describe()},
            )
    if pull_request is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "pull_request_not_found",
                "message": "No pull request from this task exists on GitHub.",
            },
        )
    return PullRequestStatus(
        task_id=normalized or task_id,
        repository=slug,
        number=pull_request.number,
        url=pull_request.html_url,
        state=_pull_request_state(pull_request),
        title=pull_request.title,
        head_branch=pull_request.head_branch,
        base_branch=pull_request.base_branch,
    )


@router.get("/{task_id}/logs")
async def get_task_logs(task_id: str):
    """Log lines of a task known to this server instance."""
    record = task_store.get(normalize_task_id(task_id) or task_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "task_not_found", "message": "Task not found on this server instance."},
        )
    return {
        "task_id": record["task_id"],
        "status": record["status"],
        "logs": record["logs"],
        "log_entries": record["log_entries"],
    }
