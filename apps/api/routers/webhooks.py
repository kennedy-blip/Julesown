"""GitHub webhook receiver.

A task is started when a trusted user writes a line beginning with ``/jules``
in an issue (when opening it) or in an issue comment, for example::

    /jules fix this and add a regression test

The same line in a comment on a *pull request* asks for more work on that pull
request: the result is added to its branch. ``/jules fix the checks`` (or just
``/jules``) there asks for its failed checks to be fixed.

Failed checks are also acted on unasked, for pull requests the agent opened
itself (branches named ``jules/...``): when GitHub reports that a check suite,
a workflow run or a commit status failed, the agent reads what failed, fixes
it and pushes the fix to the pull request - at most
``JULES_CI_AUTOFIX_ATTEMPTS`` times per pull request, counted by the comments
it leaves, so that a check it cannot fix does not keep it busy forever.

The delivery must carry a valid ``X-Hub-Signature-256`` for the configured
``GITHUB_WEBHOOK_SECRET``. The result (pull request link or failure reason) is
posted back as a comment. Events that do not start a task are acknowledged and
reported as such - nothing is ever claimed to be "queued" unless a task really
was started.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..dependencies import get_task_runner
from ..services.ci import attempted_fixes, fix_marker, names_of
from ..services.common import clip
from ..services.github import FAILED_CHECK_CONCLUSIONS, FAILED_STATUS_STATES, GitHubError, parse_repository
from ..services.runner import TaskRunner
from ..services.store import task_store
from .tasks import resolve_budget

logger = logging.getLogger("jules.webhooks")
router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])

TRIGGER_RE = re.compile(r"^[ \t]*/jules\b[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
# "/jules fix the checks", "/jules fix ci", "/jules repair the failing build" ...
FIX_CHECKS_RE = re.compile(
    r"^(?:please\s+)?(?:fix|repair|resolve)\s+(?:the\s+|this\s+)?(?:failing\s+|failed\s+|broken\s+|red\s+)?"
    r"(?:ci|checks?|builds?|tests?|deployments?|pipelines?)\W*$",
    re.IGNORECASE,
)
TRUSTED_ASSOCIATIONS = ("OWNER", "MEMBER", "COLLABORATOR")
# Branches the agent creates for its pull requests. Checks are only fixed
# unasked on these: other people's branches are theirs to fix.
OWN_BRANCH_PREFIX = "jules/"
STOPPED_MARKER = "<!-- jules-ci-fix-stopped -->"
_MAX_REMEMBERED_DELIVERIES = 500

_deliveries_lock = threading.Lock()
_seen_deliveries: "OrderedDict[str, None]" = OrderedDict()
# Commits whose failed checks are already being looked at by this instance:
# one failed commit usually arrives as several events (suite, run, status).
_claimed_commits: "OrderedDict[str, None]" = OrderedDict()


def verify_github_signature(payload_body: bytes, secret: str, signature_header: Optional[str]) -> bool:
    """Validate GitHub's ``X-Hub-Signature-256`` header (HMAC-SHA256 of the raw body)."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = signature_header[len("sha256=") :]
    computed = hmac.new(
        key=secret.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, expected)


@dataclass
class Trigger:
    repository: str
    issue_number: int
    prompt: str
    requested_by: str
    requester_id: Optional[int] = None
    # Set for a comment on a pull request: the work is added to that pull request.
    pull_request: Optional[int] = None
    fix_checks: bool = False


@dataclass
class CheckFailure:
    """A failed check GitHub told us about."""

    repository: str
    branch: str
    sha: str
    what: str  # the check that failed, in a few words


def _claim_commit(repository: str, sha: str) -> bool:
    """True the first time a commit's failure is seen by this instance."""
    key = f"{repository.lower()}@{sha}"
    with _deliveries_lock:
        if key in _claimed_commits:
            return False
        _claimed_commits[key] = None
        while len(_claimed_commits) > _MAX_REMEMBERED_DELIVERIES:
            _claimed_commits.popitem(last=False)
    return True


def extract_check_failure(event: str, payload: Dict[str, Any]) -> Tuple[Optional[CheckFailure], str]:
    """A failed check on one of the agent's own branches, from a ``check_suite``,
    ``check_run``, ``workflow_run`` or ``status`` event. ``(None, reason)`` when
    the event is nothing to act on."""
    repository_name = (payload.get("repository") or {}).get("full_name")
    try:
        repository = parse_repository(repository_name)
    except (TypeError, ValueError):
        return None, "the event did not identify a repository"
    branch, sha, what = "", "", ""
    if event == "status":
        if payload.get("state") not in FAILED_STATUS_STATES:
            return None, f"commit status '{payload.get('state')}' is not a failure"
        sha = str(payload.get("sha") or "")
        names = [str(item.get("name") or "") for item in payload.get("branches") or [] if isinstance(item, dict)]
        branch = next((name for name in names if name.startswith(OWN_BRANCH_PREFIX)), names[0] if names else "")
        what = str(payload.get("context") or "a commit status")
    elif event in ("check_suite", "workflow_run", "check_run"):
        subject = payload.get(event) or {}
        if payload.get("action") != "completed":
            return None, f"'{event}' events ({payload.get('action')}) are not results"
        if subject.get("conclusion") not in FAILED_CHECK_CONCLUSIONS:
            return None, f"{event.replace('_', ' ')} concluded '{subject.get('conclusion')}', not a failure"
        suite = subject.get("check_suite") if event == "check_run" else subject
        branch = str((suite or {}).get("head_branch") or "")
        sha = str(subject.get("head_sha") or (suite or {}).get("head_sha") or "")
        head_repository = (subject.get("head_repository") or {}).get("full_name")
        if head_repository and str(head_repository).lower() != repository.lower():
            return None, "the failed run belongs to a fork"
        what = str(subject.get("name") or (subject.get("app") or {}).get("name") or event.replace("_", " "))
    else:
        return None, ""
    if not sha or not branch:
        return None, "the event did not say which branch and commit failed"
    if not branch.startswith(OWN_BRANCH_PREFIX):
        return None, (
            f"checks are only fixed unasked on pull requests opened by Jules (branches named "
            f"{OWN_BRANCH_PREFIX}...); comment '/jules fix the checks' on a pull request to ask for it"
        )
    return CheckFailure(repository=repository, branch=branch, sha=sha, what=what), ""


def _already_delivered(delivery_id: Optional[str]) -> bool:
    """GitHub can redeliver an event; do not start the same task twice."""
    if not delivery_id:
        return False
    with _deliveries_lock:
        if delivery_id in _seen_deliveries:
            return True
        _seen_deliveries[delivery_id] = None
        while len(_seen_deliveries) > _MAX_REMEMBERED_DELIVERIES:
            _seen_deliveries.popitem(last=False)
    return False


def extract_trigger(event: str, payload: Dict[str, Any]) -> Tuple[Optional[Trigger], str]:
    """Decide whether a webhook event asks for a task. Returns ``(trigger, reason)``;
    ``trigger`` is ``None`` when the event should only be acknowledged."""
    action = payload.get("action")
    issue = payload.get("issue") or {}
    on_pull_request = False
    if event == "issue_comment" and action == "created":
        source = payload.get("comment") or {}
        on_pull_request = bool(issue.get("pull_request"))
    elif event == "issues" and action == "opened":
        source = issue
    else:
        return None, f"'{event}' events{f' ({action})' if action else ''} do not start tasks"

    text = source.get("body") or ""
    match = TRIGGER_RE.search(text)
    if not match:
        return None, "no line starting with /jules"
    if (payload.get("sender") or {}).get("type") == "Bot":
        return None, "commands from bots are ignored"
    association = str(source.get("author_association") or "").upper()
    if association not in TRUSTED_ASSOCIATIONS:
        return None, "only repository owners, members and collaborators can start tasks"

    try:
        repository = parse_repository((payload.get("repository") or {}).get("full_name"))
        issue_number = int(issue.get("number"))
    except (TypeError, ValueError):
        return None, "the event did not identify a repository and issue"

    author = str((source.get("user") or {}).get("login") or "unknown")
    author_id = (source.get("user") or {}).get("id")
    instructions = (match.group(1) + "\n" + text[match.end() :]).strip()
    if on_pull_request:
        # More work on this pull request. Without instructions, or when asked
        # to, that means fixing its failed checks.
        fix_checks = not instructions or FIX_CHECKS_RE.match(instructions) is not None
        prompt = (
            f"Fix the failed checks of pull request #{issue_number}"
            if fix_checks
            else f"Instructions from @{author} on pull request #{issue_number}: {instructions}"
        )
        return (
            Trigger(
                repository=repository,
                issue_number=issue_number,
                prompt=prompt[:8000],
                requested_by=author,
                requester_id=author_id if isinstance(author_id, int) else None,
                pull_request=issue_number,
                fix_checks=fix_checks,
            ),
            "",
        )
    issue_body = issue.get("body") or ""
    if source is issue:
        issue_body = TRIGGER_RE.sub("", issue_body, count=1)
    prompt = "\n\n".join(
        part
        for part in (
            f"GitHub issue #{issue_number}: {issue.get('title') or ''}".strip(),
            clip(issue_body.strip(), 4000, tail_share=0.0),
            f"Instructions from @{author}: {instructions or 'Resolve this issue.'}",
        )
        if part
    )
    return (
        Trigger(
            repository=repository,
            issue_number=issue_number,
            prompt=prompt[:8000],
            requested_by=author,
            requester_id=author_id if isinstance(author_id, int) else None,
        ),
        "",
    )


def build_result_comment(task: Dict[str, Any]) -> str:
    footer = f"\n\n<sub>Jules Clone · task `{task['task_id']}`</sub>"
    summary = (task.get("summary") or "").strip()
    if task.get("status") == "completed" and task.get("outcome") == "pull_request":
        tests = {
            "passed": "Tests passed in the sandbox.",
            "failed": "⚠️ The last test run failed - see the pull request.",
            "no_tests": "The repository has no automated tests, so the change is untested.",
            "not_run": "⚠️ No tests were run.",
        }.get(task.get("tests_status") or "", "")
        if task.get("docs_only") and task.get("tests_status") in ("not_run", "no_tests"):
            tests = "The change only touches documentation, so no tests were needed."
        body = f"✅ I opened #{task['pr_number']}: {task['pr_url']}"
        return "\n\n".join(part for part in (body, summary, tests) if part) + footer
    if task.get("status") == "completed" and task.get("outcome") == "initial_commit":
        where = task.get("commit_url") or f"commit `{str(task.get('commit_sha') or '')[:10]}`"
        branch = task.get("base_branch") or "the default branch"
        tests = {
            "passed": "Tests passed in the sandbox.",
            "failed": "⚠️ The last test run failed - check the project before you rely on it.",
        }.get(task.get("tests_status") or "", "The new project has no automated tests yet.")
        if task.get("docs_only"):
            tests = ""
        body = (
            f"✅ The repository was empty, so there was nothing to open a pull request against. "
            f"I committed the project directly to `{branch}`: {where}"
        )
        return "\n\n".join(part for part in (body, summary, tests) if part) + footer
    if task.get("status") == "completed" and task.get("fix_checks"):
        return "ℹ️ There was nothing for me to fix: " + (
            "no check has failed on the head commit of this pull request."
            if not task.get("failed_checks")
            else "I looked into the failed checks but did not change any files."
            + (f"\n\n{summary}" if summary else "")
        ) + footer
    if task.get("status") == "completed":
        return (
            "ℹ️ I worked on this but did not change any files, so there is "
            + ("nothing to add to this pull request." if task.get("pull_request") else "no pull request.")
            + (f"\n\n{summary}" if summary else "")
            + footer
        )
    reason = (task.get("error") or "Unknown error.").strip()
    quoted = "\n".join(f"> {line}" for line in reason.splitlines())
    return f"❌ I could not complete this task.\n\n{quoted}{footer}"


def run_and_report(runner: TaskRunner, task_id: str, budget: float) -> None:
    """Run a webhook-triggered task, then comment the outcome on its issue or pull request."""
    task = runner.run(task_id, budget_seconds=budget)
    issue_number = task.get("issue_number")
    if not issue_number:
        return
    if task.get("outcome") == "pr_updated":
        return  # the runner has told the pull request what was pushed to it
    github = None
    try:
        github = runner.deps.github_factory(runner.settings)
        github.create_issue_comment(task["repository"], int(issue_number), build_result_comment(task))
    except GitHubError as exc:
        logger.warning("could not comment on %s#%s: %s", task["repository"], issue_number, exc)
        task_store.log(task_id, "warn", f"Could not comment on the issue: {exc.describe()}")
    except Exception:
        logger.exception("could not comment on %s#%s", task["repository"], issue_number)
    finally:
        if github is not None:
            github.close()


def autofix_checks(runner: TaskRunner, failure: CheckFailure, budget: float) -> Optional[str]:
    """Fix the failed checks of one of the agent's own pull requests.

    Runs after the webhook was answered. Everything it needs to know is asked
    from GitHub - the server remembers nothing between requests: which pull
    request the branch belongs to, whether the failed commit is still its
    head, and how often a fix was already attempted (the comments this leaves
    carry a marker). Returns the id of the task that was run, if one was.
    """
    settings = runner.settings
    slug = failure.repository
    github = None
    try:
        github = runner.deps.github_factory(settings)
        pull = github.find_pull_request(slug, failure.branch)
        if pull is None or pull.merged or pull.state != "open":
            logger.info("checks failed on %s@%s, but it has no open pull request", slug, failure.branch)
            return None
        if pull.head_sha and pull.head_sha != failure.sha:
            return None  # the branch has moved on since: this result is about an older commit
        comments = github.list_issue_comments(slug, pull.number)
        attempts = attempted_fixes(comments)
        if failure.sha in attempts:
            return None  # another instance is already on it
        limit = settings.ci_autofix_attempts
        if len(attempts) >= limit:
            if not any(STOPPED_MARKER in str(comment.get("body") or "") for comment in comments):
                github.create_issue_comment(
                    slug,
                    pull.number,
                    f"⚠️ The checks still fail after {len(attempts)} automatic "
                    f"fix{'es' if len(attempts) != 1 else ''}, so I am leaving it at that. Comment "
                    "`/jules fix the checks` to have me try again, or `/jules` with instructions.\n\n"
                    f"{STOPPED_MARKER}",
                )
            return None
        checks = github.checks_for(slug, failure.sha)
        if checks["state"] != "failure":
            return None  # the check was run again and passed, or cannot be read
        github.create_issue_comment(
            slug,
            pull.number,
            f"🔧 The checks failed on `{failure.sha[:10]}` ({names_of(checks)}). I am working on a fix "
            f"(attempt {len(attempts) + 1} of {limit}).\n\n{fix_marker(failure.sha)}",
        )
    except GitHubError as exc:
        logger.warning("could not prepare a fix for %s@%s: %s", slug, failure.sha[:10], exc.describe())
        return None
    finally:
        if github is not None:
            github.close()

    record = task_store.create(
        repository=slug,
        branch=None,
        prompt=f"Fix the failed checks of pull request #{pull.number}",
        source="webhook",
        extra={"issue_number": pull.number, "pull_request": pull.number, "fix_checks": True, "automatic": True},
    )
    task_store.log(
        record["task_id"], "info", f"Task started because {failure.what} failed on {slug}#{pull.number}"
    )
    run_and_report(runner, record["task_id"], budget)
    return record["task_id"]


@router.post("/github")
async def github_webhook_endpoint(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
    x_github_delivery: Optional[str] = Header(None),
    runner: TaskRunner = Depends(get_task_runner),
):
    settings = get_settings()
    body = await request.body()

    secret = settings.github_webhook_secret
    signature_verified = False
    if secret:
        if not verify_github_signature(body, secret, x_hub_signature_256):
            logger.warning("rejected webhook with an invalid signature")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-Hub-Signature-256 HMAC signature",
            )
        signature_verified = True

    try:
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Malformed JSON: {exc}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed JSON: expected an object")

    event = x_github_event or "unknown"
    if event == "ping":
        return {
            "status": "ok",
            "message": "GitHub ping received successfully",
            "zen": payload.get("zen"),
            "signature_verified": signature_verified,
        }

    def acknowledged(reason: str) -> Dict[str, Any]:
        return {"status": "received", "event": event, "dispatched": False, "task": None, "reason": reason}

    if event in ("check_suite", "check_run", "workflow_run", "status"):
        failure, reason = extract_check_failure(event, payload)
        if failure is None:
            return acknowledged(reason or f"'{event}' events do not start tasks")
        if not signature_verified:
            return acknowledged(
                "GITHUB_WEBHOOK_SECRET is not configured, so tasks are not started from webhooks"
            )
        if not settings.ci_autofix or settings.ci_autofix_attempts <= 0:
            return acknowledged("fixing failed checks unasked is switched off (JULES_CI_AUTOFIX)")
        if _already_delivered(x_github_delivery):
            return acknowledged("this delivery was already processed")
        if not _claim_commit(failure.repository, failure.sha):
            return acknowledged("the failed checks of this commit are already being looked at")
        # Whether a task starts is decided after this response (GitHub waits 10
        # seconds): the pull request, its head and earlier attempts are looked up first.
        background_tasks.add_task(autofix_checks, runner, failure, resolve_budget(settings))
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "accepted",
                "event": event,
                "dispatched": False,
                "task": None,
                "reason": None,
                "checking": (
                    f"{failure.what} failed on {failure.branch} @ {failure.sha[:10]}: looking up its pull "
                    "request and earlier fix attempts"
                ),
            },
        )

    trigger, reason = extract_trigger(event, payload)
    if trigger is None:
        return acknowledged(reason)
    if not signature_verified:
        return acknowledged(
            "GITHUB_WEBHOOK_SECRET is not configured, so tasks are not started from webhooks"
        )
    if _already_delivered(x_github_delivery):
        return acknowledged("this delivery was already processed")

    record = task_store.create(
        repository=trigger.repository,
        branch=None,
        prompt=trigger.prompt,
        source="webhook",
        extra={
            "issue_number": trigger.issue_number,
            "pull_request": trigger.pull_request,
            "fix_checks": trigger.fix_checks,
            # Commits can be attributed to whoever asked for the work.
            "requested_by": {"login": trigger.requested_by, "id": trigger.requester_id},
        },
    )
    task_store.log(
        record["task_id"],
        "info",
        f"Task started by @{trigger.requested_by} from {trigger.repository}#{trigger.issue_number}",
    )
    # Runs after this response is sent; GitHub only waits 10 seconds for a reply.
    background_tasks.add_task(run_and_report, runner, record["task_id"], resolve_budget(settings))
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "event": event,
            "dispatched": True,
            "reason": None,
            "task": {
                "task_id": record["task_id"],
                "repository": trigger.repository,
                "issue_number": trigger.issue_number,
                "status": "pending",
            },
        },
    )
