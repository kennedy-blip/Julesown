"""GitHub webhook receiver.

A task is started when a trusted user writes a line beginning with ``/jules``
in an issue (when opening it) or in an issue comment, for example::

    /jules fix this and add a regression test

The delivery must carry a valid ``X-Hub-Signature-256`` for the configured
``GITHUB_WEBHOOK_SECRET``. The result (pull request link or failure reason) is
posted back to the issue as a comment. Events that do not start a task are
acknowledged and reported as such - nothing is ever claimed to be "queued"
unless a task really was started.
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
from ..services.common import clip
from ..services.github import GitHubError, parse_repository
from ..services.runner import TaskRunner
from ..services.store import task_store
from .tasks import resolve_budget

logger = logging.getLogger("jules.webhooks")
router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])

TRIGGER_RE = re.compile(r"^[ \t]*/jules\b[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
TRUSTED_ASSOCIATIONS = ("OWNER", "MEMBER", "COLLABORATOR")
_MAX_REMEMBERED_DELIVERIES = 500

_deliveries_lock = threading.Lock()
_seen_deliveries: "OrderedDict[str, None]" = OrderedDict()


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
    if event == "issue_comment" and action == "created":
        source = payload.get("comment") or {}
        if issue.get("pull_request"):
            return None, "comments on pull requests do not start tasks"
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
    instructions = (match.group(1) + "\n" + text[match.end() :]).strip()
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
    return Trigger(repository=repository, issue_number=issue_number, prompt=prompt[:8000], requested_by=author), ""


def build_result_comment(task: Dict[str, Any]) -> str:
    footer = f"\n\n<sub>Jules Clone · task `{task['task_id']}`</sub>"
    summary = (task.get("summary") or "").strip()
    if task.get("status") == "completed" and task.get("outcome") == "pull_request":
        tests = {
            "passed": "Tests passed in the sandbox.",
            "failed": "⚠️ The last test run failed - see the pull request.",
            "not_run": "⚠️ No tests were run.",
        }.get(task.get("tests_status") or "", "")
        body = f"✅ I opened #{task['pr_number']}: {task['pr_url']}"
        return "\n\n".join(part for part in (body, summary, tests) if part) + footer
    if task.get("status") == "completed":
        return (
            "ℹ️ I worked on this but did not change any files, so there is no pull request."
            + (f"\n\n{summary}" if summary else "")
            + footer
        )
    reason = (task.get("error") or "Unknown error.").strip()
    quoted = "\n".join(f"> {line}" for line in reason.splitlines())
    return f"❌ I could not complete this task.\n\n{quoted}{footer}"


def run_and_report(runner: TaskRunner, task_id: str, budget: float) -> None:
    """Run a webhook-triggered task, then comment the outcome on its issue."""
    task = runner.run(task_id, budget_seconds=budget)
    issue_number = task.get("issue_number")
    if not issue_number:
        return
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
        extra={"issue_number": trigger.issue_number},
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
