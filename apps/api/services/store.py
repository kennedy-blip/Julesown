"""In-memory task records.

This store lives inside one server process. On serverless platforms a later
request can land on a different instance that has never seen the task, so the
API never relies on it for correctness: results are streamed to the client
over the request that runs the task, and a finished task can always be
recovered from GitHub (its pull request lives on a branch named after the task
id, see :func:`branch_for_task`).
"""
from __future__ import annotations

import copy
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

TERMINAL_STATUSES = ("completed", "failed")
MAX_LOG_ENTRIES = 2000
_TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def new_task_id() -> str:
    return uuid.uuid4().hex


def normalize_task_id(raw: str) -> Optional[str]:
    """Accept a task id with or without dashes; ``None`` if it is not one."""
    candidate = (raw or "").strip().lower().replace("-", "")
    return candidate if _TASK_ID_RE.match(candidate) else None


def branch_for_task(task_id: str) -> str:
    """The branch a task's pull request is opened from (deterministic, so the
    outcome of a task can be looked up on GitHub from its id alone)."""
    return f"jules/task-{task_id.replace('-', '')[:12]}"


def format_log_line(entry: Dict[str, Any]) -> str:
    stamp = time.strftime("%H:%M:%S", time.gmtime(entry["ts"]))
    return f"[{stamp}] {entry['message']}"


class TaskStore:
    def __init__(self, max_tasks: int = 200) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._max_tasks = max_tasks

    def create(
        self,
        *,
        repository: str,
        branch: Optional[str],
        prompt: str,
        source: str = "api",
        task_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        record: Dict[str, Any] = {
            "task_id": task_id or new_task_id(),
            "repository": repository,
            "branch": branch or "",
            "prompt": prompt,
            "status": "pending",
            "stage": "queued",
            "logs": [],
            "log_entries": [],
            "outcome": None,
            "pr_url": None,
            "pr_number": None,
            "head_branch": None,
            "base_branch": None,
            "commit_sha": None,
            "changed_files": [],
            "docs_only": None,
            "files_read": [],
            "verification": [],
            "tests_status": None,
            "tests_passed": None,
            "summary": None,
            "error": None,
            "llm_provider": None,
            "llm_model": None,
            "sandbox": None,
            "source": source,
            "recovered": False,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
        }
        if extra:
            record.update(extra)
        with self._lock:
            self._tasks[record["task_id"]] = record
            if len(self._tasks) > self._max_tasks:
                finished = [
                    key
                    for key, value in self._tasks.items()
                    if value["status"] in TERMINAL_STATUSES
                ]
                for key in finished[: len(self._tasks) - self._max_tasks]:
                    self._tasks.pop(key, None)
            return copy.deepcopy(record)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._tasks.get(task_id)
            return copy.deepcopy(record) if record is not None else None

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = [copy.deepcopy(record) for record in self._tasks.values()]
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def update(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                record.update(fields)

    def log(self, task_id: str, level: str, message: str) -> Dict[str, Any]:
        entry = {"ts": time.time(), "level": level, "message": message}
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None and len(record["log_entries"]) < MAX_LOG_ENTRIES:
                record["log_entries"].append(entry)
                record["logs"].append(format_log_line(entry))
        return entry

    def clear(self) -> None:
        with self._lock:
            self._tasks.clear()


# Process-wide store used by the API routers.
task_store = TaskStore()
