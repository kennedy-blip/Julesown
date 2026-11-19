"""In-memory task store: create, get, list, update, log, eviction."""
from __future__ import annotations

import time

from apps.api.services.store import (
    MAX_LOG_ENTRIES,
    branch_for_task,
    new_task_id,
    normalize_task_id,
    task_store,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_new_task_id_is_unique():
    assert new_task_id() != new_task_id()


def test_normalize_task_id():
    raw = new_task_id()
    assert normalize_task_id(raw) == raw.replace("-", "")
    # With dashes
    dashed = "01234567-89ab-cdef-0123-456789abcdef"
    assert normalize_task_id(dashed) == "0123456789abcdef0123456789abcdef"
    # Invalid
    assert normalize_task_id("not-a-task-id") is None
    assert normalize_task_id("") is None


def test_branch_for_task_is_deterministic():
    task_id = "0123456789abcdef0123456789abcdef"
    assert branch_for_task(task_id) == f"jules/task-{task_id[:12]}"
    assert branch_for_task(task_id.replace("-", "")) == branch_for_task(task_id)


# --------------------------------------------------------------------------- #
# Create / get / update
# --------------------------------------------------------------------------- #


def test_create_and_get():
    task_store.clear()
    record = task_store.create(
        repository="acme/demo",
        branch="main",
        prompt="fix the bug",
        source="api",
    )
    assert record["task_id"]
    assert record["repository"] == "acme/demo"
    assert record["branch"] == "main"
    assert record["prompt"] == "fix the bug"
    assert record["source"] == "api"
    assert record["status"] == "pending"
    assert record["stage"] == "queued"
    assert record["outcome"] is None
    assert record["logs"] == []
    assert record["log_entries"] == []
    assert record["changed_files"] == []
    assert record["files_read"] == []
    assert record["verification"] == []
    assert record["llm_chain"] == []
    assert record["llm_usage"] == []
    assert record["unfinished"] is False
    assert isinstance(record["created_at"], float)

    fetched = task_store.get(record["task_id"])
    assert fetched is not None
    assert fetched["task_id"] == record["task_id"]
    assert fetched["repository"] == "acme/demo"


def test_get_unknown_task_returns_none():
    task_store.clear()
    assert task_store.get("00000000000000000000000000000000") is None


def test_update():
    task_store.clear()
    record = task_store.create(repository="acme/demo", branch="main", prompt="fix it")
    task_id = record["task_id"]
    task_store.update(
        task_id,
        status="completed",
        outcome="pull_request",
        pr_url="https://github.com/acme/demo/pull/5",
        summary="done",
    )
    fetched = task_store.get(task_id)
    assert fetched["status"] == "completed"
    assert fetched["outcome"] == "pull_request"
    assert fetched["pr_url"] == "https://github.com/acme/demo/pull/5"
    assert fetched["summary"] == "done"
    # Unchanged fields stay as they were.
    assert fetched["repository"] == "acme/demo"
    assert fetched["prompt"] == "fix it"


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_log_appends_entries():
    task_store.clear()
    record = task_store.create(repository="acme/demo", branch="main", prompt="fix it")
    task_id = record["task_id"]

    task_store.log(task_id, "info", "cloning repository")
    task_store.log(task_id, "warn", "slow network")

    fetched = task_store.get(task_id)
    assert len(fetched["log_entries"]) == 2
    assert fetched["log_entries"][0]["level"] == "info"
    assert fetched["log_entries"][0]["message"] == "cloning repository"
    assert fetched["log_entries"][0]["task_id"] == task_id
    assert "ts" in fetched["log_entries"][0]
    assert fetched["log_entries"][1]["level"] == "warn"
    assert fetched["log_entries"][1]["message"] == "slow network"

    assert len(fetched["logs"]) == 2
    assert "[INFO" not in fetched["logs"][0]  # format_log_line uses [HH:MM:SS]
    assert "cloning repository" in fetched["logs"][0]


def test_log_unknown_task_is_silent():
    task_store.clear()
    # Should not raise
    task_store.log("nonexistent", "info", "hello")


def test_log_caps_entries():
    task_store.clear()
    record = task_store.create(repository="acme/demo", branch="main", prompt="fix it")
    task_id = record["task_id"]
    for i in range(MAX_LOG_ENTRIES + 5):
        task_store.log(task_id, "info", f"message {i}")
    fetched = task_store.get(task_id)
    assert len(fetched["log_entries"]) == MAX_LOG_ENTRIES


# --------------------------------------------------------------------------- #
# List
# --------------------------------------------------------------------------- #


def test_list_is_sorted_by_created_at_newest_first():
    task_store.clear()
    t1 = task_store.create(repository="a/1", branch="main", prompt="one")
    time.sleep(0.01)
    t2 = task_store.create(repository="a/2", branch="main", prompt="two")
    time.sleep(0.01)
    t3 = task_store.create(repository="a/3", branch="main", prompt="three")

    listed = task_store.list()
    assert [r["task_id"] for r in listed] == [t3["task_id"], t2["task_id"], t1["task_id"]]


def test_list_returns_deep_copies():
    task_store.clear()
    record = task_store.create(repository="acme/demo", branch="main", prompt="fix it")
    listed = task_store.list()
    listed[0]["prompt"] = "mutated"
    fetched = task_store.get(record["task_id"])
    assert fetched["prompt"] == "fix it"


# --------------------------------------------------------------------------- #
# Eviction
# --------------------------------------------------------------------------- #


def test_eviction_removes_finished_tasks_first():
    task_store = __import__(
        "apps.api.services.store", fromlist=["TaskStore"]
    ).TaskStore(max_tasks=2)
    task_store.clear()

    t1 = task_store.create(repository="a/1", branch="main", prompt="one")
    t2 = task_store.create(repository="a/2", branch="main", prompt="two")
    t3 = task_store.create(repository="a/3", branch="main", prompt="three")

    # Mark two as finished
    task_store.update(t1["task_id"], status="completed")
    task_store.update(t2["task_id"], status="completed")

    # Adding a fourth: 4 tasks, max 2, evict 2 finished ones first
    t4 = task_store.create(repository="a/4", branch="main", prompt="four")
    ids = {r["task_id"] for r in task_store.list()}
    assert t4["task_id"] in ids
    assert t3["task_id"] in ids  # still pending, kept
    assert t1["task_id"] not in ids  # finished, evicted
    assert t2["task_id"] not in ids  # finished, evicted


def test_eviction_only_removes_finished_tasks():
    """Pending tasks are never evicted, even when over the limit."""
    local_store = __import__(
        "apps.api.services.store", fromlist=["TaskStore"]
    ).TaskStore(max_tasks=2)
    local_store.clear()

    t1 = local_store.create(repository="a/1", branch="main", prompt="one")
    t2 = local_store.create(repository="a/2", branch="main", prompt="two")
    t3 = local_store.create(repository="a/3", branch="main", prompt="three")

    # All pending: nothing is evicted even though we're over the limit.
    ids = {r["task_id"] for r in local_store.list()}
    assert ids == {t1["task_id"], t2["task_id"], t3["task_id"]}

    # Mark one as finished, add a fourth: the finished one is evicted.
    local_store.update(t1["task_id"], status="completed")
    t4 = local_store.create(repository="a/4", branch="main", prompt="four")
    ids = {r["task_id"] for r in local_store.list()}
    assert t4["task_id"] in ids
    assert t1["task_id"] not in ids
    assert t2["task_id"] in ids
    assert t3["task_id"] in ids
