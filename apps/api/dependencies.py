"""FastAPI dependencies shared by the routers.

Tests replace ``get_runner_deps`` (via ``app.dependency_overrides``) to swap the
GitHub client, sandbox and LLM for local stand-ins.
"""
from __future__ import annotations

from fastapi import Depends

from .config import get_settings
from .services.runner import RunnerDeps, TaskRunner, default_deps
from .services.store import task_store


def get_runner_deps() -> RunnerDeps:
    return default_deps()


def get_task_runner(deps: RunnerDeps = Depends(get_runner_deps)) -> TaskRunner:
    return TaskRunner(get_settings(), task_store, deps)
