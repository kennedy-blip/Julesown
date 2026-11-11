import uuid
import time
import asyncio
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from ..services.sandbox import E2BSandboxManager
from ..services.agent import AutonomousAgent

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

class TaskCreateRequest(BaseModel):
    repository: str = Field(..., json_schema_extra={"example": "https://github.com/example/repo"}, description="Target repository URL or owner/repo")
    branch: str = Field("main", json_schema_extra={"example": "main"}, description="Target branch to base changes from")
    prompt: str = Field(..., json_schema_extra={"example": "Fix bug in calculation algorithm and add test coverage"}, description="Task instructions for the agent")

class TaskStatusResponse(BaseModel):
    task_id: str
    repository: str
    branch: str
    prompt: str
    status: str  # pending, running, completed, failed
    logs: List[str]
    pr_url: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None
    tests_passed: Optional[bool] = None

# In-memory storage for demonstration and fast serverless execution state
TASKS_DB: Dict[str, Dict[str, Any]] = {}

def clean_repo_slug(raw_repo: str) -> str:
    """Extracts clean owner/repo slug from repository string or URL."""
    slug = (
        raw_repo.replace("https://github.com/", "")
        .replace("http://github.com/", "")
        .replace(".git", "")
        .strip("/")
    )
    return slug or "owner/repo"

def execute_agent_task(task_id: str, repo: str, branch: str, prompt: str):
    """Background task runner executing the autonomous agent loop."""
    task = TASKS_DB.get(task_id)
    if not task:
        return

    clean_repo = clean_repo_slug(repo)
    default_pr_url = f"https://github.com/{clean_repo}/pulls"

    task["status"] = "running"
    task["logs"].append(f"[{time.strftime('%X')}] 🚀 Task initiated for {clean_repo}:{branch}")

    def log_handler(msg: str):
        task["logs"].append(f"[{time.strftime('%X')}] {msg}")

    try:
        sandbox = E2BSandboxManager()
        agent = AutonomousAgent(sandbox=sandbox, log_callback=log_handler)
        result = agent.run_agent_loop(prompt=prompt, repo=repo, branch=branch)

        task["completed_at"] = time.time()
        if result.get("success"):
            task["status"] = "completed"
            task["tests_passed"] = True
            pr_info = result.get("pr_result") or {}
            task["pr_url"] = pr_info.get("pr_url") or default_pr_url
            task["logs"].append(f"[{time.strftime('%X')}] 🎉 PR created: {task['pr_url']}")
        else:
            task["status"] = "failed"
            task["logs"].append(f"[{time.strftime('%X')}] ❌ Execution failed: {result.get('error')}")

    except Exception as e:
        task["status"] = "failed"
        task["logs"].append(f"[{time.strftime('%X')}] ❌ Unhandled error: {str(e)}")
        task["completed_at"] = time.time()


@router.post("", response_model=TaskStatusResponse, status_code=status.HTTP_201_CREATED)
async def create_task(req: TaskCreateRequest, background_tasks: BackgroundTasks):
    """
    Creates a new autonomous coding agent task and dispatches background execution.
    """
    task_id = str(uuid.uuid4())
    task_record = {
        "task_id": task_id,
        "repository": req.repository,
        "branch": req.branch,
        "prompt": req.prompt,
        "status": "pending",
        "logs": [f"[{time.strftime('%X')}] Task queued."],
        "pr_url": None,
        "created_at": time.time(),
        "completed_at": None,
        "tests_passed": None
    }
    TASKS_DB[task_id] = task_record

    # Schedule agent execution
    background_tasks.add_task(execute_agent_task, task_id, req.repository, req.branch, req.prompt)

    return task_record


@router.get("", response_model=List[TaskStatusResponse])
async def list_tasks():
    """Lists all active and completed tasks."""
    return list(TASKS_DB.values())


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Fetches real-time status and logs for a given task."""
    task = TASKS_DB.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/{task_id}/logs")
async def get_task_logs(task_id: str):
    """Fetches real-time logs for a specific task."""
    task = TASKS_DB.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "logs": task["logs"], "status": task["status"]}
