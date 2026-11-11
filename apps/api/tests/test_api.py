import pytest
import hmac
import hashlib
import json
from fastapi.testclient import TestClient
from apps.api.main import app
from apps.api.routers.webhooks import verify_github_signature
from apps.api.services.sandbox import E2BSandboxManager
from apps.api.services.agent import AutonomousAgent

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "jules-clone-api"

def test_github_signature_verification():
    secret = "test_secret_key"
    payload = b'{"zen": "Practicality beats purity."}'
    valid_sig = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    assert verify_github_signature(payload, secret, valid_sig) is True
    assert verify_github_signature(payload, secret, "sha256=invalid") is False
    assert verify_github_signature(payload, secret, None) is False

def test_github_webhook_ping():
    response = client.post(
        "/api/v1/webhooks/github",
        headers={"x-github-event": "ping"},
        json={"zen": "Non-functional requirements matter."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["zen"] == "Non-functional requirements matter."

def test_create_task_and_get_status():
    payload = {
        "repository": "https://github.com/org/sample-repo",
        "branch": "main",
        "prompt": "Fix typo in readme and run tests"
    }
    response = client.post("/api/v1/tasks", json=payload)
    assert response.status_code == 201
    created_task = response.json()
    assert "task_id" in created_task
    assert created_task["repository"] == payload["repository"]

    # Poll status
    task_id = created_task["task_id"]
    get_res = client.get(f"/api/v1/tasks/{task_id}")
    assert get_res.status_code == 200
    assert get_res.json()["task_id"] == task_id

def test_sandbox_and_agent_execution():
    sandbox = E2BSandboxManager()
    agent = AutonomousAgent(sandbox=sandbox)
    result = agent.run_agent_loop(prompt="Add unit tests", repo="my/repo", branch="main")
    assert result["success"] is True
    assert "pr_result" in result
