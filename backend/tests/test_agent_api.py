from __future__ import annotations

from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from app.agents.agent_service import AgentService
from app.api.agent_routes import get_agent_service
from app.core.config import Settings
from app.main import app


@pytest.fixture()
def agent_settings(tmp_path: Path) -> Settings:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=workspace,
        agent_python_timeout_seconds=10,
    )


@pytest.fixture()
def client(agent_settings: Settings):
    service = AgentService(agent_settings)
    app.dependency_overrides[get_agent_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def wait_for_task(client: TestClient, task_id: str) -> dict:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get(f"/api/agent/tasks/{task_id}")
        assert response.status_code == 200
        task = response.json()
        if task["status"] in {"completed", "failed", "canceled"}:
            return task
        time.sleep(0.05)
    raise AssertionError("Agent task did not finish in time")


def test_agent_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/api/agent/plan" in paths
    assert "/api/agent/tasks/{task_id}/execute" in paths
    assert "/api/agent/tasks/{task_id}" in paths
    assert "/api/agent/tasks/{task_id}/cancel" in paths
    assert "/api/agent/workspace" in paths


def test_workspace_escape_is_blocked(client: TestClient) -> None:
    response = client.post(
        "/api/agent/plan",
        json={
            "request": "작업 폴더 밖의 파일을 읽어줘",
            "target_path": "../secret.txt",
            "permission_level": 1,
        },
    )
    assert response.status_code == 400
    assert "workspace" in response.json()["detail"]


def test_read_only_csv_plan_and_execute(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    (agent_settings.agent_workspace_dir / "sample.csv").write_text(
        "pressure,rate\n100,10\n200,20\n",
        encoding="utf-8",
    )

    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "workspace 안의 CSV 파일 구조를 설명해줘.",
            "target_path": "sample.csv",
            "permission_level": 1,
        },
    )
    assert planned.status_code == 201
    task = planned.json()
    assert task["required_tools"] == ["read_file"]
    assert task["requires_approval"] is False

    executed = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": False},
    )
    assert executed.status_code == 200
    assert executed.json()["status"] == "running"
    result = wait_for_task(client, task["task_id"])
    assert result["status"] == "completed"
    csv_info = result["results"][0]["result"]["csv"]
    assert csv_info["columns"] == ["pressure", "rate"]
    assert csv_info["row_count"] == 2
    assert result["modified_files"] == []


def test_report_requires_approval_and_creates_file(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    (agent_settings.agent_workspace_dir / "sample.csv").write_text(
        "captured,free\n0.8,0.2\n0.7,0.3\n",
        encoding="utf-8",
    )

    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "CSV 내용을 분석하고 결과를 텍스트 파일로 저장해줘.",
            "target_path": "sample.csv",
            "output_path": "results/report.txt",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201
    task = planned.json()
    assert task["requires_approval"] is True
    assert "run_python" in task["required_tools"]

    denied = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": False},
    )
    assert denied.status_code == 409

    executed = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    assert executed.status_code == 200
    result = wait_for_task(client, task["task_id"])
    assert result["status"] == "completed"
    assert "results/report.txt" in result["created_files"]
    assert (agent_settings.agent_workspace_dir / "results/report.txt").is_file()
    assert (agent_settings.agent_runs_dir / f"{task['task_id']}.json").is_file()


def test_edit_creates_backup_after_approval(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    target = agent_settings.agent_workspace_dir / "plot.py"
    target.write_text('title = "Old title"\n', encoding="utf-8")

    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "분석 코드의 그래프 제목을 변경해줘.",
            "target_path": "plot.py",
            "old_text": "Old title",
            "new_text": "CO2 Storage Ratio",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201
    task = planned.json()
    edit_action = next(
        action for action in task["actions"] if action["tool"] == "edit_file"
    )
    assert "Old title" in edit_action["preview"]
    assert "CO2 Storage Ratio" in edit_action["preview"]

    executed = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    assert executed.status_code == 200
    result = wait_for_task(client, task["task_id"])
    assert result["status"] == "completed"
    assert target.read_text(encoding="utf-8") == 'title = "CO2 Storage Ratio"\n'
    assert result["backups"]
    backup = agent_settings.data_dir / result["backups"][0]
    assert backup.is_file()
    assert "Old title" in backup.read_text(encoding="utf-8")


def test_dangerous_python_import_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/agent/plan",
        json={
            "request": "시스템 명령을 실행해줘",
            "python_code": "import os\nos.system('echo no')",
            "permission_level": 3,
        },
    )
    assert response.status_code == 400
    assert "Blocked Python import" in response.json()["detail"]
