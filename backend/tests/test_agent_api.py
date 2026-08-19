from __future__ import annotations

from pathlib import Path
import os
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
        agent_python_timeout_seconds=30,
    )


@pytest.fixture()
def client(agent_settings: Settings):
    service = AgentService(agent_settings)
    app.dependency_overrides[get_agent_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def wait_for_task(client: TestClient, task_id: str) -> dict:
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        response = client.get(f"/api/agent/tasks/{task_id}")
        assert response.status_code == 200
        task = response.json()
        if task["status"] in {"completed", "failed", "canceled"}:
            return task
        time.sleep(0.05)
    raise AssertionError("Agent task did not finish in time")


def write_co2_csv(agent_settings: Settings) -> Path:
    target = agent_settings.agent_workspace_dir / "co2_result.csv"
    target.write_text(
        "srco2,trapped_ratio,free_ratio,total_storage_mt\n"
        "0.10,94.9,5.1,5.2\n"
        "0.20,88.7,11.3,5.2\n"
        "0.30,81.2,18.8,5.2\n"
        "0.40,71.8,28.2,5.2\n",
        encoding="utf-8",
    )
    return target


def test_agent_routes_are_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/api/agent/plan" in paths
    assert "/api/agent/tasks/{task_id}/execute" in paths
    assert "/api/agent/tasks/{task_id}" in paths
    assert "/api/agent/tasks/{task_id}/cancel" in paths
    assert "/api/agent/workspace" in paths
    assert "/api/agent/csv-columns" in paths
    assert "/api/agent/files/preview" in paths
    assert "/api/agent/files/content" in paths
    assert "/api/agent/runs" in paths


def test_agent_run_history_is_newest_first_and_skips_invalid_files(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    first = client.post(
        "/api/agent/plan",
        json={"request": "workspace 목록을 확인해줘.", "permission_level": 1},
    ).json()
    second = client.post(
        "/api/agent/plan",
        json={"request": "다시 workspace 목록을 확인해줘.", "permission_level": 1},
    ).json()
    invalid = agent_settings.agent_runs_dir / "AT-BROKEN.json"
    invalid.write_text("not-json", encoding="utf-8")
    future = time.time() + 10
    invalid.touch()
    os.utime(invalid, (future, future))

    response = client.get("/api/agent/runs", params={"limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert [run["task_id"] for run in payload["runs"]][:2] == [
        second["task_id"],
        first["task_id"],
    ]
    assert payload["runs"][0]["request"] == "다시 workspace 목록을 확인해줘."
    assert payload["skipped_files"] == 1


def test_agent_run_history_respects_limit(client: TestClient) -> None:
    for index in range(3):
        response = client.post(
            "/api/agent/plan",
            json={"request": f"목록 확인 {index}", "permission_level": 1},
        )
        assert response.status_code == 201

    response = client.get("/api/agent/runs", params={"limit": 2})
    assert response.status_code == 200
    assert len(response.json()["runs"]) == 2


def test_agent_task_is_restored_by_new_service_instance(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    planned = client.post(
        "/api/agent/plan",
        json={"request": "workspace 목록을 확인해줘.", "permission_level": 1},
    )
    assert planned.status_code == 201
    task_id = planned.json()["task_id"]

    restarted_service = AgentService(agent_settings)
    restored = restarted_service.get_task(task_id)

    assert restored.task_id == task_id
    assert restored.request == "workspace 목록을 확인해줘."
    assert restored.status == "planned"


def test_planned_task_can_be_canceled(client: TestClient) -> None:
    planned = client.post(
        "/api/agent/plan",
        json={"request": "workspace 목록을 확인해줘.", "permission_level": 1},
    )
    assert planned.status_code == 201
    task_id = planned.json()["task_id"]

    canceled = client.post(f"/api/agent/tasks/{task_id}/cancel")

    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"
    assert canceled.json()["cancel_requested"] is True

    execute = client.post(
        f"/api/agent/tasks/{task_id}/execute",
        json={"approved": True},
    )
    assert execute.status_code == 409


def test_agent_result_file_preview_and_download(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    results = agent_settings.agent_workspace_dir / "results"
    results.mkdir()
    (results / "report.txt").write_text("analysis complete", encoding="utf-8")
    (results / "data.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")

    text_response = client.get(
        "/api/agent/files/preview", params={"path": "results/report.txt"}
    )
    assert text_response.status_code == 200
    assert text_response.json() == {
        "path": "results/report.txt", "kind": "text", "content": "analysis complete"
    }

    csv_response = client.get(
        "/api/agent/files/preview", params={"path": "results/data.csv"}
    )
    assert csv_response.status_code == 200
    assert csv_response.json()["columns"] == ["x", "y"]
    assert csv_response.json()["rows"] == [["1", "2"], ["3", "4"]]

    download = client.get(
        "/api/agent/files/content",
        params={"path": "results/report.txt", "download": "true"},
    )
    assert download.status_code == 200
    assert download.content == b"analysis complete"
    assert "attachment" in download.headers["content-disposition"]


@pytest.mark.parametrize("path", ["../.env", r"C:\\Windows\\win.ini", ".env"])
def test_agent_result_file_endpoints_reject_unsafe_paths(
    client: TestClient, path: str
) -> None:
    assert client.get("/api/agent/files/preview", params={"path": path}).status_code == 400
    assert client.get("/api/agent/files/content", params={"path": path}).status_code == 400


@pytest.mark.parametrize(
    ("request_text", "target_path", "expected_message"),
    [
        (
            r"C:\Windows\System32\drivers\etc\hosts 파일을 읽어줘.",
            None,
            "workspace",
        ),
        ("../.env 파일을 읽어줘.", None, "workspace"),
        (".env 파일을 읽어줘.", None, "비밀정보"),
        ("https://example.com/data.csv 파일을 읽어줘.", None, "URL"),
        ("시스템 파일을 읽어줘.", r"C:\Windows\win.ini", "workspace"),
    ],
)
def test_unsafe_path_requests_are_recorded_as_failed(
    client: TestClient,
    agent_settings: Settings,
    request_text: str,
    target_path: str | None,
    expected_message: str,
) -> None:
    payload = {
        "request": request_text,
        "permission_level": 1,
    }
    if target_path is not None:
        payload["target_path"] = target_path

    response = client.post("/api/agent/plan", json=payload)

    assert response.status_code == 201
    task = response.json()
    assert task["status"] == "failed"
    assert task["required_tools"] == []
    assert task["actions"] == []
    assert expected_message in task["error"]
    assert task["read_files"] == []
    assert task["created_files"] == []
    assert task["modified_files"] == []
    assert (agent_settings.agent_runs_dir / f"{task['task_id']}.json").is_file()


def test_rejected_task_cannot_be_executed(client: TestClient) -> None:
    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "../secret.txt 파일을 읽어줘.",
            "permission_level": 1,
        },
    )
    assert planned.status_code == 201
    task = planned.json()
    assert task["status"] == "failed"

    executed = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    assert executed.status_code == 409


@pytest.mark.parametrize(
    ("request_text", "expected_message"),
    [
        ("workspace의 result.csv 파일을 삭제해줘.", "삭제"),
        ("PowerShell 명령을 실행해줘.", "shell"),
    ],
)
def test_destructive_or_shell_requests_are_rejected_before_planning(
    client: TestClient,
    request_text: str,
    expected_message: str,
) -> None:
    response = client.post(
        "/api/agent/plan",
        json={"request": request_text, "permission_level": 3},
    )

    assert response.status_code == 201
    task = response.json()
    assert task["status"] == "failed"
    assert task["actions"] == []
    assert task["required_tools"] == []
    assert expected_message in task["error"]


def test_safe_directory_request_still_uses_list_directory(client: TestClient) -> None:
    response = client.post(
        "/api/agent/plan",
        json={
            "request": "workspace 폴더 목록을 확인해줘.",
            "permission_level": 1,
        },
    )
    assert response.status_code == 201
    task = response.json()
    assert task["status"] == "planned"
    assert task["required_tools"] == ["list_directory"]


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


def test_csv_columns_endpoint(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.get(
        "/api/agent/csv-columns",
        params={"path": "co2_result.csv"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "path": "co2_result.csv",
        "columns": ["srco2", "trapped_ratio", "free_ratio", "total_storage_mt"],
    }


def test_scatter_plan_uses_explicit_columns(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "co2_result.csv로 산점도를 만들어줘.",
            "target_path": "co2_result.csv",
            "output_path": "results/selected.png",
            "x_column": "trapped_ratio",
            "y_column": "free_ratio",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201
    task = response.json()
    run_action = next(
        action for action in task["actions"] if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["x_column"] == "trapped_ratio"
    assert run_action["arguments"]["y_column"] == "free_ratio"
    assert "x_name = 'trapped_ratio'" in run_action["preview"]
    assert "y_name = 'free_ratio'" in run_action["preview"]
    assert "X축은 trapped_ratio, Y축은 free_ratio" in " ".join(task["plan"])


def test_scatter_plan_infers_korean_column_aliases(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": (
                "co2_result.csv에서 포획된 CO₂ 비율과 자유 상태 CO₂ 비율의 "
                "관계를 산점도로 만들어줘."
            ),
            "target_path": "co2_result.csv",
            "output_path": "results/inferred.png",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201
    task = response.json()
    run_action = next(
        action for action in task["actions"] if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["x_column"] == "trapped_ratio"
    assert run_action["arguments"]["y_column"] == "free_ratio"


def test_scatter_plan_rejects_unknown_column(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "co2_result.csv로 산점도를 만들어줘.",
            "target_path": "co2_result.csv",
            "x_column": "missing_column",
            "y_column": "free_ratio",
            "permission_level": 3,
        },
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "missing_column" in detail
    assert "trapped_ratio" in detail


def test_scatter_executes_with_selected_columns(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "co2_result.csv로 산점도를 만들어줘.",
            "target_path": "co2_result.csv",
            "output_path": "results/trapped_vs_free.png",
            "x_column": "trapped_ratio",
            "y_column": "free_ratio",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201
    task = planned.json()

    executed = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    assert executed.status_code == 200
    result = wait_for_task(client, task["task_id"])

    assert result["status"] == "completed", result.get("error")
    assert "results/trapped_vs_free.png" in result["created_files"]
    output = agent_settings.agent_workspace_dir / "results/trapped_vs_free.png"
    assert output.is_file()
    assert output.stat().st_size > 0


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
