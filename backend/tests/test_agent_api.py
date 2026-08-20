from __future__ import annotations

import csv
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


def write_comparison_csvs(agent_settings: Settings) -> tuple[Path, Path]:
    first = agent_settings.agent_workspace_dir / "srco2_0.10.csv"
    second = agent_settings.agent_workspace_dir / "srco2_0.20.csv"
    first.write_text(
        "step,trapped_ratio,free_ratio\n"
        "1,94.0,6.0\n"
        "2,95.0,5.0\n",
        encoding="utf-8",
    )
    second.write_text(
        "step,trapped_ratio,free_ratio\n"
        "1,88.0,12.0\n"
        "2,90.0,10.0\n",
        encoding="utf-8",
    )
    return first, second


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
    assert "/api/agent/conversations" in paths
    assert "/api/agent/conversations/{conversation_id}" in paths


def test_conversation_groups_multiple_tasks_and_restores_them(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    created = client.post(
        "/api/agent/conversations",
        json={"title": "CO2 결과 분석"},
    )
    assert created.status_code == 201
    conversation_id = created.json()["conversation_id"]

    first = client.post(
        "/api/agent/plan",
        json={
            "request": "workspace 목록을 확인해줘.",
            "conversation_id": conversation_id,
            "permission_level": 1,
        },
    )
    second = client.post(
        "/api/agent/plan",
        json={
            "request": "같은 대화에서 다시 목록을 확인해줘.",
            "conversation_id": conversation_id,
            "permission_level": 1,
        },
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["conversation_id"] == conversation_id
    assert second.json()["conversation_id"] == conversation_id

    restarted_service = AgentService(agent_settings)
    restored = restarted_service.get_conversation(conversation_id)

    assert restored.title == "CO2 결과 분석"
    assert restored.task_count == 2
    assert [task.task_id for task in restored.tasks] == [
        first.json()["task_id"],
        second.json()["task_id"],
    ]
    assert [task.request for task in restored.tasks] == [
        "workspace 목록을 확인해줘.",
        "같은 대화에서 다시 목록을 확인해줘.",
    ]


def test_plan_without_conversation_creates_one_automatically(client: TestClient) -> None:
    planned = client.post(
        "/api/agent/plan",
        json={"request": "새 자동 대화를 시작해줘.", "permission_level": 1},
    )
    assert planned.status_code == 201
    conversation_id = planned.json()["conversation_id"]
    assert conversation_id.startswith("CV-")

    detail = client.get(f"/api/agent/conversations/{conversation_id}")
    assert detail.status_code == 200
    assert detail.json()["task_count"] == 1
    assert detail.json()["title"] == "새 자동 대화를 시작해줘."


def test_conversation_list_is_newest_first(client: TestClient) -> None:
    first = client.post(
        "/api/agent/conversations",
        json={"title": "첫 번째 대화"},
    ).json()
    second = client.post(
        "/api/agent/conversations",
        json={"title": "두 번째 대화"},
    ).json()

    response = client.get("/api/agent/conversations", params={"limit": 2})

    assert response.status_code == 200
    assert [item["conversation_id"] for item in response.json()["conversations"]] == [
        second["conversation_id"],
        first["conversation_id"],
    ]


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


def test_multi_csv_request_rejects_unsafe_target_path(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    response = client.post(
        "/api/agent/plan",
        json={
            "request": "두 CSV를 비교해줘.",
            "target_paths": ["safe.csv", "../outside.csv"],
            "permission_level": 3,
        },
    )

    assert response.status_code == 201
    task = response.json()
    assert task["status"] == "failed"
    assert "workspace" in task["error"]
    assert task["actions"] == []
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


def test_non_csv_path_in_request_still_uses_file_reader(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    (agent_settings.agent_workspace_dir / "notes.txt").write_text(
        "safe notes",
        encoding="utf-8",
    )

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "notes.txt 파일을 읽어줘.",
            "permission_level": 1,
        },
    )

    assert response.status_code == 201
    task = response.json()
    assert task["required_tools"] == ["read_file"]
    assert task["actions"][0]["arguments"]["path"] == "notes.txt"


def test_knowledge_search_plan_is_read_only(client: TestClient) -> None:
    response = client.post(
        "/api/agent/plan",
        json={
            "request": "pressure buildup 관련 문헌과 이론을 찾아줘.",
            "permission_level": 1,
        },
    )

    assert response.status_code == 201
    task = response.json()
    assert task["required_tools"] == ["search_knowledge_base"]
    assert task["requires_approval"] is False
    assert task["actions"][0]["arguments"]["top_k"] == 5


def test_related_figure_search_plan_is_read_only(client: TestClient) -> None:
    response = client.post(
        "/api/agent/plan",
        json={
            "request": "wellbore storage 관련 Figure를 찾아줘.",
            "permission_level": 1,
        },
    )

    assert response.status_code == 201
    task = response.json()
    assert task["required_tools"] == ["get_related_figures"]
    assert task["requires_approval"] is False


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


def test_multi_csv_plan_uses_common_numeric_column(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_comparison_csvs(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "두 CSV의 포획 비율을 비교해줘.",
            "target_path": "srco2_0.10.csv",
            "target_paths": ["srco2_0.10.csv", "srco2_0.20.csv"],
            "compare_column": "trapped_ratio",
            "output_path": "results/srco2_comparison.csv",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201, response.text
    task = response.json()
    read_actions = [
        action for action in task["actions"] if action["tool"] == "read_file"
    ]
    run_action = next(
        action for action in task["actions"] if action["tool"] == "run_python"
    )
    assert len(read_actions) == 2
    assert run_action["arguments"]["input_paths"] == [
        "srco2_0.10.csv",
        "srco2_0.20.csv",
    ]
    assert run_action["arguments"]["compare_column"] == "trapped_ratio"
    assert run_action["arguments"]["common_columns"] == [
        "step",
        "trapped_ratio",
        "free_ratio",
    ]
    assert run_action["arguments"]["expected_outputs"] == [
        "results/srco2_comparison.csv",
        "results/srco2_comparison.png",
    ]
    assert "statistics.fmean" in run_action["preview"]


def test_multi_csv_plan_infers_paths_and_korean_column_alias(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_comparison_csvs(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": (
                "srco2_0.10.csv와 srco2_0.20.csv의 "
                "포획된 CO₂ 비율을 비교해줘."
            ),
            "permission_level": 3,
        },
    )

    assert response.status_code == 201, response.text
    run_action = next(
        action
        for action in response.json()["actions"]
        if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["compare_column"] == "trapped_ratio"
    assert run_action["arguments"]["input_paths"] == [
        "srco2_0.10.csv",
        "srco2_0.20.csv",
    ]


def test_multi_csv_plan_rejects_files_without_common_numeric_column(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    (agent_settings.agent_workspace_dir / "pressure.csv").write_text(
        "pressure\n100\n",
        encoding="utf-8",
    )
    (agent_settings.agent_workspace_dir / "facies.csv").write_text(
        "facies\nsand\n",
        encoding="utf-8",
    )

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "두 CSV를 비교해줘.",
            "target_paths": ["pressure.csv", "facies.csv"],
            "permission_level": 3,
        },
    )

    assert response.status_code == 400
    assert "공통 숫자 열" in response.json()["detail"]


def test_multi_csv_comparison_executes_and_validates_outputs(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_comparison_csvs(agent_settings)

    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "조건별 trapped_ratio를 비교해줘.",
            "target_paths": ["srco2_0.10.csv", "srco2_0.20.csv"],
            "compare_column": "trapped_ratio",
            "output_path": "results/srco2_comparison.csv",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201, planned.text
    task = planned.json()

    executed = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    assert executed.status_code == 200
    result = wait_for_task(client, task["task_id"])

    assert result["status"] == "completed", result.get("error")
    assert result["validation_passed"] is True
    assert {
        "results/srco2_comparison.csv",
        "results/srco2_comparison.png",
    }.issubset(result["created_files"])
    checks = [
        check
        for record in result["validation_records"]
        for check in record["checks"]
    ]
    assert any(check["name"] == "csv_content" and check["passed"] for check in checks)
    assert any(check["name"] == "png_integrity" and check["passed"] for check in checks)

    output = agent_settings.agent_workspace_dir / "results/srco2_comparison.csv"
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["source_file"] for row in rows] == [
        "srco2_0.10.csv",
        "srco2_0.20.csv",
    ]
    assert [float(row["mean"]) for row in rows] == [94.5, 89.0]


def test_mrst_co2_plan_maps_domain_columns_and_three_outputs(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "MRST CO2 저장 결과를 분석해줘.",
            "target_path": "co2_result.csv",
            "analysis_profile": "mrst_co2",
            "output_path": "results/storage_report.md",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201, response.text
    task = response.json()
    run_action = next(
        action for action in task["actions"] if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["analysis_profile"] == "mrst_co2"
    assert run_action["arguments"]["column_maps"] == {
        "co2_result.csv": {
            "srco2": "srco2",
            "trapped_ratio": "trapped_ratio",
            "free_ratio": "free_ratio",
            "total_storage": "total_storage_mt",
        }
    }
    assert run_action["arguments"]["expected_outputs"] == [
        "results/storage_report.csv",
        "results/storage_report.png",
        "results/storage_report.md",
    ]
    assert "원본 CSV를 변경하지 않고" in " ".join(task["plan"])


def test_mrst_co2_plan_is_inferred_from_domain_analysis_request(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "co2_result.csv의 잔류 포화도별 포획 비율 경향을 분석해줘.",
            "target_path": "co2_result.csv",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201, response.text
    run_action = next(
        action
        for action in response.json()["actions"]
        if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["analysis_profile"] == "mrst_co2"


def test_mrst_co2_direct_ratios_execute_and_validate_all_outputs(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)
    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "MRST CO2 결과를 전용 분석해줘.",
            "target_path": "co2_result.csv",
            "analysis_profile": "mrst_co2",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201, planned.text
    task = planned.json()

    executed = client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    assert executed.status_code == 200
    result = wait_for_task(client, task["task_id"])

    assert result["status"] == "completed", result.get("error")
    assert result["validation_passed"] is True
    expected = {
        "results/mrst_co2_analysis.csv",
        "results/mrst_co2_analysis.png",
        "results/mrst_co2_analysis.md",
    }
    assert expected.issubset(result["created_files"])
    summary = agent_settings.agent_workspace_dir / "results/mrst_co2_analysis.csv"
    with summary.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["srco2"]) == pytest.approx(0.1)
    assert float(rows[0]["trapped_ratio_pct"]) == pytest.approx(94.9)
    assert float(rows[-1]["free_ratio_pct"]) == pytest.approx(28.2)
    assert rows[0]["storage_unit"] == "Mt"
    assert rows[0]["calculation_basis"] == "direct_ratio"
    report = (
        agent_settings.agent_workspace_dir / "results/mrst_co2_analysis.md"
    ).read_text(encoding="utf-8")
    assert "Pearson r=" in report
    assert "does not establish causality" in report


def test_mrst_co2_derives_ratios_from_amount_columns(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    (agent_settings.agent_workspace_dir / "mrst_amounts.csv").write_text(
        "time_years,trapped_co2_mt,free_co2_mt,total_storage_mt\n"
        "100,4,1,5\n"
        "200,4.5,0.5,5\n",
        encoding="utf-8",
    )
    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "MRST 저장량 결과를 분석해줘.",
            "target_path": "mrst_amounts.csv",
            "analysis_profile": "mrst_co2",
            "output_path": "results/amount_analysis.md",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201, planned.text
    task = planned.json()
    client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    result = wait_for_task(client, task["task_id"])
    assert result["status"] == "completed", result.get("error")

    summary = agent_settings.agent_workspace_dir / "results/amount_analysis.csv"
    with summary.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["trapped_ratio_pct"]) for row in rows] == [80.0, 90.0]
    assert [float(row["free_ratio_pct"]) for row in rows] == [20.0, 10.0]
    assert all(
        row["calculation_basis"] == "derived_from_trapped_and_free_amounts"
        for row in rows
    )


def test_mrst_co2_normalizes_fraction_ratios_to_percent(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    (agent_settings.agent_workspace_dir / "fraction_ratios.csv").write_text(
        "srco2,trapped_ratio,free_ratio\n0.1,0.8,0.2\n",
        encoding="utf-8",
    )
    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "MRST CO2 비율을 분석해줘.",
            "target_path": "fraction_ratios.csv",
            "analysis_profile": "mrst_co2",
            "output_path": "results/fractions.md",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201, planned.text
    task = planned.json()
    client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    result = wait_for_task(client, task["task_id"])
    assert result["status"] == "completed", result.get("error")

    with (
        agent_settings.agent_workspace_dir / "results/fractions.csv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["trapped_ratio_pct"]) == 80.0
    assert float(row["free_ratio_pct"]) == 20.0


def test_mrst_co2_multiple_files_infer_srco2_from_filenames(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    for filename, trapped in (("srco2_0.10.csv", 95), ("srco2_0.20.csv", 89)):
        (agent_settings.agent_workspace_dir / filename).write_text(
            f"trapped_ratio,free_ratio\n{trapped},{100 - trapped}\n",
            encoding="utf-8",
        )
    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "MRST CO2 조건별 결과를 분석해줘.",
            "target_paths": ["srco2_0.10.csv", "srco2_0.20.csv"],
            "analysis_profile": "mrst_co2",
            "output_path": "results/conditions.md",
            "permission_level": 3,
        },
    )
    assert planned.status_code == 201, planned.text
    task = planned.json()
    client.post(
        f"/api/agent/tasks/{task['task_id']}/execute",
        json={"approved": True},
    )
    result = wait_for_task(client, task["task_id"])
    assert result["status"] == "completed", result.get("error")

    with (
        agent_settings.agent_workspace_dir / "results/conditions.csv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [float(row["srco2"]) for row in rows] == [0.1, 0.2]


def test_mrst_co2_plan_rejects_csv_without_ratio_or_amount_columns(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    (agent_settings.agent_workspace_dir / "pressure.csv").write_text(
        "time_years,pressure_bar\n100,250\n",
        encoding="utf-8",
    )
    response = client.post(
        "/api/agent/plan",
        json={
            "request": "MRST CO2 결과를 분석해줘.",
            "target_path": "pressure.csv",
            "analysis_profile": "mrst_co2",
            "permission_level": 3,
        },
    )
    assert response.status_code == 400
    assert "trapped/free" in response.json()["detail"]


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


def test_line_chart_plan_uses_explicit_chart_type(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "시간 순서처럼 값의 변화를 그래프로 보여줘.",
            "target_path": "co2_result.csv",
            "output_path": "results/trapped_line.png",
            "chart_type": "line",
            "x_column": "srco2",
            "y_column": "trapped_ratio",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201
    task = response.json()
    run_action = next(
        action for action in task["actions"] if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["chart_type"] == "line"
    assert run_action["arguments"]["x_column"] == "srco2"
    assert run_action["arguments"]["y_column"] == "trapped_ratio"
    assert "plt.plot" in run_action["preview"]


def test_bar_chart_plan_is_inferred_from_korean_request(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": (
                "co2_result.csv에서 srco2와 trapped_ratio를 "
                "막대그래프로 만들어줘."
            ),
            "target_path": "co2_result.csv",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201
    run_action = next(
        action
        for action in response.json()["actions"]
        if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["chart_type"] == "bar"
    assert run_action["arguments"]["x_column"] == "srco2"
    assert run_action["arguments"]["y_column"] == "trapped_ratio"
    assert "plt.bar" in run_action["preview"]


def test_histogram_plan_uses_one_numeric_column(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    response = client.post(
        "/api/agent/plan",
        json={
            "request": "co2_result.csv의 포획 비율 분포를 보여줘.",
            "target_path": "co2_result.csv",
            "output_path": "results/trapped_histogram.png",
            "chart_type": "histogram",
            "x_column": "trapped_ratio",
            "permission_level": 3,
        },
    )

    assert response.status_code == 201
    run_action = next(
        action
        for action in response.json()["actions"]
        if action["tool"] == "run_python"
    )
    assert run_action["arguments"]["chart_type"] == "histogram"
    assert run_action["arguments"]["x_column"] == "trapped_ratio"
    assert "y_column" not in run_action["arguments"]
    assert "plt.hist" in run_action["preview"]


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
    assert result["validation_passed"] is True
    assert any(
        check["name"] == "png_integrity" and check["passed"]
        for record in result["validation_records"]
        for check in record["checks"]
    )
    assert "results/trapped_vs_free.png" in result["created_files"]
    output = agent_settings.agent_workspace_dir / "results/trapped_vs_free.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_histogram_executes_and_creates_valid_png(
    client: TestClient,
    agent_settings: Settings,
) -> None:
    write_co2_csv(agent_settings)

    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "co2_result.csv의 trapped_ratio 히스토그램을 만들어줘.",
            "target_path": "co2_result.csv",
            "output_path": "results/trapped_histogram.png",
            "chart_type": "histogram",
            "x_column": "trapped_ratio",
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
    assert result["validation_passed"] is True
    assert any(
        check["name"] == "png_integrity" and check["passed"]
        for record in result["validation_records"]
        for check in record["checks"]
    )
    assert "results/trapped_histogram.png" in result["created_files"]


@pytest.mark.parametrize("chart_type", ["line", "bar"])
def test_two_axis_chart_types_execute_and_create_valid_png(
    client: TestClient,
    agent_settings: Settings,
    chart_type: str,
) -> None:
    write_co2_csv(agent_settings)
    output_path = f"results/trapped_{chart_type}.png"

    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "srco2에 따른 trapped_ratio 변화를 보여줘.",
            "target_path": "co2_result.csv",
            "output_path": output_path,
            "chart_type": chart_type,
            "x_column": "srco2",
            "y_column": "trapped_ratio",
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
    assert result["validation_passed"] is True
    assert output_path in result["created_files"]
    assert (agent_settings.agent_workspace_dir / output_path).stat().st_size > 0


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
    assert result["validation_passed"] is True
    assert any(
        check["name"] == "text_content" and check["passed"]
        for record in result["validation_records"]
        for check in record["checks"]
    )
    assert "results/report.txt" in result["created_files"]
    report_path = agent_settings.agent_workspace_dir / "results/report.txt"
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "- 중앙값:" in report
    assert "- 표준편차(표본):" in report
    assert "[Pearson 상관계수]" in report
    assert "captured ↔ free: -1" in report
    assert (agent_settings.agent_runs_dir / f"{task['task_id']}.json").is_file()

    restored = AgentService(agent_settings).get_task(task["task_id"])
    assert restored.validation_passed is True
    assert restored.validation_records


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


def test_python_error_fails_result_validation(client: TestClient) -> None:
    planned = client.post(
        "/api/agent/plan",
        json={
            "request": "오류가 발생하는 Python 코드를 실행해줘.",
            "python_code": "raise ValueError('intentional failure')",
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

    assert result["status"] == "failed"
    assert result["validation_passed"] is False
    assert result["validation_records"]
    assert any(
        check["name"] == "python_execution" and not check["passed"]
        for record in result["validation_records"]
        for check in record["checks"]
    )
    assert "결과 자동 검증 실패" in result["error"]
