from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.agents.permission_manager import PermissionManager
from app.agents.result_validator import AgentResultValidator
from app.core.config import Settings
from app.models.agent_schemas import AgentAction, AgentToolName


def make_validator(tmp_path: Path) -> tuple[AgentResultValidator, Settings]:
    settings = Settings(
        data_dir=tmp_path / "data",
        agent_workspace_dir=tmp_path / "workspace",
    )
    permissions = PermissionManager(settings)
    return AgentResultValidator(settings, permissions), settings


def python_action(output_path: str) -> AgentAction:
    return AgentAction(
        action_id="A-validation",
        tool=AgentToolName.RUN_PYTHON,
        description="결과 생성",
        arguments={
            "code": "print('done')",
            "expected_outputs": [output_path],
        },
    )


def python_result(output_path: str) -> dict:
    return {
        "success": True,
        "exit_code": 0,
        "timed_out": False,
        "created_files": [output_path],
        "modified_files": [],
    }


def test_valid_png_output_passes_integrity_check(tmp_path: Path) -> None:
    validator, settings = make_validator(tmp_path)
    output = settings.agent_workspace_dir / "results" / "scatter.png"
    output.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12), "white").save(output)

    record = validator.validate_action(
        python_action("results/scatter.png"),
        python_result("results/scatter.png"),
    )

    assert record is not None
    assert record["passed"] is True
    assert any(
        check["name"] == "png_integrity" and check["passed"]
        for check in record["checks"]
    )


def test_missing_expected_output_fails_validation(tmp_path: Path) -> None:
    validator, _ = make_validator(tmp_path)

    record = validator.validate_action(
        python_action("results/missing.png"),
        {
            "success": True,
            "exit_code": 0,
            "timed_out": False,
            "created_files": [],
            "modified_files": [],
        },
    )

    assert record is not None
    assert record["passed"] is False
    assert any("결과 파일 없음" in error for error in record["errors"])


def test_csv_without_data_rows_fails_validation(tmp_path: Path) -> None:
    validator, settings = make_validator(tmp_path)
    output = settings.agent_workspace_dir / "results" / "empty.csv"
    output.parent.mkdir(parents=True)
    output.write_text("pressure,rate\n", encoding="utf-8")

    record = validator.validate_action(
        python_action("results/empty.csv"),
        python_result("results/empty.csv"),
    )

    assert record is not None
    assert record["passed"] is False
    assert any(check["name"] == "csv_content" for check in record["checks"])


def test_corrupted_png_fails_validation(tmp_path: Path) -> None:
    validator, settings = make_validator(tmp_path)
    output = settings.agent_workspace_dir / "results" / "broken.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"not-a-png")

    record = validator.validate_action(
        python_action("results/broken.png"),
        python_result("results/broken.png"),
    )

    assert record is not None
    assert record["passed"] is False
    assert any(
        check["name"] == "png_integrity" and not check["passed"]
        for check in record["checks"]
    )


def test_python_timeout_fails_validation(tmp_path: Path) -> None:
    validator, _ = make_validator(tmp_path)
    action = AgentAction(
        action_id="A-timeout",
        tool=AgentToolName.RUN_PYTHON,
        description="timeout 검증",
        arguments={"code": "while True:\n    pass"},
    )

    record = validator.validate_action(
        action,
        {
            "success": False,
            "exit_code": 124,
            "timed_out": True,
            "created_files": [],
            "modified_files": [],
        },
    )

    assert record is not None
    assert record["passed"] is False
    assert any(
        check["name"] == "python_timeout" and not check["passed"]
        for check in record["checks"]
    )
