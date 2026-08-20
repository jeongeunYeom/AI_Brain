import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.benchmark_dashboard import (
    BenchmarkDashboardService,
    BenchmarkRunNotFound,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path)
    settings.evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    settings.agent_runs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return settings


def test_lists_runs_and_loads_latest(tmp_path):
    settings = make_settings(tmp_path)

    older = {
        "run_id": "20260701T010101Z",
        "created_at": "2026-07-01T01:01:01+00:00",
        "model": "qwen3:8b",
        "question_count": 16,
        "summary": {
            "final_benchmark_pass_rate": 0.9,
        },
        "results": [],
    }
    newer = {
        "run_id": "20260703T050508Z",
        "created_at": "2026-07-03T05:05:08+00:00",
        "condition": "qwen3_rag",
        "mode": "rag",
        "model": "qwen3:8b",
        "question_count": 16,
        "summary": {
            "final_benchmark_pass_rate": 1.0,
        },
        "results": [],
    }

    write_json(
        settings.evaluation_dir
        / "well_test_benchmark_20260701T010101Z.json",
        older,
    )
    write_json(
        settings.evaluation_dir
        / "well_test_benchmark_20260703T050508Z.json",
        newer,
    )
    write_json(
        settings.evaluation_dir
        / "well_test_benchmark_latest.json",
        newer,
    )

    service = BenchmarkDashboardService(settings)
    runs = service.list_runs()

    assert [run["run_id"] for run in runs] == [
        "20260703T050508Z",
        "20260701T010101Z",
    ]
    assert runs[0]["condition"] == "qwen3_rag"
    assert runs[0]["mode"] == "rag"
    assert (
        service.get_latest()["summary"][
            "final_benchmark_pass_rate"
        ]
        == 1.0
    )


def test_sanitizes_paths_and_loads_attempts(tmp_path):
    settings = make_settings(tmp_path)

    agent_run = settings.agent_runs_dir / "run.json"
    write_json(
        agent_run,
        {
            "attempts": [
                {
                    "attempt": 1,
                    "answer": "first",
                    "elapsed_seconds": 2.5,
                    "validation_passed": False,
                    "errors": ["missing value"],
                    "warnings": [],
                    "rule_ids": ["WT-RFT-002"],
                }
            ]
        },
    )

    payload = {
        "run_id": "20260703T050508Z",
        "summary": {},
        "results": [
            {
                "id": "WT-007",
                "agent_run_file": str(agent_run),
                "sources": [{"document": "secret.pdf"}],
                "final_answer": "corrected",
            }
        ],
    }

    write_json(
        settings.evaluation_dir
        / "well_test_benchmark_20260703T050508Z.json",
        payload,
    )

    result = BenchmarkDashboardService(settings).get_run(
        "20260703T050508Z"
    )["results"][0]

    assert "agent_run_file" not in result
    assert "sources" not in result
    assert result["attempt_details"][0]["rule_ids"] == [
        "WT-RFT-002"
    ]


def test_rejects_invalid_run_id(tmp_path):
    settings = make_settings(tmp_path)
    service = BenchmarkDashboardService(settings)

    with pytest.raises(BenchmarkRunNotFound):
        service.get_run("../../outside")
