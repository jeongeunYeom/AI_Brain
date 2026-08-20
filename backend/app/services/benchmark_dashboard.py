from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings


_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")


class BenchmarkRunNotFound(FileNotFoundError):
    pass


class BenchmarkDashboardService:
    """Read-only access to benchmark result JSON files."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        runs: list[dict[str, Any]] = []

        for path in self._run_paths():
            try:
                payload = self._read_json(path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            runs.append(self._run_summary(payload, path))

        runs.sort(
            key=lambda item: (
                str(item.get("created_at") or ""),
                str(item.get("run_id") or ""),
            ),
            reverse=True,
        )
        return runs[:safe_limit]

    def get_latest(self) -> dict[str, Any]:
        latest_path = (
            self.settings.evaluation_dir
            / "well_test_benchmark_latest.json"
        )
        if latest_path.exists():
            return self._sanitize_payload(
                self._read_json(latest_path)
            )

        paths = self._run_paths()
        if not paths:
            raise BenchmarkRunNotFound(
                "저장된 benchmark 실행 결과가 없습니다."
            )

        newest = max(
            paths,
            key=lambda path: path.stat().st_mtime,
        )
        return self._sanitize_payload(self._read_json(newest))

    def get_run(self, run_id: str) -> dict[str, Any]:
        normalized = str(run_id or "").strip()
        if not _RUN_ID_RE.fullmatch(normalized):
            raise BenchmarkRunNotFound(
                "유효하지 않은 benchmark run ID입니다."
            )

        path = (
            self.settings.evaluation_dir
            / f"well_test_benchmark_{normalized}.json"
        )
        if not path.exists():
            raise BenchmarkRunNotFound(
                f"Benchmark run을 찾지 못했습니다: {normalized}"
            )

        return self._sanitize_payload(self._read_json(path))

    def _run_paths(self) -> list[Path]:
        return [
            path
            for path in self.settings.evaluation_dir.glob(
                "well_test_benchmark_*.json"
            )
            if path.name != "well_test_benchmark_latest.json"
        ]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                "Benchmark JSON root must be an object."
            )
        return payload

    def _run_summary(
        self,
        payload: dict[str, Any],
        path: Path,
    ) -> dict[str, Any]:
        run_id = str(payload.get("run_id") or "")
        if not _RUN_ID_RE.fullmatch(run_id):
            match = re.search(
                r"well_test_benchmark_(\d{8}T\d{6}Z)\.json$",
                path.name,
            )
            run_id = match.group(1) if match else path.stem

        return {
            "run_id": run_id,
            "created_at": payload.get("created_at"),
            "condition": payload.get("condition"),
            "mode": payload.get("mode"),
            "model": payload.get("model"),
            "question_count": payload.get(
                "question_count",
                payload.get("summary", {}).get("questions_total"),
            ),
            "wall_clock_seconds": payload.get(
                "wall_clock_seconds"
            ),
            "summary": payload.get("summary") or {},
        }

    def _sanitize_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        sanitized = {
            key: value
            for key, value in payload.items()
            if key != "results"
        }
        sanitized["results"] = [
            self._sanitize_result(result)
            for result in payload.get("results") or []
            if isinstance(result, dict)
        ]
        return sanitized

    def _sanitize_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        sanitized = {
            key: value
            for key, value in result.items()
            if key not in {"sources", "agent_run_file"}
        }

        run_file_value = str(
            result.get("agent_run_file") or ""
        ).strip()
        sanitized["attempt_details"] = []

        if not run_file_value:
            return sanitized

        run_file = (
            self.settings.agent_runs_dir
            / Path(run_file_value).name
        )
        if not run_file.exists():
            return sanitized

        try:
            record = self._read_json(run_file)
        except (OSError, json.JSONDecodeError, ValueError):
            return sanitized

        sanitized["attempt_details"] = [
            {
                "attempt": attempt.get("attempt"),
                "answer": attempt.get("answer"),
                "elapsed_seconds": attempt.get(
                    "elapsed_seconds"
                ),
                "validation_passed": attempt.get(
                    "validation_passed"
                ),
                "errors": attempt.get("errors") or [],
                "warnings": attempt.get("warnings") or [],
                "rule_ids": attempt.get("rule_ids") or [],
            }
            for attempt in record.get("attempts") or []
            if isinstance(attempt, dict)
        ]
        return sanitized
