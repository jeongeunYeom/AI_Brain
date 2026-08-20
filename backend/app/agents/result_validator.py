from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.agents.permission_manager import PermissionManager
from app.core.config import Settings
from app.models.agent_schemas import AgentAction, AgentToolName


class AgentResultValidationError(RuntimeError):
    pass


class AgentResultValidator:
    """Validate execution results before an Agent task is completed."""

    def __init__(self, settings: Settings, permissions: PermissionManager):
        self.settings = settings
        self.permissions = permissions

    def validate_action(
        self,
        action: AgentAction,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        if action.tool == AgentToolName.RUN_PYTHON:
            return self._validate_python(action, result)
        if action.tool == AgentToolName.CREATE_FILE:
            return self._validate_workspace_file(
                action,
                result,
                expected_path=result.get("path") or action.arguments.get("path"),
                require_nonempty=True,
            )
        if action.tool == AgentToolName.EDIT_FILE:
            return self._validate_edit(action, result)
        return None

    def execution_failure(
        self,
        action: AgentAction,
        error: Exception,
    ) -> dict[str, Any]:
        message = str(error) or type(error).__name__
        return {
            "action_id": action.action_id,
            "tool": action.tool.value,
            "passed": False,
            "checks": [
                self._check(
                    "tool_execution",
                    False,
                    f"도구 실행 오류: {message}",
                )
            ],
            "errors": [message],
        }

    def _validate_python(
        self,
        action: AgentAction,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        errors: list[str] = []

        success = bool(result.get("success"))
        exit_code = result.get("exit_code")
        timed_out = bool(result.get("timed_out"))
        self._add_check(
            checks,
            errors,
            "python_execution",
            success and exit_code == 0,
            f"Python 종료코드: {exit_code}",
        )
        self._add_check(
            checks,
            errors,
            "python_timeout",
            not timed_out,
            "Python timeout 없음" if not timed_out else "Python 실행 timeout 발생",
        )

        produced_files = {
            str(path)
            for path in (
                list(result.get("created_files") or [])
                + list(result.get("modified_files") or [])
            )
        }
        expected_outputs = action.arguments.get("expected_outputs") or []
        for raw_path in expected_outputs:
            expected_path = str(raw_path)
            self._add_check(
                checks,
                errors,
                "expected_output_recorded",
                expected_path in produced_files,
                (
                    f"생성·수정 기록 확인: {expected_path}"
                    if expected_path in produced_files
                    else f"요청한 결과물이 생성·수정 기록에 없음: {expected_path}"
                ),
                path=expected_path,
            )
            file_checks, file_errors = self._inspect_workspace_file(
                expected_path,
                require_nonempty=True,
            )
            checks.extend(file_checks)
            errors.extend(file_errors)

        return self._record(action, checks, errors)

    def _validate_workspace_file(
        self,
        action: AgentAction,
        result: dict[str, Any],
        *,
        expected_path: object,
        require_nonempty: bool,
    ) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        errors: list[str] = []
        if not expected_path:
            self._add_check(
                checks,
                errors,
                "result_path",
                False,
                "결과 파일 경로가 반환되지 않음",
            )
            return self._record(action, checks, errors)

        file_checks, file_errors = self._inspect_workspace_file(
            str(expected_path),
            require_nonempty=require_nonempty,
        )
        checks.extend(file_checks)
        errors.extend(file_errors)
        return self._record(action, checks, errors)

    def _validate_edit(
        self,
        action: AgentAction,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        record = self._validate_workspace_file(
            action,
            result,
            expected_path=result.get("path") or action.arguments.get("path"),
            require_nonempty=False,
        )
        backup = result.get("backup")
        backup_valid = False
        if backup:
            backup_path = (self.settings.data_dir / str(backup)).resolve()
            backups_root = self.settings.agent_backups_dir.resolve()
            try:
                backup_path.relative_to(backups_root)
            except ValueError:
                backup_valid = False
            else:
                backup_valid = backup_path.is_file()

        detail = (
            f"수정 전 백업 확인: {backup}"
            if backup_valid
            else "수정 전 백업 파일을 확인할 수 없음"
        )
        check = self._check("edit_backup", backup_valid, detail, path=backup)
        record["checks"].append(check)
        if not backup_valid:
            record["errors"].append(detail)
            record["passed"] = False
        return record

    def _inspect_workspace_file(
        self,
        raw_path: str,
        *,
        require_nonempty: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        checks: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            path = self.permissions.resolve_path(
                raw_path,
                must_exist=False,
                allow_directory=False,
            )
        except (OSError, ValueError) as exc:
            self._add_check(
                checks,
                errors,
                "output_path",
                False,
                f"결과 경로 검증 실패: {exc}",
                path=raw_path,
            )
            return checks, errors

        exists = path.is_file()
        self._add_check(
            checks,
            errors,
            "file_exists",
            exists,
            f"결과 파일 존재: {raw_path}" if exists else f"결과 파일 없음: {raw_path}",
            path=raw_path,
        )
        if not exists:
            return checks, errors

        size = path.stat().st_size
        size_valid = size > 0 if require_nonempty else size >= 0
        self._add_check(
            checks,
            errors,
            "file_size",
            size_valid,
            f"결과 파일 크기: {size} bytes",
            path=raw_path,
        )
        if not size_valid:
            return checks, errors

        suffix = path.suffix.lower()
        if suffix == ".png":
            self._inspect_png(path, raw_path, checks, errors)
        elif suffix == ".csv":
            self._inspect_csv(path, raw_path, checks, errors)
        elif suffix in {".txt", ".md", ".log", ".py", ".yaml", ".yml"}:
            self._inspect_text(path, raw_path, checks, errors)
        elif suffix == ".json":
            self._inspect_json(path, raw_path, checks, errors)
        return checks, errors

    def _inspect_png(
        self,
        path: Path,
        display_path: str,
        checks: list[dict[str, Any]],
        errors: list[str],
    ) -> None:
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            valid = width > 0 and height > 0
            detail = f"PNG 열기 성공: {width}×{height}"
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            valid = False
            detail = f"PNG 파일 손상 또는 형식 오류: {exc}"
        self._add_check(
            checks,
            errors,
            "png_integrity",
            valid,
            detail,
            path=display_path,
        )

    def _inspect_csv(
        self,
        path: Path,
        display_path: str,
        checks: list[dict[str, Any]],
        errors: list[str],
    ) -> None:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
                first_row = next(reader, [])
            valid = bool(header) and bool(first_row)
            detail = (
                f"CSV 구조 확인: {len(header)}개 열, 데이터 행 존재"
                if valid
                else "CSV 헤더 또는 데이터 행이 비어 있음"
            )
        except (OSError, csv.Error, UnicodeError) as exc:
            valid = False
            detail = f"CSV 읽기 실패: {exc}"
        self._add_check(
            checks,
            errors,
            "csv_content",
            valid,
            detail,
            path=display_path,
        )

    def _inspect_text(
        self,
        path: Path,
        display_path: str,
        checks: list[dict[str, Any]],
        errors: list[str],
    ) -> None:
        try:
            valid = bool(path.read_text(encoding="utf-8-sig", errors="replace").strip())
            detail = "텍스트 내용 존재" if valid else "텍스트 내용이 비어 있음"
        except OSError as exc:
            valid = False
            detail = f"텍스트 읽기 실패: {exc}"
        self._add_check(
            checks,
            errors,
            "text_content",
            valid,
            detail,
            path=display_path,
        )

    def _inspect_json(
        self,
        path: Path,
        display_path: str,
        checks: list[dict[str, Any]],
        errors: list[str],
    ) -> None:
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
            valid = True
            detail = "JSON 형식 정상"
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            valid = False
            detail = f"JSON 형식 오류: {exc}"
        self._add_check(
            checks,
            errors,
            "json_integrity",
            valid,
            detail,
            path=display_path,
        )

    @staticmethod
    def _record(
        action: AgentAction,
        checks: list[dict[str, Any]],
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "action_id": action.action_id,
            "tool": action.tool.value,
            "passed": not errors,
            "checks": checks,
            "errors": errors,
        }

    @classmethod
    def _add_check(
        cls,
        checks: list[dict[str, Any]],
        errors: list[str],
        name: str,
        passed: bool,
        detail: str,
        *,
        path: object = None,
    ) -> None:
        checks.append(cls._check(name, passed, detail, path=path))
        if not passed:
            errors.append(detail)

    @staticmethod
    def _check(
        name: str,
        passed: bool,
        detail: str,
        *,
        path: object = None,
    ) -> dict[str, Any]:
        check = {
            "name": name,
            "passed": passed,
            "detail": detail,
        }
        if path:
            check["path"] = str(path)
        return check
