from __future__ import annotations

import csv
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.agents.agent_service import (
    AgentApprovalRequired,
    AgentPermissionError,
    AgentSecurityError,
    AgentService,
    AgentTaskConflict,
    AgentTaskNotFound,
)
from app.core.config import Settings, get_settings
from app.models.agent_schemas import (
    AgentCsvColumnsResponse,
    AgentExecuteRequest,
    AgentPlanRequest,
    AgentTaskResponse,
    AgentWorkspaceResponse,
)


router = APIRouter(prefix="/agent", tags=["agent"])

_PREVIEW_EXTENSIONS = {".png", ".txt", ".md", ".csv"}
_MAX_TEXT_PREVIEW_BYTES = 1_000_000
_MAX_CSV_PREVIEW_ROWS = 100


def get_agent_service(
    settings: Settings = Depends(get_settings),
) -> AgentService:
    return AgentService(settings)


@router.post("/plan", response_model=AgentTaskResponse, status_code=201)
def create_agent_plan(
    request: AgentPlanRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentTaskResponse:
    try:
        return service.create_plan(request)
    except (AgentSecurityError, AgentPermissionError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/tasks/{task_id}/execute",
    response_model=AgentTaskResponse,
)
def execute_agent_task(
    task_id: str,
    request: AgentExecuteRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentTaskResponse:
    try:
        return service.execute_task(task_id, approved=request.approved)
    except AgentTaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Agent task not found") from exc
    except AgentApprovalRequired as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AgentTaskConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/tasks/{task_id}", response_model=AgentTaskResponse)
def get_agent_task(
    task_id: str,
    service: AgentService = Depends(get_agent_service),
) -> AgentTaskResponse:
    try:
        return service.get_task(task_id)
    except AgentTaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Agent task not found") from exc


@router.post("/tasks/{task_id}/cancel", response_model=AgentTaskResponse)
def cancel_agent_task(
    task_id: str,
    service: AgentService = Depends(get_agent_service),
) -> AgentTaskResponse:
    try:
        return service.cancel_task(task_id)
    except AgentTaskNotFound as exc:
        raise HTTPException(status_code=404, detail="Agent task not found") from exc
    except AgentTaskConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/workspace", response_model=AgentWorkspaceResponse)
def list_agent_workspace(
    path: str = Query(default=".", max_length=500),
    service: AgentService = Depends(get_agent_service),
) -> AgentWorkspaceResponse:
    try:
        return AgentWorkspaceResponse.model_validate(service.list_workspace(path))
    except (AgentSecurityError, FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/csv-columns", response_model=AgentCsvColumnsResponse)
def get_agent_csv_columns(
    path: str = Query(min_length=1, max_length=500),
    service: AgentService = Depends(get_agent_service),
) -> AgentCsvColumnsResponse:
    try:
        return AgentCsvColumnsResponse.model_validate(service.get_csv_columns(path))
    except (AgentSecurityError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_preview_file(service: AgentService, path: str) -> Path:
    resolved = service.permissions.resolve_path(
        path,
        must_exist=True,
        allow_directory=False,
        allowed_extensions=_PREVIEW_EXTENSIONS,
    )
    if resolved.stat().st_size > _MAX_TEXT_PREVIEW_BYTES and resolved.suffix.lower() != ".png":
        raise ValueError("미리보기 가능한 파일 크기(1 MB)를 초과했습니다.")
    return resolved


@router.get("/files/preview")
def preview_agent_file(
    path: str = Query(min_length=1, max_length=500),
    service: AgentService = Depends(get_agent_service),
):
    try:
        resolved = _resolve_preview_file(service, path)
        relative = service.permissions.to_relative(resolved)
        suffix = resolved.suffix.lower()
        if suffix == ".png":
            return {"path": relative, "kind": "image"}
        if suffix == ".csv":
            with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                rows = []
                truncated = False
                for index, row in enumerate(reader):
                    if index > _MAX_CSV_PREVIEW_ROWS:
                        truncated = True
                        break
                    rows.append(row)
            return {
                "path": relative,
                "kind": "csv",
                "columns": rows[0] if rows else [],
                "rows": rows[1:] if rows else [],
                "truncated": truncated,
            }
        return {"path": relative, "kind": "text", "content": resolved.read_text(encoding="utf-8")}
    except (AgentSecurityError, FileNotFoundError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/files/content")
def get_agent_file_content(
    path: str = Query(min_length=1, max_length=500),
    download: bool = Query(default=False),
    service: AgentService = Depends(get_agent_service),
):
    try:
        resolved = service.permissions.resolve_path(
            path, must_exist=True, allow_directory=False, allowed_extensions=_PREVIEW_EXTENSIONS
        )
        return FileResponse(
            resolved,
            filename=resolved.name,
            content_disposition_type="attachment" if download else "inline",
        )
    except (AgentSecurityError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
