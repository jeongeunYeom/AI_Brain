from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

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
