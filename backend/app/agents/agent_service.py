from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock, Thread
from uuid import uuid4

from app.agents.agent_executor import AgentExecutor
from app.agents.agent_planner import AgentPlanner
from app.agents.permission_manager import (
    AgentPermissionError,
    AgentSecurityError,
    PermissionManager,
)
from app.agents.tool_registry import ToolRegistry
from app.core.config import Settings
from app.models.agent_schemas import (
    AgentPlanRequest,
    AgentTaskResponse,
    AgentTaskStatus,
)


class AgentTaskNotFound(KeyError):
    pass


class AgentTaskConflict(RuntimeError):
    pass


class AgentApprovalRequired(PermissionError):
    pass


class AgentService:
    _lock = RLock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self.permissions = PermissionManager(settings)
        self.tools = ToolRegistry(settings, self.permissions)
        self.planner = AgentPlanner(
            self.permissions,
            self.tools.file_tools,
            self.tools.python_tools,
        )
        self.executor = AgentExecutor(self.permissions, self.tools)

    def create_plan(self, request: AgentPlanRequest) -> AgentTaskResponse:
        planned = self.planner.plan(request)
        for action in planned.actions:
            self.permissions.require_tool_level(request.permission_level, action.tool)
        task_id = self._new_task_id()
        created_at = datetime.now(timezone.utc).isoformat()
        actions = [action.model_dump(mode="json") for action in planned.actions]
        task = {
            "task_id": task_id,
            "request": request.request,
            "status": AgentTaskStatus.PLANNED.value,
            "permission_level": int(request.permission_level),
            "plan": planned.plan,
            "required_tools": list(dict.fromkeys(action["tool"] for action in actions)),
            "actions": actions,
            "requires_approval": any(
                bool(action["requires_approval"]) for action in actions
            ),
            "approved": False,
            "workspace": str(self.permissions.workspace),
            "created_at": created_at,
            "started_at": None,
            "completed_at": None,
            "current_action": None,
            "progress_step": 0,
            "progress_total": len(actions),
            "tools_used": [],
            "read_files": [],
            "created_files": [],
            "modified_files": [],
            "backups": [],
            "execution_records": [],
            "results": [],
            "error": None,
            "cancel_requested": False,
        }
        self._write_task(task)
        return AgentTaskResponse.model_validate(task)

    def get_task(self, task_id: str) -> AgentTaskResponse:
        return AgentTaskResponse.model_validate(self._read_task(task_id))

    def execute_task(self, task_id: str, *, approved: bool) -> AgentTaskResponse:
        with self._lock:
            task = self._read_task(task_id)
            status = AgentTaskStatus(task["status"])
            if status == AgentTaskStatus.CANCELED:
                raise AgentTaskConflict("Canceled tasks cannot be executed.")
            if status in {AgentTaskStatus.RUNNING, AgentTaskStatus.COMPLETED}:
                raise AgentTaskConflict(f"Task is already {status.value}.")
            if task["requires_approval"] and not approved:
                raise AgentApprovalRequired(
                    "이 작업은 파일 생성·수정 또는 Python 실행을 포함하므로 승인이 필요합니다."
                )

            task["approved"] = bool(approved)
            task["status"] = AgentTaskStatus.RUNNING.value
            task["started_at"] = datetime.now(timezone.utc).isoformat()
            task["progress_step"] = 0
            task["current_action"] = "실행 준비"
            self._write_task(task)

            worker = Thread(
                target=self._execute_in_background,
                args=(task_id,),
                daemon=True,
                name=f"agent-{task_id}",
            )
            worker.start()
            return AgentTaskResponse.model_validate(task)

    def _execute_in_background(self, task_id: str) -> None:
        task = self._read_task(task_id)
        try:
            task = self.executor.execute(
                task,
                is_canceled=lambda: self._cancel_requested(task_id),
                on_progress=self._write_task,
            )
        except Exception as exc:
            task["status"] = AgentTaskStatus.FAILED.value
            task["error"] = str(exc)
            task["current_action"] = None
            task["completed_at"] = datetime.now(timezone.utc).isoformat()
        self._write_task(task)

    def cancel_task(self, task_id: str) -> AgentTaskResponse:
        with self._lock:
            task = self._read_task(task_id)
            status = AgentTaskStatus(task["status"])
            if status in {AgentTaskStatus.COMPLETED, AgentTaskStatus.FAILED}:
                raise AgentTaskConflict(f"A {status.value} task cannot be canceled.")

            task["cancel_requested"] = True
            if status == AgentTaskStatus.PLANNED:
                task["status"] = AgentTaskStatus.CANCELED.value
                task["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._write_task(task)
            return AgentTaskResponse.model_validate(task)

    def list_workspace(self, path: str = ".") -> dict:
        return self.tools.directory_tools.list_directory(path)

    def _cancel_requested(self, task_id: str) -> bool:
        try:
            return bool(self._read_task(task_id).get("cancel_requested"))
        except AgentTaskNotFound:
            return True

    def _task_path(self, task_id: str) -> Path:
        if not task_id.startswith("AT-") or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in task_id
        ):
            raise AgentTaskNotFound(task_id)
        return self.settings.agent_runs_dir / f"{task_id}.json"

    def _read_task(self, task_id: str) -> dict:
        path = self._task_path(task_id)
        with self._lock:
            if not path.is_file():
                raise AgentTaskNotFound(task_id)
            return json.loads(path.read_text(encoding="utf-8"))

    def _write_task(self, task: dict) -> None:
        path = self._task_path(task["task_id"])
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("cancel_requested"):
                    task["cancel_requested"] = True
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(task, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)

    @staticmethod
    def _new_task_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"AT-{now}-{uuid4().hex[:6].upper()}"


__all__ = [
    "AgentApprovalRequired",
    "AgentPermissionError",
    "AgentSecurityError",
    "AgentService",
    "AgentTaskConflict",
    "AgentTaskNotFound",
]
