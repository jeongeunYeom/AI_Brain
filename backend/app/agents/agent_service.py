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
from app.agents.request_security import (
    AgentRequestRejected,
    validate_agent_plan_request,
)
from app.agents.tool_registry import ToolRegistry
from app.core.config import Settings
from app.models.agent_schemas import (
    AgentConversationDetail,
    AgentConversationListResponse,
    AgentConversationSummary,
    AgentPlanRequest,
    AgentTaskResponse,
    AgentTaskStatus,
)


class AgentTaskNotFound(KeyError):
    pass


class AgentConversationNotFound(KeyError):
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
        conversation_id = self._ensure_conversation(
            request.conversation_id,
            title=request.request,
        )
        try:
            validate_agent_plan_request(request)
        except AgentRequestRejected as exc:
            return self._record_rejected_plan(
                request,
                str(exc),
                conversation_id=conversation_id,
            )

        planned = self.planner.plan(request)
        for action in planned.actions:
            self.permissions.require_tool_level(request.permission_level, action.tool)
        task_id = self._new_task_id()
        created_at = datetime.now(timezone.utc).isoformat()
        actions = [action.model_dump(mode="json") for action in planned.actions]
        task = {
            "task_id": task_id,
            "conversation_id": conversation_id,
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
        self._append_task_to_conversation(conversation_id, task_id)
        return AgentTaskResponse.model_validate(task)

    def create_conversation(self, title: str | None = None) -> AgentConversationSummary:
        conversation_id = self._new_conversation_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "conversation_id": conversation_id,
            "title": self._conversation_title(title),
            "created_at": timestamp,
            "updated_at": timestamp,
            "task_ids": [],
        }
        self._write_conversation(record)
        return self._conversation_summary(record, [])

    def list_conversations(self, limit: int = 50) -> AgentConversationListResponse:
        conversations: list[AgentConversationSummary] = []
        skipped_files = 0
        paths = sorted(
            self.settings.agent_conversations_dir.glob("CV-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            if len(conversations) >= limit:
                break
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                tasks = self._conversation_tasks(record)
                conversations.append(self._conversation_summary(record, tasks))
            except (OSError, json.JSONDecodeError, ValueError, AgentTaskNotFound):
                skipped_files += 1
        return AgentConversationListResponse(
            conversations=conversations,
            skipped_files=skipped_files,
        )

    def get_conversation(self, conversation_id: str) -> AgentConversationDetail:
        record = self._read_conversation(conversation_id)
        tasks = self._conversation_tasks(record)
        summary = self._conversation_summary(record, tasks)
        return AgentConversationDetail(**summary.model_dump(), tasks=tasks)

    def get_task(self, task_id: str) -> AgentTaskResponse:
        return AgentTaskResponse.model_validate(self._read_task(task_id))

    def execute_task(self, task_id: str, *, approved: bool) -> AgentTaskResponse:
        with self._lock:
            task = self._read_task(task_id)
            status = AgentTaskStatus(task["status"])
            if status == AgentTaskStatus.CANCELED:
                raise AgentTaskConflict("Canceled tasks cannot be executed.")
            if status in {
                AgentTaskStatus.RUNNING,
                AgentTaskStatus.COMPLETED,
                AgentTaskStatus.FAILED,
            }:
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

    def get_csv_columns(self, path: str) -> dict:
        result = self.tools.file_tools.read_file(path, start_line=1, end_line=5)
        csv_info = result.get("csv")
        if csv_info is None:
            raise AgentSecurityError("CSV 파일 경로가 필요합니다.")
        return {
            "path": result["path"],
            "columns": csv_info["columns"],
        }

    def list_runs(self, limit: int = 50) -> dict:
        runs: list[dict] = []
        skipped_files = 0
        paths = sorted(
            self.settings.agent_runs_dir.glob("AT-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            if len(runs) >= limit:
                break
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                task = AgentTaskResponse.model_validate(record)
            except (OSError, json.JSONDecodeError, ValueError):
                skipped_files += 1
                continue
            runs.append(
                {
                    "task_id": task.task_id,
                    "request": task.request,
                    "status": task.status,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "tools_used": task.tools_used,
                    "read_files": task.read_files,
                    "created_files": task.created_files,
                    "modified_files": task.modified_files,
                    "error": task.error,
                }
            )
        return {"runs": runs, "skipped_files": skipped_files}

    def _record_rejected_plan(
        self,
        request: AgentPlanRequest,
        message: str,
        *,
        conversation_id: str,
    ) -> AgentTaskResponse:
        task_id = self._new_task_id()
        timestamp = datetime.now(timezone.utc).isoformat()
        task = {
            "task_id": task_id,
            "conversation_id": conversation_id,
            "request": request.request,
            "status": AgentTaskStatus.FAILED.value,
            "permission_level": int(request.permission_level),
            "plan": [
                "요청에 포함된 경로를 Agent 안전성 규칙으로 검사합니다.",
                f"요청을 거부합니다: {message}",
                "파일 또는 명령을 실행하지 않고 거부 결과를 작업 기록에 저장합니다.",
            ],
            "required_tools": [],
            "actions": [],
            "requires_approval": False,
            "approved": False,
            "workspace": str(self.permissions.workspace),
            "created_at": timestamp,
            "started_at": None,
            "completed_at": timestamp,
            "current_action": None,
            "progress_step": 0,
            "progress_total": 0,
            "tools_used": [],
            "read_files": [],
            "created_files": [],
            "modified_files": [],
            "backups": [],
            "execution_records": [],
            "results": [],
            "error": message,
            "cancel_requested": False,
        }
        self._write_task(task)
        self._append_task_to_conversation(conversation_id, task_id)
        return AgentTaskResponse.model_validate(task)

    def _ensure_conversation(self, conversation_id: str | None, *, title: str) -> str:
        if conversation_id:
            self._read_conversation(conversation_id)
            return conversation_id
        return self.create_conversation(title=title).conversation_id

    def _append_task_to_conversation(self, conversation_id: str, task_id: str) -> None:
        with self._lock:
            record = self._read_conversation(conversation_id)
            task_ids = record.setdefault("task_ids", [])
            if not task_ids and record.get("title") == "새 Agent 대화":
                task = self._read_task(task_id)
                record["title"] = self._conversation_title(task.get("request"))
            if task_id not in task_ids:
                task_ids.append(task_id)
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_conversation(record)

    def _conversation_tasks(self, record: dict) -> list[AgentTaskResponse]:
        return [self.get_task(str(task_id)) for task_id in record.get("task_ids", [])]

    @staticmethod
    def _conversation_summary(
        record: dict,
        tasks: list[AgentTaskResponse],
    ) -> AgentConversationSummary:
        return AgentConversationSummary(
            conversation_id=record["conversation_id"],
            title=record["title"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            task_count=len(tasks),
            last_task_status=tasks[-1].status if tasks else None,
        )

    @staticmethod
    def _conversation_title(value: str | None) -> str:
        normalized = " ".join((value or "").split())
        return normalized[:80] or "새 Agent 대화"

    def _conversation_path(self, conversation_id: str) -> Path:
        if not conversation_id.startswith("CV-") or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in conversation_id
        ):
            raise AgentConversationNotFound(conversation_id)
        return self.settings.agent_conversations_dir / f"{conversation_id}.json"

    def _read_conversation(self, conversation_id: str) -> dict:
        path = self._conversation_path(conversation_id)
        with self._lock:
            if not path.is_file():
                raise AgentConversationNotFound(conversation_id)
            return json.loads(path.read_text(encoding="utf-8"))

    def _write_conversation(self, record: dict) -> None:
        path = self._conversation_path(record["conversation_id"])
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)

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

    @staticmethod
    def _new_conversation_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"CV-{now}-{uuid4().hex[:6].upper()}"


__all__ = [
    "AgentApprovalRequired",
    "AgentConversationNotFound",
    "AgentPermissionError",
    "AgentSecurityError",
    "AgentService",
    "AgentTaskConflict",
    "AgentTaskNotFound",
]
