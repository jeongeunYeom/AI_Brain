from __future__ import annotations

from datetime import datetime, timezone

from app.agents.permission_manager import PermissionManager
from app.agents.tool_registry import ToolRegistry
from app.models.agent_schemas import (
    AgentAction,
    AgentPermissionLevel,
    AgentTaskStatus,
)


class AgentExecutor:
    def __init__(
        self,
        permissions: PermissionManager,
        tools: ToolRegistry,
    ):
        self.permissions = permissions
        self.tools = tools

    def execute(self, task: dict, *, is_canceled=None, on_progress=None) -> dict:
        permission_level = AgentPermissionLevel(task["permission_level"])
        actions = [AgentAction.model_validate(value) for value in task["actions"]]

        task["status"] = AgentTaskStatus.RUNNING.value
        task["started_at"] = datetime.now(timezone.utc).isoformat()
        task["error"] = None
        task["progress_total"] = len(actions)
        task["progress_step"] = 0
        task["current_action"] = None
        if on_progress:
            on_progress(task)

        for index, action in enumerate(actions, start=1):
            if task.get("cancel_requested") or (is_canceled and is_canceled()):
                task["status"] = AgentTaskStatus.CANCELED.value
                task["completed_at"] = datetime.now(timezone.utc).isoformat()
                task["current_action"] = None
                if on_progress:
                    on_progress(task)
                return task

            task["current_action"] = action.description
            task["progress_step"] = index
            if on_progress:
                on_progress(task)

            self.permissions.require_tool_level(permission_level, action.tool)
            result = self.tools.execute(action, task_id=task["task_id"])
            task.setdefault("tools_used", []).append(action.tool.value)
            task.setdefault("results", []).append(
                {
                    "action_id": action.action_id,
                    "tool": action.tool.value,
                    "description": action.description,
                    "result": result,
                }
            )
            self._collect_artifacts(task, action, result)

            if action.tool.value == "run_python":
                task.setdefault("execution_records", []).append(
                    {
                        "action_id": action.action_id,
                        "code": action.arguments.get("code", ""),
                        "exit_code": result.get("exit_code"),
                        "duration_seconds": result.get("duration_seconds"),
                        "stdout_record": result.get("stdout_record"),
                        "stderr_record": result.get("stderr_record"),
                    }
                )
                if not result.get("success"):
                    raise RuntimeError(
                        result.get("stderr") or "Python execution failed."
                    )
            if on_progress:
                on_progress(task)

        task["status"] = AgentTaskStatus.COMPLETED.value
        task["current_action"] = None
        task["completed_at"] = datetime.now(timezone.utc).isoformat()
        return task

    @staticmethod
    def _collect_artifacts(task: dict, action: AgentAction, result: dict) -> None:
        if action.tool.value == "read_file" and result.get("path"):
            AgentExecutor._append_unique(task, "read_files", result["path"])
        elif action.tool.value == "create_file" and result.get("path"):
            AgentExecutor._append_unique(task, "created_files", result["path"])
        elif action.tool.value == "edit_file":
            if result.get("path"):
                AgentExecutor._append_unique(task, "modified_files", result["path"])
            if result.get("backup"):
                AgentExecutor._append_unique(task, "backups", result["backup"])
        elif action.tool.value == "run_python":
            for path in result.get("created_files", []):
                AgentExecutor._append_unique(task, "created_files", path)
            for path in result.get("modified_files", []):
                AgentExecutor._append_unique(task, "modified_files", path)

    @staticmethod
    def _append_unique(task: dict, key: str, value: str) -> None:
        values = task.setdefault(key, [])
        if value not in values:
            values.append(value)
