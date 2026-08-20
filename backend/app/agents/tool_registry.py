from __future__ import annotations

from typing import Any

from app.agents.permission_manager import PermissionManager
from app.core.config import Settings
from app.models.agent_schemas import AgentAction, AgentToolName
from app.tools.directory_tools import DirectoryTools
from app.tools.file_tools import FileTools
from app.tools.python_tools import PythonTools


class ToolRegistry:
    def __init__(self, settings: Settings, permissions: PermissionManager):
        self.settings = settings
        self.directory_tools = DirectoryTools(permissions)
        self.file_tools = FileTools(settings, permissions)
        self.python_tools = PythonTools(settings, permissions)
        self._knowledge_tools = None

    def _get_knowledge_tools(self):
        if self._knowledge_tools is None:
            from app.tools.knowledge_tools import KnowledgeTools

            self._knowledge_tools = KnowledgeTools(self.settings)
        return self._knowledge_tools

    def execute(self, action: AgentAction, *, task_id: str) -> dict[str, Any]:
        args = action.arguments
        if action.tool == AgentToolName.LIST_DIRECTORY:
            return self.directory_tools.list_directory(**args)
        if action.tool == AgentToolName.READ_FILE:
            return self.file_tools.read_file(**args)
        if action.tool == AgentToolName.SEARCH_KNOWLEDGE_BASE:
            return self._get_knowledge_tools().search_knowledge_base(**args)
        if action.tool == AgentToolName.GET_RELATED_FIGURES:
            return self._get_knowledge_tools().get_related_figures(**args)
        if action.tool == AgentToolName.CREATE_FILE:
            return self.file_tools.create_file(**args)
        if action.tool == AgentToolName.EDIT_FILE:
            return self.file_tools.edit_file(task_id=task_id, **args)
        if action.tool == AgentToolName.RUN_PYTHON:
            code = args.get("code")
            if not isinstance(code, str):
                raise ValueError("run_python requires a code string.")
            return self.python_tools.run_python(code, task_id=task_id)
        raise ValueError(f"Unknown Agent tool: {action.tool}")
