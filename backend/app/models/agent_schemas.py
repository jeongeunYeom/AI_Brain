from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentPermissionLevel(IntEnum):
    READ_ONLY = 1
    SAFE_CREATE = 2
    APPROVED_EXECUTION = 3


class AgentTaskStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class AgentToolName(str, Enum):
    LIST_DIRECTORY = "list_directory"
    READ_FILE = "read_file"
    CREATE_FILE = "create_file"
    EDIT_FILE = "edit_file"
    RUN_PYTHON = "run_python"


class AgentAction(BaseModel):
    action_id: str
    tool: AgentToolName
    description: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    target_files: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    preview: str | None = None


class AgentPlanRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, pattern=r"^CV-[A-Z0-9-]+$")
    target_path: str | None = Field(default=None, max_length=500)
    output_path: str | None = Field(default=None, max_length=500)
    x_column: str | None = Field(default=None, max_length=200)
    y_column: str | None = Field(default=None, max_length=200)
    content: str | None = Field(default=None, max_length=100_000)
    old_text: str | None = Field(default=None, max_length=50_000)
    new_text: str | None = Field(default=None, max_length=50_000)
    python_code: str | None = Field(default=None, max_length=30_000)
    permission_level: AgentPermissionLevel = AgentPermissionLevel.APPROVED_EXECUTION


class AgentExecuteRequest(BaseModel):
    approved: bool = False


class AgentTaskResponse(BaseModel):
    task_id: str
    conversation_id: str | None = None
    request: str
    status: AgentTaskStatus
    permission_level: AgentPermissionLevel
    plan: list[str]
    required_tools: list[AgentToolName]
    actions: list[AgentAction]
    requires_approval: bool
    approved: bool = False
    workspace: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    current_action: str | None = None
    progress_step: int = 0
    progress_total: int = 0
    tools_used: list[str] = Field(default_factory=list)
    read_files: list[str] = Field(default_factory=list)
    created_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    backups: list[str] = Field(default_factory=list)
    execution_records: list[dict[str, Any]] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    cancel_requested: bool = False


class AgentWorkspaceEntry(BaseModel):
    name: str
    path: str
    kind: str
    extension: str | None = None
    size_bytes: int | None = None
    modified_at: str


class AgentWorkspaceResponse(BaseModel):
    path: str
    entries: list[AgentWorkspaceEntry]


class AgentCsvColumnsResponse(BaseModel):
    path: str
    columns: list[str]


class AgentRunSummary(BaseModel):
    task_id: str
    request: str
    status: AgentTaskStatus
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    read_files: list[str] = Field(default_factory=list)
    created_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    error: str | None = None


class AgentRunListResponse(BaseModel):
    runs: list[AgentRunSummary]
    skipped_files: int = 0


class AgentConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class AgentConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    task_count: int = 0
    last_task_status: AgentTaskStatus | None = None


class AgentConversationListResponse(BaseModel):
    conversations: list[AgentConversationSummary]
    skipped_files: int = 0


class AgentConversationDetail(AgentConversationSummary):
    tasks: list[AgentTaskResponse] = Field(default_factory=list)
