import { API_BASE, requestJson } from "./http";

export type AgentPermissionLevel = 1 | 2 | 3;
export type AgentChartType = "scatter" | "line" | "bar" | "histogram";
export type AgentTaskStatus =
  | "planned"
  | "running"
  | "completed"
  | "failed"
  | "canceled";

export type AgentToolName =
  | "list_directory"
  | "read_file"
  | "search_knowledge_base"
  | "get_related_figures"
  | "create_file"
  | "edit_file"
  | "run_python";

export type AgentAction = {
  action_id: string;
  tool: AgentToolName;
  description: string;
  arguments: Record<string, unknown>;
  target_files: string[];
  requires_approval: boolean;
  preview?: string | null;
};

export type AgentTask = {
  task_id: string;
  conversation_id?: string | null;
  request: string;
  status: AgentTaskStatus;
  permission_level: AgentPermissionLevel;
  plan: string[];
  required_tools: AgentToolName[];
  actions: AgentAction[];
  requires_approval: boolean;
  approved: boolean;
  workspace: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  current_action?: string | null;
  progress_step: number;
  progress_total: number;
  tools_used: string[];
  read_files: string[];
  created_files: string[];
  modified_files: string[];
  backups: string[];
  execution_records: Array<Record<string, unknown>>;
  results: Array<{
    action_id: string;
    tool: AgentToolName;
    description: string;
    result: Record<string, unknown>;
  }>;
  validation_passed?: boolean | null;
  validation_records: Array<{
    action_id: string;
    tool: AgentToolName;
    passed: boolean;
    checks: Array<{
      name: string;
      passed: boolean;
      detail: string;
      path?: string;
    }>;
    errors: string[];
  }>;
  error?: string | null;
  cancel_requested: boolean;
};

export type AgentPlanInput = {
  request: string;
  conversation_id?: string;
  target_path?: string;
  output_path?: string;
  x_column?: string;
  y_column?: string;
  chart_type?: AgentChartType;
  content?: string;
  old_text?: string;
  new_text?: string;
  python_code?: string;
  permission_level: AgentPermissionLevel;
};

export type WorkspaceEntry = {
  name: string;
  path: string;
  kind: "file" | "directory";
  extension?: string | null;
  size_bytes?: number | null;
  modified_at: string;
};

export type WorkspaceListing = {
  path: string;
  entries: WorkspaceEntry[];
};

export type CsvColumnsResponse = {
  path: string;
  columns: string[];
};

export type AgentFilePreview =
  | { path: string; kind: "image" }
  | { path: string; kind: "text"; content: string }
  | { path: string; kind: "csv"; columns: string[]; rows: string[][]; truncated: boolean };

export type AgentRunSummary = Pick<
  AgentTask,
  | "task_id"
  | "request"
  | "status"
  | "created_at"
  | "started_at"
  | "completed_at"
  | "tools_used"
  | "read_files"
  | "created_files"
  | "modified_files"
  | "error"
>;

export type AgentRunList = {
  runs: AgentRunSummary[];
  skipped_files: number;
};

export type AgentConversationSummary = {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  task_count: number;
  last_task_status?: AgentTaskStatus | null;
};

export type AgentConversationDetail = AgentConversationSummary & {
  tasks: AgentTask[];
};

export type AgentConversationList = {
  conversations: AgentConversationSummary[];
  skipped_files: number;
};

export function createAgentPlan(input: AgentPlanInput): Promise<AgentTask> {
  return requestJson("/agent/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export function executeAgentTask(
  taskId: string,
  approved: boolean,
): Promise<AgentTask> {
  return requestJson(`/agent/tasks/${taskId}/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved }),
  });
}

export function getAgentTask(taskId: string): Promise<AgentTask> {
  return requestJson(`/agent/tasks/${taskId}`, { cache: "no-store" });
}

export function cancelAgentTask(taskId: string): Promise<AgentTask> {
  return requestJson(`/agent/tasks/${taskId}/cancel`, { method: "POST" });
}

export function listAgentWorkspace(path = "."): Promise<WorkspaceListing> {
  const query = new URLSearchParams({ path });
  return requestJson(`/agent/workspace?${query}`, { cache: "no-store" });
}

export function getAgentCsvColumns(path: string): Promise<CsvColumnsResponse> {
  const query = new URLSearchParams({ path });
  return requestJson(`/agent/csv-columns?${query}`, { cache: "no-store" });
}

export function getAgentFilePreview(path: string): Promise<AgentFilePreview> {
  const query = new URLSearchParams({ path });
  return requestJson(`/agent/files/preview?${query}`, { cache: "no-store" });
}

export function getAgentFileUrl(path: string, download = false): string {
  const query = new URLSearchParams({ path, download: String(download) });
  return `${API_BASE}/agent/files/content?${query}`;
}

export function listAgentRuns(limit = 50): Promise<AgentRunList> {
  const query = new URLSearchParams({ limit: String(limit) });
  return requestJson(`/agent/runs?${query}`, { cache: "no-store" });
}

export function createAgentConversation(
  title?: string,
): Promise<AgentConversationSummary> {
  return requestJson("/agent/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export function listAgentConversations(
  limit = 50,
): Promise<AgentConversationList> {
  const query = new URLSearchParams({ limit: String(limit) });
  return requestJson(`/agent/conversations?${query}`, { cache: "no-store" });
}

export function getAgentConversation(
  conversationId: string,
): Promise<AgentConversationDetail> {
  return requestJson(`/agent/conversations/${conversationId}`, {
    cache: "no-store",
  });
}
