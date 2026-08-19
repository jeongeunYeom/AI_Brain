"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppIconRail, MobileModeTabs } from "@/components/AppNavigation";
import {
  AgentPermissionLevel,
  AgentFilePreview,
  AgentConversationSummary,
  AgentTask,
  AgentRunSummary,
  cancelAgentTask,
  createAgentConversation,
  createAgentPlan,
  executeAgentTask,
  getAgentCsvColumns,
  getAgentConversation,
  getAgentFilePreview,
  getAgentFileUrl,
  getAgentTask,
  listAgentWorkspace,
  listAgentConversations,
  listAgentRuns,
  WorkspaceEntry,
} from "@/lib/agentApi";

function formatBytes(value?: number | null): string {
  if (value == null) return "-";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function statusClass(status: AgentTask["status"]): string {
  if (status === "completed") return "bg-emerald-100 text-emerald-700";
  if (status === "failed") return "bg-red-100 text-red-700";
  if (status === "running") return "bg-amber-100 text-amber-700";
  if (status === "canceled") return "bg-slate-200 text-slate-600";
  return "bg-sky-100 text-sky-700";
}

function formatRunTime(value: string): string {
  return new Intl.DateTimeFormat("ko-KR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(new Date(value));
}

function formatDuration(run: AgentRunSummary): string {
  if (!run.started_at || !run.completed_at) return "-";
  const milliseconds = new Date(run.completed_at).getTime() - new Date(run.started_at).getTime();
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return "-";
  return `${(milliseconds / 1000).toFixed(1)}초`;
}

export default function AgentPage() {
  const [request, setRequest] = useState("");
  const [targetPath, setTargetPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [xColumn, setXColumn] = useState("");
  const [yColumn, setYColumn] = useState("");
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [columnsLoading, setColumnsLoading] = useState(false);
  const [columnError, setColumnError] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [oldText, setOldText] = useState("");
  const [newText, setNewText] = useState("");
  const [pythonCode, setPythonCode] = useState("");
  const [permissionLevel, setPermissionLevel] =
    useState<AgentPermissionLevel>(3);
  const [task, setTask] = useState<AgentTask | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversationTasks, setConversationTasks] = useState<AgentTask[]>([]);
  const [conversations, setConversations] = useState<AgentConversationSummary[]>([]);
  const [entries, setEntries] = useState<WorkspaceEntry[]>([]);
  const [workspacePath, setWorkspacePath] = useState(".");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<AgentFilePreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [skippedRuns, setSkippedRuns] = useState(0);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  async function refreshConversations() {
    try {
      const result = await listAgentConversations();
      setConversations(result.conversations);
    } catch (err) {
      setError(err instanceof Error ? err.message : "대화 목록을 읽지 못했습니다.");
    }
  }

  async function openConversation(nextConversationId: string) {
    setBusy(true);
    setError(null);
    try {
      const detail = await getAgentConversation(nextConversationId);
      setConversationId(detail.conversation_id);
      setConversationTasks(detail.tasks);
      setTask(detail.tasks.at(-1) ?? null);
      setRequest("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "대화를 복원하지 못했습니다.");
    } finally {
      setBusy(false);
    }
  }

  async function startNewConversation() {
    setBusy(true);
    setError(null);
    try {
      const created = await createAgentConversation();
      setConversationId(created.conversation_id);
      setConversationTasks([]);
      setTask(null);
      setRequest("");
      await refreshConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "새 대화를 만들지 못했습니다.");
    } finally {
      setBusy(false);
    }
  }

  async function refreshRuns() {
    setRunsLoading(true);
    try {
      const result = await listAgentRuns();
      setRuns(result.runs);
      setSkippedRuns(result.skipped_files);
    } catch (err) {
      setError(err instanceof Error ? err.message : "작업 기록을 읽지 못했습니다.");
    } finally {
      setRunsLoading(false);
    }
  }

  async function openRun(taskId: string) {
    setBusy(true);
    setError(null);
    try {
      const selected = await getAgentTask(taskId);
      if (selected.conversation_id) {
        await openConversation(selected.conversation_id);
      } else {
        setConversationId(null);
        setConversationTasks([selected]);
        setTask(selected);
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "작업 상세 기록을 읽지 못했습니다.");
    } finally {
      setBusy(false);
    }
  }

  async function openPreview(path: string) {
    setPreviewLoading(true);
    setError(null);
    try {
      setPreview(await getAgentFilePreview(path));
    } catch (err) {
      setError(err instanceof Error ? err.message : "파일을 미리 볼 수 없습니다.");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function refreshWorkspace(path = workspacePath) {
    try {
      const listing = await listAgentWorkspace(path);
      setWorkspacePath(listing.path || ".");
      setEntries(listing.entries);
    } catch (err) {
      setError(err instanceof Error ? err.message : "작업공간을 읽지 못했습니다.");
    }
  }

  useEffect(() => {
    void refreshWorkspace(".");
    void refreshRuns();
    void refreshConversations();
  }, []);

  useEffect(() => {
    const path = targetPath.trim();
    setCsvColumns([]);
    setXColumn("");
    setYColumn("");
    setColumnError(null);

    if (!path.toLowerCase().endsWith(".csv")) return;

    let canceled = false;
    setColumnsLoading(true);
    void getAgentCsvColumns(path)
      .then((result) => {
        if (!canceled) setCsvColumns(result.columns);
      })
      .catch((err) => {
        if (!canceled) {
          setColumnError(
            err instanceof Error ? err.message : "CSV 열 목록을 읽지 못했습니다.",
          );
        }
      })
      .finally(() => {
        if (!canceled) setColumnsLoading(false);
      });

    return () => {
      canceled = true;
    };
  }, [targetPath]);

  async function makePlan() {
    if (!request.trim()) return;
    setBusy(true);
    setError(null);
    setTask(null);
    try {
      const planned = await createAgentPlan({
        request: request.trim(),
        conversation_id: conversationId || undefined,
        target_path: targetPath.trim() || undefined,
        output_path: outputPath.trim() || undefined,
        x_column: xColumn || undefined,
        y_column: yColumn || undefined,
        content: content || undefined,
        old_text: oldText || undefined,
        new_text: newText || undefined,
        python_code: pythonCode || undefined,
        permission_level: permissionLevel,
      });
      setConversationId(planned.conversation_id ?? null);
      setConversationTasks((current) => {
        const sameConversation =
          current.length === 0 || current[0].conversation_id === planned.conversation_id;
        return sameConversation ? [...current, planned] : [planned];
      });
      setTask(planned);
      setRequest("");
      await refreshConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "작업 계획 생성에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  }

  async function execute(approved: boolean) {
    if (!task) return;
    setBusy(true);
    setError(null);
    try {
      let current = await executeAgentTask(task.task_id, approved);
      setTask(current);
      setConversationTasks((tasks) =>
        tasks.map((item) => item.task_id === current.task_id ? current : item),
      );
      while (current.status === "running") {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        current = await getAgentTask(task.task_id);
        setTask(current);
        setConversationTasks((tasks) =>
          tasks.map((item) => item.task_id === current.task_id ? current : item),
        );
      }
      await refreshWorkspace(".");
      await refreshRuns();
      await refreshConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Agent 작업 실행에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    if (!task) return;
    setBusy(true);
    setError(null);
    try {
      const canceled = await cancelAgentTask(task.task_id);
      setTask(canceled);
      setConversationTasks((tasks) =>
        tasks.map((item) => item.task_id === canceled.task_id ? canceled : item),
      );
      await refreshRuns();
      await refreshConversations();
    } catch (err) {
      setError(err instanceof Error ? err.message : "작업 취소에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  }

  function chooseEntry(entry: WorkspaceEntry) {
    if (entry.kind === "directory") {
      void refreshWorkspace(entry.path);
      return;
    }
    setTargetPath(entry.path);
  }

  const isCsvTarget = targetPath.trim().toLowerCase().endsWith(".csv");

  return (
    <main className="min-h-screen bg-white text-slate-900 md:pl-[72px] lg:pl-[356px]">
      <AppIconRail />
      <AgentMobileDrawer
        open={mobileMenuOpen}
        runs={runs}
        onClose={() => setMobileMenuOpen(false)}
        onOpenRun={(taskId) => {
          setMobileMenuOpen(false);
          void openRun(taskId);
        }}
      />
      <div className="mx-auto max-w-6xl space-y-5 px-4 py-4 sm:px-6 lg:px-8 lg:py-6">
        <header className="border-b border-slate-200 bg-white pb-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex min-w-0 items-start gap-3">
              <button
                type="button"
                onClick={() => setMobileMenuOpen(true)}
                className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 md:hidden"
                aria-label="메뉴 열기"
              >
                <span className="text-lg">☰</span>
              </button>
              <div>
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-600">
                AI_Brain Research Agent
              </p>
              <h1 className="mt-1 text-xl font-black sm:text-2xl">석유공학 작업 Agent</h1>
              <p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500 sm:text-sm">
                계획을 확인하고 승인한 뒤 안전한 workspace 안에서 연구 작업을 실행하세요.
              </p>
              </div>
            </div>
            <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
              <select
                value={conversationId ?? ""}
                onChange={(event) => {
                  if (event.target.value) void openConversation(event.target.value);
                }}
                disabled={busy}
                aria-label="Agent 대화 선택"
                className="min-w-0 flex-1 rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-bold sm:max-w-64"
              >
                <option value="">대화 선택</option>
                {conversations.map((conversation) => (
                  <option key={conversation.conversation_id} value={conversation.conversation_id}>
                    {conversation.title} ({conversation.task_count})
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={busy}
                onClick={() => void startNewConversation()}
                className="rounded-xl bg-indigo-600 px-4 py-2 text-xs font-bold text-white disabled:opacity-40"
              >
                ＋ 새 대화
              </button>
              <span className="rounded-full bg-emerald-50 px-3 py-2 text-[11px] font-bold text-emerald-700">
                삭제 · 셸 · 인터넷 차단
              </span>
            </div>
            <MobileModeTabs />
          </div>
        </header>

        <div className="space-y-5 pb-48 sm:pb-44">
          <section className="space-y-5">
            {conversationTasks
              .filter((item) => item.task_id !== task?.task_id)
              .map((item) => (
                <div key={item.task_id} className="space-y-3 border-b border-slate-100 pb-5">
                  <div className="flex justify-end">
                    <div className="max-w-[82%] rounded-3xl rounded-br-md bg-indigo-50 px-5 py-3 text-sm leading-6 text-indigo-950">
                      {item.request}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setTask(item)}
                    className="ml-0 flex w-full items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-left text-xs shadow-sm sm:ml-12 sm:w-[calc(100%-3rem)]"
                  >
                    <span className="min-w-0">
                      <span className="block font-bold">이전 Agent 작업</span>
                      <span className="mt-1 block truncate text-slate-400">{item.task_id}</span>
                    </span>
                    <span className={`shrink-0 rounded-full px-2.5 py-1 font-bold ${statusClass(item.status)}`}>
                      {item.status}
                    </span>
                  </button>
                </div>
              ))}
            {!task && (
              <div className="flex items-start gap-3">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-xs font-black text-white">
                  AI
                </span>
                <div className="max-w-2xl rounded-3xl rounded-tl-md border border-slate-200 bg-white px-5 py-4 text-sm leading-6 shadow-sm">
                  <p className="font-black">어떤 연구 작업을 도와드릴까요?</p>
                  <p className="mt-1 text-slate-500">
                    아래 입력창에 요청을 적어주세요. 실행 전에 작업 계획과 대상 파일을 먼저 보여드릴게요.
                  </p>
                </div>
              </div>
            )}

            {task && (
              <div className="flex justify-end">
                <div className="max-w-[82%] rounded-3xl rounded-br-md bg-indigo-600 px-5 py-3 text-sm leading-6 text-white shadow-sm">
                  {task.request}
                </div>
              </div>
            )}

            <div className="relative ml-0 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:ml-12">
              <span className="absolute -left-12 top-0 hidden h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-xs font-black text-white sm:flex">
                AI
              </span>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="font-black">작업 계획과 승인</h2>
                  <p className="mt-1 text-xs text-slate-500">
                    파일 변경 또는 Python 실행은 승인 전에는 수행되지 않습니다.
                  </p>
                </div>
                {task && (
                  <span className={`rounded-full px-3 py-1 text-xs font-bold ${statusClass(task.status)}`}>
                    {task.status}
                  </span>
                )}
              </div>

              {!task ? (
                <p className="mt-6 rounded-2xl bg-slate-50 p-5 text-sm text-slate-500">
                  메시지를 보내면 이곳에 계획, 승인 버튼, 진행 상태와 결과가 대화 순서대로 표시됩니다.
                </p>
              ) : (
                <div className="mt-5 space-y-5">
                  <div className="rounded-2xl bg-slate-900 p-4 text-xs text-slate-200">
                    <p className="font-bold text-white">{task.task_id}</p>
                    <p className="mt-1 break-all text-slate-400">workspace: {task.workspace}</p>
                    {task.status === "running" && (
                      <div className="mt-4">
                        <div className="flex items-center justify-between gap-3 text-[11px]">
                          <span className="truncate text-emerald-300">
                            {task.current_action ?? "실행 중"}
                          </span>
                          <span>
                            {task.progress_step}/{task.progress_total}
                          </span>
                        </div>
                        <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-700">
                          <div
                            className="h-full rounded-full bg-emerald-400 transition-all"
                            style={{
                              width: `${task.progress_total ? Math.max(5, (task.progress_step / task.progress_total) * 100) : 5}%`,
                            }}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  <ol className="space-y-2">
                    {task.plan.map((step, index) => (
                      <li key={`${index}:${step}`} className="flex gap-3 rounded-2xl bg-slate-50 p-3 text-sm">
                        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-xs font-black text-emerald-700">
                          {index + 1}
                        </span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ol>

                  <div className="space-y-3">
                    {task.actions.map((action) => (
                      <article key={action.action_id} className="rounded-2xl border border-slate-200 p-4">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="rounded-full bg-indigo-50 px-2 py-1 text-[11px] font-bold text-indigo-700">
                            {action.tool}
                          </span>
                          {action.requires_approval && (
                            <span className="rounded-full bg-amber-50 px-2 py-1 text-[11px] font-bold text-amber-700">
                              승인 필요
                            </span>
                          )}
                        </div>
                        <p className="mt-2 text-sm font-semibold">{action.description}</p>
                        {action.target_files.length > 0 && (
                          <p className="mt-2 text-xs text-slate-500">
                            대상: {action.target_files.join(", ")}
                          </p>
                        )}
                        {action.preview && (
                          <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-xl bg-slate-950 p-3 text-[11px] leading-5 text-slate-200">
                            {action.preview}
                          </pre>
                        )}
                      </article>
                    ))}
                  </div>

                  {task.status === "planned" && (
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => execute(task.requires_approval)}
                        className="flex-1 rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-bold text-white disabled:opacity-40"
                      >
                        {task.requires_approval ? "2. 승인하고 실행" : "2. 읽기 작업 실행"}
                      </button>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={cancel}
                        className="rounded-2xl border border-slate-300 px-4 py-3 text-sm font-bold text-slate-600 disabled:opacity-40"
                      >
                        취소
                      </button>
                    </div>
                  )}

                  {(task.status === "completed" || task.status === "failed") && (
                    <div className="space-y-3 rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm">
                      <h3 className="font-black">작업 결과</h3>
                      {task.error && <p className="text-red-700">오류: {task.error}</p>}
                      <ResultList title="읽은 파일" values={task.read_files} />
                      <ResultFiles title="생성된 파일" values={task.created_files} onPreview={openPreview} previewLoading={previewLoading} />
                      <ResultList title="수정된 파일" values={task.modified_files} />
                      <ResultList title="백업" values={task.backups} />
                      {task.results.map((item) => (
                        <details key={item.action_id} className="rounded-xl border border-slate-200 bg-white p-3">
                          <summary className="cursor-pointer text-xs font-bold">
                            {item.tool} · {item.description}
                          </summary>
                          <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap text-[11px] leading-5 text-slate-600">
                            {JSON.stringify(item.result, null, 2)}
                          </pre>
                        </details>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-5">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-black">Agent workspace</h2>
                  <p className="mt-1 text-xs text-slate-500">현재 경로: {workspacePath}</p>
                </div>
                <button
                  type="button"
                  onClick={() => refreshWorkspace(workspacePath)}
                  className="rounded-xl border border-slate-200 px-3 py-2 text-xs font-bold"
                >
                  새로고침
                </button>
              </div>
              <div className="mt-4 max-h-96 overflow-auto rounded-2xl border border-slate-200">
                {workspacePath !== "." && (
                  <button
                    type="button"
                    onClick={() => refreshWorkspace(".")}
                    className="w-full border-b border-slate-200 px-4 py-3 text-left text-xs font-bold text-indigo-600"
                  >
                    ↩ workspace 루트
                  </button>
                )}
                {entries.length === 0 ? (
                  <p className="p-6 text-center text-sm text-slate-400">workspace가 비어 있습니다.</p>
                ) : (
                  entries.map((entry) => (
                    <button
                      key={entry.path}
                      type="button"
                      onClick={() => chooseEntry(entry)}
                      className="grid w-full grid-cols-[1fr_auto] gap-3 border-b border-slate-100 px-4 py-3 text-left text-sm last:border-0 hover:bg-slate-50"
                    >
                      <span className="min-w-0">
                        <span className="block truncate font-semibold">
                          {entry.kind === "directory" ? "📁" : "📄"} {entry.name}
                        </span>
                        <span className="mt-1 block truncate text-[11px] text-slate-400">{entry.path}</span>
                      </span>
                      <span className="text-xs text-slate-400">{formatBytes(entry.size_bytes)}</span>
                    </button>
                  ))
                )}
              </div>
            </div>
          </section>
        </div>

        <section className="fixed bottom-0 right-0 z-30 border-t border-slate-200 bg-white/95 px-4 pb-4 pt-3 shadow-[0_-12px_36px_rgba(15,23,42,0.08)] backdrop-blur md:left-[72px] lg:left-[356px] sm:px-6 lg:px-8">
          <div className="mx-auto max-w-6xl">
            {error && (
              <div className="mb-2 rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-xs text-red-700">
                {error}
              </div>
            )}

            <details className="group mb-2 rounded-2xl border border-slate-200 bg-white">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-4 py-2 text-xs font-bold text-slate-600">
                <span>＋ 파일·출력·권한 설정</span>
                <span className="font-normal text-slate-400">
                  {targetPath || "대상 미지정"} · Level {permissionLevel}
                </span>
              </summary>
              <div className="max-h-[52vh] space-y-4 overflow-y-auto border-t border-slate-100 p-4">
                <div className="grid gap-3 sm:grid-cols-3">
                  <label className="text-xs font-semibold text-slate-600">
                    대상 파일/폴더
                    <input
                      value={targetPath}
                      onChange={(event) => setTargetPath(event.target.value)}
                      placeholder="예: data.csv"
                      className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none focus:border-indigo-400"
                    />
                  </label>
                  <label className="text-xs font-semibold text-slate-600">
                    결과 파일 경로
                    <input
                      value={outputPath}
                      onChange={(event) => setOutputPath(event.target.value)}
                      placeholder="예: results/report.txt"
                      className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none focus:border-indigo-400"
                    />
                  </label>
                  <label className="text-xs font-semibold text-slate-600">
                    권한 단계
                    <select
                      value={permissionLevel}
                      onChange={(event) => setPermissionLevel(Number(event.target.value) as AgentPermissionLevel)}
                      className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none"
                    >
                      <option value={1}>Level 1 · 읽기</option>
                      <option value={2}>Level 2 · 파일 생성</option>
                      <option value={3}>Level 3 · 수정·Python</option>
                    </select>
                  </label>
                </div>

                {isCsvTarget && (
                  <div className="rounded-2xl bg-indigo-50 p-4">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="text-xs font-semibold text-slate-600">
                        X축 열
                        <select value={xColumn} onChange={(event) => setXColumn(event.target.value)} disabled={columnsLoading || csvColumns.length === 0} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm disabled:opacity-50">
                          <option value="">자동 인식</option>
                          {csvColumns.map((column) => <option key={column} value={column} disabled={column === yColumn}>{column}</option>)}
                        </select>
                      </label>
                      <label className="text-xs font-semibold text-slate-600">
                        Y축 열
                        <select value={yColumn} onChange={(event) => setYColumn(event.target.value)} disabled={columnsLoading || csvColumns.length === 0} className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm disabled:opacity-50">
                          <option value="">자동 인식</option>
                          {csvColumns.map((column) => <option key={column} value={column} disabled={column === xColumn}>{column}</option>)}
                        </select>
                      </label>
                    </div>
                    {columnsLoading && <p className="mt-2 text-xs text-indigo-600">CSV 열을 불러오는 중...</p>}
                    {columnError && <p className="mt-2 text-xs text-red-700">{columnError}</p>}
                  </div>
                )}

                <details className="rounded-xl bg-slate-50 p-3">
                  <summary className="cursor-pointer text-xs font-bold text-slate-600">고급 입력</summary>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <label className="text-xs font-semibold text-slate-600 sm:col-span-2">새 파일 내용<textarea value={content} onChange={(event) => setContent(event.target.value)} rows={3} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs" /></label>
                    <label className="text-xs font-semibold text-slate-600">기존 문구<textarea value={oldText} onChange={(event) => setOldText(event.target.value)} rows={3} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs" /></label>
                    <label className="text-xs font-semibold text-slate-600">새 문구<textarea value={newText} onChange={(event) => setNewText(event.target.value)} rows={3} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs" /></label>
                    <label className="text-xs font-semibold text-slate-600 sm:col-span-2">Python 코드<textarea value={pythonCode} onChange={(event) => setPythonCode(event.target.value)} rows={5} className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs" /></label>
                  </div>
                </details>
              </div>
            </details>

            <div className="flex items-end gap-2 rounded-3xl border border-slate-300 bg-white p-2 shadow-lg focus-within:border-indigo-400 focus-within:ring-4 focus-within:ring-indigo-50">
              <textarea
                value={request}
                onChange={(event) => setRequest(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    if (!busy && request.trim()) void makePlan();
                  }
                }}
                rows={2}
                placeholder="Agent에게 연구 작업을 요청하세요..."
                className="max-h-32 min-h-[52px] flex-1 resize-none bg-transparent px-3 py-2 text-sm leading-6 outline-none placeholder:text-slate-400"
              />
              <button
                type="button"
                disabled={busy || !request.trim()}
                onClick={() => void makePlan()}
                aria-label="작업 계획 생성"
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-lg font-black text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {busy ? "…" : "↑"}
              </button>
            </div>
            <p className="mt-1 text-center text-[10px] text-slate-400">
              Enter로 전송 · Shift+Enter로 줄바꿈 · 실행은 계획 확인 후 별도 승인
            </p>
          </div>
        </section>

        <section className="hidden border-r border-slate-200 bg-slate-50 p-4 lg:fixed lg:inset-y-0 lg:left-[72px] lg:block lg:w-[284px] lg:overflow-y-auto">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="font-black">Agent 작업 기록</h2>
              <p className="mt-1 text-xs text-slate-500">최근 작업을 최신순으로 표시합니다.</p>
            </div>
            <button
              type="button"
              disabled={runsLoading}
              onClick={() => void refreshRuns()}
              className="rounded-xl border border-slate-200 px-3 py-2 text-xs font-bold disabled:opacity-40"
            >
              {runsLoading ? "불러오는 중..." : "새로고침"}
            </button>
          </div>
          {skippedRuns > 0 && (
            <p className="mt-3 rounded-xl bg-amber-50 p-3 text-xs text-amber-700">
              손상된 기록 파일 {skippedRuns}개는 목록에서 제외했습니다.
            </p>
          )}
          {runs.length === 0 ? (
            <p className="mt-4 rounded-2xl bg-slate-50 p-6 text-center text-sm text-slate-400">
              저장된 작업 기록이 없습니다.
            </p>
          ) : (
            <div className="mt-4 grid gap-3">
              {runs.map((run) => (
                <article key={run.task_id} className="rounded-2xl border border-slate-200 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="truncate font-mono text-[11px] text-slate-400">{run.task_id}</p>
                      <p className="mt-2 line-clamp-2 text-sm font-bold">{run.request}</p>
                    </div>
                    <span className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-bold ${statusClass(run.status)}`}>
                      {run.status}
                    </span>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-500">
                    <span>{formatRunTime(run.created_at)}</span>
                    <span>실행 {formatDuration(run)}</span>
                    <span>도구 {run.tools_used.length}개</span>
                    <span>결과 {run.created_files.length + run.modified_files.length}개</span>
                  </div>
                  {run.error && <p className="mt-3 line-clamp-2 text-xs text-red-700">{run.error}</p>}
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void openRun(run.task_id)}
                    className="mt-4 w-full rounded-xl bg-white px-3 py-2 text-xs font-bold text-emerald-700 ring-1 ring-slate-200 disabled:opacity-40"
                  >
                    상세 보기
                  </button>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
      {preview && <FilePreviewModal preview={preview} onClose={() => setPreview(null)} />}
    </main>
  );
}

const MOBILE_NAV_ITEMS = [
  { href: "/", icon: "⌂", label: "RAG 채팅" },
  { href: "/agent", icon: "✦", label: "Agent", active: true },
  { href: "/review", icon: "▧", label: "Figure Review" },
  { href: "/evaluation", icon: "◫", label: "평가" },
];

function AgentMobileDrawer({
  open,
  runs,
  onClose,
  onOpenRun,
}: {
  open: boolean;
  runs: AgentRunSummary[];
  onClose: () => void;
  onOpenRun: (taskId: string) => void;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 lg:hidden">
      <button type="button" onClick={onClose} className="absolute inset-0 bg-slate-950/35" aria-label="메뉴 닫기" />
      <aside className="relative h-full w-[86%] max-w-sm overflow-y-auto bg-white p-5 shadow-2xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-sm font-black text-white">AI</span>
            <div><p className="font-black">AI_Brain</p><p className="text-xs text-slate-400">Research workspace</p></div>
          </div>
          <button type="button" onClick={onClose} className="rounded-xl border border-slate-200 px-3 py-2 text-xs font-bold">닫기</button>
        </div>
        <nav className="mt-6 space-y-1 border-b border-slate-200 pb-5">
          {MOBILE_NAV_ITEMS.map((item) => (
            <Link key={item.href} href={item.href} onClick={onClose} className={`flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-bold ${item.active ? "bg-emerald-50 text-emerald-700" : "text-slate-600"}`}>
              <span className="w-6 text-center text-lg">{item.icon}</span>{item.label}
            </Link>
          ))}
        </nav>
        <div className="mt-5">
          <div className="flex items-center justify-between"><h2 className="font-black">최근 작업</h2><span className="text-xs text-slate-400">{runs.length}개</span></div>
          <div className="mt-3 space-y-2">
            {runs.slice(0, 12).map((run) => (
              <button key={run.task_id} type="button" onClick={() => onOpenRun(run.task_id)} className="w-full rounded-xl border border-slate-200 p-3 text-left">
                <div className="flex items-start justify-between gap-2"><p className="line-clamp-2 text-xs font-bold leading-5">{run.request}</p><span className={`shrink-0 rounded-full px-2 py-1 text-[10px] font-bold ${statusClass(run.status)}`}>{run.status}</span></div>
                <p className="mt-2 text-[10px] text-slate-400">{formatRunTime(run.created_at)}</p>
              </button>
            ))}
            {runs.length === 0 && <p className="rounded-xl bg-slate-50 p-4 text-center text-xs text-slate-400">저장된 작업이 없습니다.</p>}
          </div>
        </div>
      </aside>
    </div>
  );
}

function ResultFiles({ title, values, onPreview, previewLoading }: {
  title: string;
  values: string[];
  onPreview: (path: string) => void;
  previewLoading: boolean;
}) {
  if (values.length === 0) return null;
  return (
    <div>
      <p className="text-xs font-bold text-slate-500">{title}</p>
      <div className="mt-2 space-y-2">
        {values.map((value) => (
          <div key={value} className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white p-3">
            <span className="break-all font-mono text-xs">📄 {value}</span>
            <span className="flex gap-2">
              <button type="button" disabled={previewLoading} onClick={() => onPreview(value)} className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-bold text-indigo-700 disabled:opacity-40">미리보기</button>
              <a href={getAgentFileUrl(value, true)} className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-bold text-white">다운로드</a>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FilePreviewModal({ preview, onClose }: { preview: AgentFilePreview; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 p-4" onClick={onClose}>
      <section className="max-h-[90vh] w-full max-w-4xl overflow-hidden rounded-3xl bg-white shadow-2xl" onClick={(event) => event.stopPropagation()}>
        <header className="flex items-center justify-between gap-3 border-b border-slate-200 px-5 py-4">
          <div className="min-w-0"><h3 className="font-black">결과 파일 미리보기</h3><p className="truncate text-xs text-slate-500">{preview.path}</p></div>
          <button type="button" onClick={onClose} className="rounded-xl border border-slate-200 px-3 py-2 text-xs font-bold">닫기</button>
        </header>
        <div className="max-h-[72vh] overflow-auto p-5">
          {preview.kind === "image" && <img src={getAgentFileUrl(preview.path)} alt={preview.path} className="mx-auto max-h-[65vh] max-w-full rounded-xl border border-slate-200 object-contain" />}
          {preview.kind === "text" && <pre className="whitespace-pre-wrap rounded-xl bg-slate-950 p-4 text-xs leading-6 text-slate-100">{preview.content}</pre>}
          {preview.kind === "csv" && (
            <div>
              <div className="overflow-x-auto rounded-xl border border-slate-200">
                <table className="min-w-full text-left text-xs">
                  <thead className="bg-slate-100"><tr>{preview.columns.map((column, index) => <th key={`${index}:${column}`} className="whitespace-nowrap px-3 py-2 font-black">{column}</th>)}</tr></thead>
                  <tbody>{preview.rows.map((row, rowIndex) => <tr key={rowIndex} className="border-t border-slate-100">{row.map((cell, cellIndex) => <td key={cellIndex} className="whitespace-nowrap px-3 py-2">{cell}</td>)}</tr>)}</tbody>
                </table>
              </div>
              {preview.truncated && <p className="mt-3 text-xs text-amber-700">처음 100개 행만 표시합니다.</p>}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

function ResultList({ title, values }: { title: string; values: string[] }) {
  if (values.length === 0) return null;
  return (
    <div>
      <p className="text-xs font-bold text-slate-500">{title}</p>
      <ul className="mt-1 space-y-1 font-mono text-xs">
        {values.map((value) => (
          <li key={value}>- {value}</li>
        ))}
      </ul>
    </div>
  );
}
