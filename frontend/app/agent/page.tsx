"use client";

import { useEffect, useState } from "react";
import {
  AgentPermissionLevel,
  AgentFilePreview,
  AgentTask,
  cancelAgentTask,
  createAgentPlan,
  executeAgentTask,
  getAgentCsvColumns,
  getAgentFilePreview,
  getAgentFileUrl,
  getAgentTask,
  listAgentWorkspace,
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
  const [entries, setEntries] = useState<WorkspaceEntry[]>([]);
  const [workspacePath, setWorkspacePath] = useState(".");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<AgentFilePreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

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
      setTask(planned);
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
      while (current.status === "running") {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        current = await getAgentTask(task.task_id);
        setTask(current);
      }
      await refreshWorkspace(".");
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
      setTask(await cancelAgentTask(task.task_id));
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
    <main className="min-h-screen bg-slate-100 px-4 py-8 text-slate-900">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-600">
                AI_Brain Research Agent
              </p>
              <h1 className="mt-2 text-2xl font-black">승인 기반 석유공학 작업 Agent</h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
                먼저 작업 계획과 대상 파일, 실행 코드를 확인한 뒤 승인하세요. Agent는 설정된
                workspace 안에서만 파일을 읽고, 생성·수정·Python 실행 기록을 data/agent_runs에
                저장합니다.
              </p>
            </div>
            <span className="rounded-full bg-emerald-50 px-4 py-2 text-xs font-bold text-emerald-700">
              삭제 · 셸 명령 · 인터넷 차단
            </span>
          </div>
        </header>

        <div className="grid gap-6 xl:grid-cols-[1fr_1.35fr]">
          <section className="space-y-5 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <div>
              <label className="text-sm font-bold">사용자 요청</label>
              <textarea
                value={request}
                onChange={(event) => setRequest(event.target.value)}
                rows={5}
                placeholder="예: johansen_results.csv를 분석해서 산점도를 만들어줘."
                className="mt-2 w-full rounded-2xl border border-slate-200 p-3 text-sm outline-none focus:border-emerald-400"
              />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <label className="text-xs font-semibold text-slate-600">
                대상 파일/폴더
                <input
                  value={targetPath}
                  onChange={(event) => setTargetPath(event.target.value)}
                  placeholder="예: data.csv 또는 scripts"
                  className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400"
                />
              </label>
              <label className="text-xs font-semibold text-slate-600">
                결과 파일 경로
                <input
                  value={outputPath}
                  onChange={(event) => setOutputPath(event.target.value)}
                  placeholder="예: results/report.txt"
                  className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none focus:border-emerald-400"
                />
              </label>
            </div>

            {isCsvTarget && (
              <div className="rounded-2xl border border-indigo-100 bg-indigo-50/60 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-bold text-indigo-950">산점도 열 선택</p>
                    <p className="mt-1 text-xs text-indigo-700">
                      비워두면 요청 문장에서 열을 인식하고, 찾지 못하면 숫자 열을 자동 선택합니다.
                    </p>
                  </div>
                  {columnsLoading && (
                    <span className="text-xs font-semibold text-indigo-600">열 불러오는 중...</span>
                  )}
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  <label className="text-xs font-semibold text-slate-600">
                    X축 열
                    <select
                      value={xColumn}
                      onChange={(event) => setXColumn(event.target.value)}
                      disabled={columnsLoading || csvColumns.length === 0}
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm outline-none disabled:opacity-50"
                    >
                      <option value="">자동 인식</option>
                      {csvColumns.map((column) => (
                        <option key={column} value={column} disabled={column === yColumn}>
                          {column}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs font-semibold text-slate-600">
                    Y축 열
                    <select
                      value={yColumn}
                      onChange={(event) => setYColumn(event.target.value)}
                      disabled={columnsLoading || csvColumns.length === 0}
                      className="mt-1 w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm outline-none disabled:opacity-50"
                    >
                      <option value="">자동 인식</option>
                      {csvColumns.map((column) => (
                        <option key={column} value={column} disabled={column === xColumn}>
                          {column}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                {columnError && (
                  <p className="mt-3 rounded-xl bg-red-50 p-3 text-xs text-red-700">
                    CSV 열을 불러오지 못했습니다: {columnError}
                  </p>
                )}
              </div>
            )}

            <label className="block text-xs font-semibold text-slate-600">
              권한 단계
              <select
                value={permissionLevel}
                onChange={(event) =>
                  setPermissionLevel(Number(event.target.value) as AgentPermissionLevel)
                }
                className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm outline-none"
              >
                <option value={1}>Level 1 · 읽기 전용</option>
                <option value={2}>Level 2 · 새 파일 생성</option>
                <option value={3}>Level 3 · 승인 기반 수정·Python 실행</option>
              </select>
            </label>

            <details className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <summary className="cursor-pointer text-sm font-bold text-slate-700">
                고급 입력: 파일 생성·부분 수정·Python 코드
              </summary>
              <div className="mt-4 space-y-3">
                <label className="block text-xs font-semibold text-slate-600">
                  새 파일 내용
                  <textarea
                    value={content}
                    onChange={(event) => setContent(event.target.value)}
                    rows={4}
                    className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs outline-none"
                  />
                </label>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="text-xs font-semibold text-slate-600">
                    기존 문구
                    <textarea
                      value={oldText}
                      onChange={(event) => setOldText(event.target.value)}
                      rows={3}
                      className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs outline-none"
                    />
                  </label>
                  <label className="text-xs font-semibold text-slate-600">
                    새 문구
                    <textarea
                      value={newText}
                      onChange={(event) => setNewText(event.target.value)}
                      rows={3}
                      className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs outline-none"
                    />
                  </label>
                </div>
                <label className="block text-xs font-semibold text-slate-600">
                  실행할 Python 코드
                  <textarea
                    value={pythonCode}
                    onChange={(event) => setPythonCode(event.target.value)}
                    rows={8}
                    placeholder="허용 모듈: csv, json, math, statistics, numpy, pandas, matplotlib 등"
                    className="mt-1 w-full rounded-xl border border-slate-200 p-3 font-mono text-xs outline-none"
                  />
                </label>
              </div>
            </details>

            <button
              type="button"
              disabled={busy || !request.trim()}
              onClick={makePlan}
              className="w-full rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {busy ? "처리 중..." : "1. 작업 계획 생성"}
            </button>

            {error && (
              <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {error}
              </div>
            )}
          </section>

          <section className="space-y-5">
            <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
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
                <p className="mt-6 rounded-2xl bg-slate-50 p-6 text-center text-sm text-slate-500">
                  요청을 입력하고 작업 계획을 생성하세요.
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

            <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
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
      </div>
      {preview && <FilePreviewModal preview={preview} onClose={() => setPreview(null)} />}
    </main>
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
