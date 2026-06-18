"use client";

import { FormEvent, useState } from "react";
import { analyzeImage, askQuestion, ChatResponse, createUploadJob, getUploadJob, uploadDocument, UploadJob } from "@/lib/api";
import { DocumentInfoPanel } from "@/components/DocumentInfoPanel";
import { PlotPanel } from "@/components/PlotPanel";
import { SystemStatusPanel } from "@/components/SystemStatusPanel";

export default function Home() {
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<ChatResponse | null>(null);
  const [vision, setVision] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploadJob, setUploadJob] = useState<UploadJob | null>(null);
  const [expandedSource, setExpandedSource] = useState<string | null>(null);
  const [sidebarWidth, setSidebarWidth] = useState(300);

  async function onUpload() {
    if (!documentFile) return;
    setBusy(true);
    setStatus("Uploading and indexing document...");
    setUploadJob(null);
    let poller: ReturnType<typeof setInterval> | null = null;
    try {
      const job = await createUploadJob();
      poller = setInterval(async () => {
        try {
          setUploadJob(await getUploadJob(job.job_id));
        } catch {
          // Keep upload running even if one polling request fails.
        }
      }, 1000);
      const result = await uploadDocument(documentFile, true, job.job_id);
      const finalJob = await getUploadJob(job.job_id);
      setUploadJob(finalJob);
      setStatus(result.skipped ? "Already processed; reused persistent vector store." : `Processed ${result.document.chunks} chunks.`);
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Upload failed");
    } finally {
      if (poller) clearInterval(poller);
      setBusy(false);
    }
  }

  async function onAsk(event: FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    setBusy(true);
    setStatus("Searching vector database and asking Qwen3...");
    try {
      const result = await askQuestion(question);
      setAnswer(result);
      setStatus("Answer generated from retrieved sources.");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Question failed");
    } finally {
      setBusy(false);
    }
  }

  async function onAnalyzeImage() {
    if (!imageFile) return;
    setBusy(true);
    setStatus("Analyzing graph image with Qwen2.5-VL...");
    try {
      const result = await analyzeImage(imageFile);
      setVision(result.analysis);
      setStatus("Vision analysis saved to figure notes.");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Vision analysis failed");
    } finally {
      setBusy(false);
    }
  }

  const sourceCount = answer?.sources?.length ?? 0;

  function startSidebarResize() {
    const onMouseMove = (event: MouseEvent) => {
      const nextWidth = Math.min(520, Math.max(248, event.clientX));
      setSidebarWidth(nextWidth);
    };

    const onMouseUp = () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  }

  return (
    <main className="flex h-screen overflow-hidden bg-[#eef1f6] text-slate-900">
      <aside className="relative hidden shrink-0 border-r border-slate-200 bg-[#f8fafc] md:flex md:flex-col" style={{ width: sidebarWidth }}>
        <div className="flex h-14 items-center gap-3 border-b border-slate-200 px-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-600 text-sm font-bold text-white">A</div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-bold">AI_Brain</h1>
            <p className="truncate text-[11px] text-slate-500">Petroleum RAG Agent</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-4">
          <button className="mb-4 flex w-full items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium shadow-sm hover:bg-slate-50">
            <span>＋</span>
            New chat
          </button>

          <div className="space-y-5">
            <section>
              <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Recent Chats</p>
              <div className="space-y-1 text-sm">
                <button className="w-full truncate rounded-lg bg-indigo-50 px-3 py-2 text-left font-medium text-indigo-700">💬 Ask from uploaded PDFs</button>
                <button className="w-full truncate rounded-lg px-3 py-2 text-left text-slate-600 hover:bg-white">💬 Bottomhole pressure</button>
                <button className="w-full truncate rounded-lg px-3 py-2 text-left text-slate-600 hover:bg-white">💬 Formation pressure</button>
                <button className="w-full truncate rounded-lg px-3 py-2 text-left text-slate-600 hover:bg-white">💬 Kick analysis</button>
              </div>
            </section>

            <section>
              <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Documents</p>
              <div className="mb-2 rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
                <input
                  className="w-full bg-transparent text-xs outline-none placeholder:text-slate-400"
                  placeholder="Search documents..."
                  disabled
                />
              </div>
              <DocumentInfoPanel />
            </section>

            <section>
              <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Tools</p>
              <div className="space-y-2">
                <label className="block cursor-pointer rounded-xl border border-slate-200 bg-white p-3 text-sm shadow-sm hover:border-indigo-200">
                  <input
                    className="hidden"
                    type="file"
                    accept=".pdf,.txt,.png,.jpg,.jpeg,.ppt,.pptx"
                    onChange={(event) => setDocumentFile(event.target.files?.[0] ?? null)}
                  />
                  <span className="font-semibold">Upload document</span>
                  <span className="mt-1 block truncate text-xs text-slate-500">{documentFile?.name ?? "PDF, TXT, PPT, image"}</span>
                </label>
                <button
                  disabled={!documentFile || busy}
                  onClick={onUpload}
                  className="w-full rounded-xl bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Upload · Extract · Embed
                </button>

                <label className="block cursor-pointer rounded-xl border border-slate-200 bg-white p-3 text-sm shadow-sm hover:border-indigo-200">
                  <input
                    className="hidden"
                    type="file"
                    accept=".png,.jpg,.jpeg"
                    onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
                  />
                  <span className="font-semibold">Graph/Image analysis</span>
                  <span className="mt-1 block truncate text-xs text-slate-500">{imageFile?.name ?? "PNG or JPG graph"}</span>
                </label>
                <button
                  disabled={!imageFile || busy}
                  onClick={onAnalyzeImage}
                  className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 shadow-sm disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Analyze Image
                </button>
              </div>
            </section>

            <section>
              <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">System</p>
              <SystemStatusPanel />
            </section>
          </div>
        </div>

        <div className="border-t border-slate-200 p-3">
          <div className="rounded-xl bg-white p-3 text-xs text-slate-500 shadow-sm">
            <p className="font-semibold text-slate-700">Local-first mode</p>
            <p className="mt-1">Ollama · ChromaDB · Source-grounded answers</p>
          </div>
        </div>

        <button
          type="button"
          aria-label="Resize sidebar"
          onMouseDown={startSidebarResize}
          className="absolute -right-1 top-0 z-20 hidden h-full w-2 cursor-col-resize bg-transparent transition hover:bg-indigo-200/70 md:block"
        >
          <span className="mx-auto mt-16 block h-12 w-1 rounded-full bg-slate-300" />
        </button>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-4 md:px-6">
          <div className="flex items-center gap-3">
            <button className="rounded-lg border border-slate-200 px-2 py-1 text-sm md:hidden">☰</button>
            <div>
              <h2 className="text-sm font-bold">AI_Brain 7.0 ▾</h2>
              <p className="hidden text-xs text-slate-500 sm:block">Citation-grounded Q&A with local knowledge base</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-slate-500">
            <span className="rounded-full border border-slate-200 px-3 py-1 text-xs">Qwen3 8B</span>
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-100">☾</span>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto bg-[#f8fafc] px-4 py-6">
          <div className="mx-auto flex max-w-4xl flex-col gap-5">
            <div className="flex justify-end">
              <div className="max-w-[78%] rounded-3xl rounded-br-md bg-white px-4 py-3 text-sm text-slate-700 shadow-sm ring-1 ring-slate-200">
                {question || "PDF나 교재 내용을 기반으로 궁금한 내용을 질문해봐."}
              </div>
            </div>

            <div className="flex gap-3">
              <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">AI</div>
              <div className="max-w-[86%] rounded-3xl rounded-tl-md bg-white px-5 py-4 text-sm leading-7 text-slate-700 shadow-sm ring-1 ring-slate-200">
                {answer ? (
                  <div className="whitespace-pre-wrap">{answer.answer}</div>
                ) : (
                  <div>
                    <p className="font-semibold text-slate-900">AI_Brain이 준비됐어.</p>
                    <p className="mt-2">
                      왼쪽에서 PDF/TXT/PPT 자료를 업로드한 뒤, 아래 입력창에 질문하면 검색된 문서 근거를 바탕으로 답변해.
                    </p>
                  </div>
                )}

                {sourceCount > 0 && (
                  <div className="mt-5 border-t border-slate-100 pt-4">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Sources</h3>
                    <ul className="mt-2 space-y-2">
                      {answer?.sources.map((source) => (
                        <li key={source.chunk_id} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                          <button
                            className="w-full text-left"
                            onClick={() => setExpandedSource(expandedSource === source.chunk_id ? null : source.chunk_id)}
                          >
                            <span className="font-semibold text-slate-800">{source.document}</span>
                            {source.page ? <span className="ml-1 text-slate-500">p.{source.page}</span> : null}
                            <span className="ml-2 text-xs text-indigo-600">score {source.score?.toFixed(3) ?? "n/a"}</span>
                            <span className="block text-xs text-slate-400">chunk {source.chunk_id}</span>
                          </button>
                          <p className="mt-2 text-xs leading-5 text-slate-500">{source.preview ?? source.excerpt}</p>
                          {expandedSource === source.chunk_id && (
                            <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-xl bg-white p-3 text-xs leading-5 text-slate-700 ring-1 ring-slate-200">
                              {source.excerpt}
                            </pre>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {uploadJob && (
              <div className="mx-auto w-full max-w-3xl rounded-2xl border border-slate-200 bg-white p-4 text-sm shadow-sm">
                <div className="flex items-center justify-between text-slate-700">
                  <span>{uploadJob.status === "failed" ? "❌" : uploadJob.status === "completed" ? "✅" : "⏳"} {uploadJob.message}</span>
                  <span>{uploadJob.step}/{uploadJob.total_steps}</span>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full bg-indigo-600" style={{ width: `${Math.min(100, (uploadJob.step / uploadJob.total_steps) * 100)}%` }} />
                </div>
                {uploadJob.error && <p className="mt-2 text-red-500">{uploadJob.error}</p>}
              </div>
            )}

            {vision && (
              <div className="flex gap-3">
                <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-violet-600 text-xs font-bold text-white">V</div>
                <div className="max-w-[86%] whitespace-pre-wrap rounded-3xl rounded-tl-md bg-white px-5 py-4 text-sm leading-7 text-slate-700 shadow-sm ring-1 ring-slate-200">
                  {vision}
                </div>
              </div>
            )}

            {status && (
              <div className="mx-auto rounded-full border border-slate-200 bg-white px-4 py-2 text-xs text-slate-500 shadow-sm">
                {busy ? "⏳ " : "✅ "}{status}
              </div>
            )}

            <details className="rounded-2xl border border-slate-200 bg-white p-4 text-sm shadow-sm">
              <summary className="cursor-pointer font-semibold text-slate-700">Plot panel</summary>
              <div className="mt-4">
                <PlotPanel />
              </div>
            </details>
          </div>
        </div>

        <form onSubmit={onAsk} className="shrink-0 border-t border-slate-200 bg-white px-4 py-4">
          <div className="mx-auto flex max-w-4xl items-end gap-2 rounded-3xl border border-slate-200 bg-white px-4 py-3 shadow-lg shadow-slate-200/70">
            <span className="pb-2 text-slate-400">＋</span>
            <textarea
              className="max-h-40 min-h-[36px] flex-1 resize-none bg-transparent py-2 text-sm outline-none placeholder:text-slate-400"
              placeholder="Message AI_Brain..."
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <button
              disabled={busy || !question.trim()}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white shadow-sm transition disabled:cursor-not-allowed disabled:opacity-40"
              aria-label="Send message"
            >
              ↑
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-slate-400">AI_Brain can make mistakes. Check retrieved sources and page numbers.</p>
        </form>
      </section>
    </main>
  );
}
