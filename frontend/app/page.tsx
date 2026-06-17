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

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,#1f4d3f,transparent_35%),#071311] px-6 py-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <header className="rounded-3xl border border-petroleum-700 bg-petroleum-900/80 p-8 shadow-2xl">
          <p className="text-sm uppercase tracking-[0.4em] text-petroleum-300">Local-first RAG Agent</p>
          <h1 className="mt-3 text-4xl font-bold md:text-6xl">Petroleum Engineering AI Agent</h1>
          <p className="mt-4 max-w-3xl text-emerald-50/75">
            Upload textbooks, handbooks, PDFs, and graph images once. The backend extracts text and figures,
            persists ChromaDB vectors to disk, and answers with source documents and page numbers.
          </p>
        </header>

        <SystemStatusPanel />
        <DocumentInfoPanel />

        {uploadJob && (
          <div className="rounded-xl border border-petroleum-400/40 bg-petroleum-950 p-4 text-sm text-emerald-100">
            <div className="flex items-center justify-between">
              <span>{uploadJob.status === "failed" ? "❌" : uploadJob.status === "completed" ? "✅" : "⏳"} {uploadJob.message}</span>
              <span>{uploadJob.step}/{uploadJob.total_steps}</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-petroleum-700">
              <div className="h-full bg-petroleum-400" style={{ width: `${Math.min(100, (uploadJob.step / uploadJob.total_steps) * 100)}%` }} />
            </div>
            {uploadJob.error && <p className="mt-2 text-red-200">{uploadJob.error}</p>}
          </div>
        )}

        {status && <div className="rounded-xl border border-petroleum-400/40 bg-petroleum-950 p-4 text-sm text-emerald-100">{busy ? "⏳ " : "✅ "}{status}</div>}

        <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <section className="rounded-2xl border border-petroleum-700 bg-petroleum-900/70 p-5 shadow-xl">
            <h2 className="text-xl font-semibold">Knowledge Base Upload</h2>
            <p className="mt-2 text-sm text-emerald-100/70">PDF/TXT/image inputs are hashed and processed only once.</p>
            <input
              className="mt-4 w-full rounded-lg border border-petroleum-700 bg-petroleum-950 p-3 text-sm"
              type="file"
              accept=".pdf,.txt,.png,.jpg,.jpeg,.ppt,.pptx"
              onChange={(event) => setDocumentFile(event.target.files?.[0] ?? null)}
            />
            <button disabled={!documentFile || busy} onClick={onUpload} className="mt-4 w-full rounded-lg bg-petroleum-400 px-4 py-3 font-semibold text-petroleum-950 disabled:opacity-40">
              Upload, Extract, Embed
            </button>
          </section>

          <section className="rounded-2xl border border-petroleum-700 bg-petroleum-900/70 p-5 shadow-xl">
            <h2 className="text-xl font-semibold">Citation-grounded Q&A</h2>
            <form onSubmit={onAsk} className="mt-4 flex flex-col gap-3">
              <textarea
                className="min-h-28 rounded-lg border border-petroleum-700 bg-petroleum-950 p-3 text-sm outline-none focus:border-petroleum-300"
                placeholder="예: Darcy 법칙에서 permeability와 pressure gradient의 관계를 설명해줘."
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
              />
              <button disabled={busy} className="rounded-lg bg-petroleum-400 px-4 py-3 font-semibold text-petroleum-950 disabled:opacity-40">
                Ask Qwen3 8B
              </button>
            </form>
            {answer && (
              <div className="mt-5 space-y-4">
                <div className="whitespace-pre-wrap rounded-xl bg-petroleum-950 p-4 text-emerald-50">{answer.answer}</div>
                <div>
                  <h3 className="font-semibold">Sources</h3>
                  <ul className="mt-2 space-y-2 text-sm text-emerald-100/80">
                    {answer.sources.map((source) => (
                      <li key={source.chunk_id} className="rounded-lg border border-petroleum-700 p-3">
                        <button className="w-full text-left" onClick={() => setExpandedSource(expandedSource === source.chunk_id ? null : source.chunk_id)}>
                          <span className="font-semibold">{source.document}</span> {source.page ? `p.${source.page}` : ""}
                          <span className="ml-2 text-petroleum-300">score {source.score?.toFixed(3) ?? "n/a"}</span>
                          <span className="ml-2 text-emerald-100/50">vector {source.vector_score?.toFixed(3) ?? "0.000"} · keyword {source.keyword_score?.toFixed(3) ?? "0.000"}</span>
                          <span className="block text-xs text-emerald-100/50">chunk {source.chunk_id}</span>
                        </button>
                        <p className="mt-2 text-emerald-100/60">{source.preview ?? source.excerpt}</p>
                        {expandedSource === source.chunk_id && (
                          <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded-lg bg-petroleum-950 p-3 text-xs text-emerald-50">{source.excerpt}</pre>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </section>
        </div>

        <section className="rounded-2xl border border-petroleum-700 bg-petroleum-900/70 p-5 shadow-xl">
          <h2 className="text-xl font-semibold">Graph/Image Understanding</h2>
          <p className="mt-2 text-sm text-emerald-100/70">Upload PNG/JPG graphs for axis, unit, legend, trend, and numeric-value analysis.</p>
          <div className="mt-4 flex flex-col gap-3 md:flex-row">
            <input className="flex-1 rounded-lg border border-petroleum-700 bg-petroleum-950 p-3 text-sm" type="file" accept=".png,.jpg,.jpeg" onChange={(event) => setImageFile(event.target.files?.[0] ?? null)} />
            <button disabled={!imageFile || busy} onClick={onAnalyzeImage} className="rounded-lg bg-petroleum-400 px-5 py-3 font-semibold text-petroleum-950 disabled:opacity-40">
              Analyze Image
            </button>
          </div>
          {vision && <div className="mt-4 whitespace-pre-wrap rounded-xl bg-petroleum-950 p-4 text-emerald-50">{vision}</div>}
        </section>

        <PlotPanel />
      </div>
    </main>
  );
}
