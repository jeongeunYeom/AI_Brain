"use client";

import { useEffect, useState } from "react";
import { getSystemStatus, SystemStatus } from "@/lib/api";

function Dot({ ok }: { ok: boolean }) {
  return <span className={`h-2.5 w-2.5 rounded-full ${ok ? "bg-emerald-500" : "bg-red-500"}`} />;
}

function SystemRow({ label, value, ok }: { label: string; value: string | number; ok?: boolean }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2">
      <div className="flex min-w-0 items-center gap-2">
        {typeof ok === "boolean" ? <Dot ok={ok} /> : null}
        <span className="shrink-0 text-xs font-semibold text-slate-700">{label}</span>
      </div>
      <span className="truncate text-right text-[11px] text-slate-500" title={String(value)}>{value}</span>
    </div>
  );
}

export function SystemStatusPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      setStatus(await getSystemStatus());
    } catch (err) {
      setError(err instanceof Error ? err.message : "System status request failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3 text-slate-900 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-bold">System</h2>
          <p className="mt-0.5 truncate text-[11px] text-slate-500">Ollama · models · ChromaDB</p>
        </div>
        <button onClick={refresh} className="shrink-0 rounded-lg border border-slate-200 px-2 py-1 text-[11px] font-semibold text-indigo-600 hover:bg-indigo-50">
          {loading ? "..." : "Refresh"}
        </button>
      </div>

      {error && <p className="mt-3 rounded-lg bg-red-50 p-2 text-[11px] leading-4 text-red-600">{error}</p>}

      {status && (
        <div className="mt-3 space-y-2">
          <SystemRow label="Ollama" value={status.checks.ollama.base_url} ok={status.checks.ollama.ok} />
          <SystemRow label="Text" value={status.checks.text_model.model} ok={status.checks.text_model.ok} />
          <SystemRow label="Vision" value={status.checks.vision_model.model} ok={status.checks.vision_model.ok} />
          <div className="grid grid-cols-2 gap-2">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <p className="text-[11px] font-semibold text-slate-500">Documents</p>
              <p className="mt-1 text-lg font-bold text-slate-900">{status.knowledge_base.documents}</p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <p className="text-[11px] font-semibold text-slate-500">Chunks</p>
              <p className="mt-1 truncate text-lg font-bold text-slate-900" title={String(status.knowledge_base.chunks)}>{status.knowledge_base.chunks}</p>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
