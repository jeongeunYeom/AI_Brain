"use client";

import { useEffect, useState } from "react";
import { getSystemStatus, SystemStatus } from "@/lib/api";

function StatusBadge({ ok }: { ok: boolean }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${ok ? "bg-emerald-100 text-emerald-700" : "bg-red-100 text-red-700"}`}>
      {ok ? "OK" : "Check"}
    </span>
  );
}

function CheckRow({ label, value, ok }: { label: string; value: string | number; ok?: boolean }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold text-slate-700">{label}</span>
        {typeof ok === "boolean" ? <StatusBadge ok={ok} /> : null}
      </div>
      <p className="mt-1 truncate text-[11px] text-slate-500" title={String(value)}>{value}</p>
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
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-bold">System Status</h2>
          <p className="mt-1 text-[11px] leading-4 text-slate-500">Ollama, models, storage, and knowledge base.</p>
        </div>
        <button onClick={refresh} className="shrink-0 rounded-lg border border-slate-200 px-2 py-1 text-[11px] font-semibold text-indigo-600 hover:bg-indigo-50">
          {loading ? "Checking" : "Refresh"}
        </button>
      </div>

      {error && <p className="mt-3 rounded-lg bg-red-50 p-2 text-[11px] leading-4 text-red-600">{error}</p>}

      {status && (
        <div className="mt-3 grid grid-cols-1 gap-2 min-[360px]:grid-cols-2">
          <CheckRow label="Ollama" value={status.checks.ollama.base_url} ok={status.checks.ollama.ok} />
          <CheckRow label="Text model" value={status.checks.text_model.model} ok={status.checks.text_model.ok} />
          <CheckRow label="Vision model" value={status.checks.vision_model.model} ok={status.checks.vision_model.ok} />
          <CheckRow label="Documents" value={status.knowledge_base.documents} />
          <CheckRow label="Vector chunks" value={status.knowledge_base.chunks} />
        </div>
      )}
    </section>
  );
}
