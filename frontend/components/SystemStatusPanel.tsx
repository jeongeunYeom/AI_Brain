"use client";

import { useEffect, useState } from "react";
import { getSystemStatus, SystemStatus } from "@/lib/api";

function StatusBadge({ ok }: { ok: boolean }) {
  return <span className={`rounded-full px-2 py-1 text-xs font-semibold ${ok ? "bg-emerald-400 text-petroleum-950" : "bg-red-400 text-red-950"}`}>{ok ? "OK" : "Action needed"}</span>;
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
    <section className="rounded-2xl border border-petroleum-700 bg-petroleum-900/70 p-5 shadow-xl">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">System Status</h2>
          <p className="text-sm text-emerald-100/70">First-run checklist for Ollama, models, storage, and the knowledge base.</p>
        </div>
        <button onClick={refresh} className="rounded-lg border border-petroleum-400 px-3 py-2 text-sm font-semibold text-petroleum-300">
          {loading ? "Checking..." : "Refresh"}
        </button>
      </div>

      {error && <p className="mt-4 rounded-lg bg-red-950 p-3 text-sm text-red-100">{error}</p>}

      {status && (
        <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <div className="rounded-xl border border-petroleum-700 bg-petroleum-950 p-3">
            <div className="flex items-center justify-between"><span>Ollama</span><StatusBadge ok={status.checks.ollama.ok} /></div>
            <p className="mt-2 text-xs text-emerald-100/60">{status.checks.ollama.base_url}</p>
          </div>
          <div className="rounded-xl border border-petroleum-700 bg-petroleum-950 p-3">
            <div className="flex items-center justify-between"><span>Text model</span><StatusBadge ok={status.checks.text_model.ok} /></div>
            <p className="mt-2 text-xs text-emerald-100/60">{status.checks.text_model.model}</p>
          </div>
          <div className="rounded-xl border border-petroleum-700 bg-petroleum-950 p-3">
            <div className="flex items-center justify-between"><span>Vision model</span><StatusBadge ok={status.checks.vision_model.ok} /></div>
            <p className="mt-2 text-xs text-emerald-100/60">{status.checks.vision_model.model}</p>
          </div>
          <div className="rounded-xl border border-petroleum-700 bg-petroleum-950 p-3">
            <div className="flex items-center justify-between"><span>Documents</span><span className="text-lg font-bold text-petroleum-300">{status.knowledge_base.documents}</span></div>
            <p className="mt-2 text-xs text-emerald-100/60">metadata records</p>
          </div>
          <div className="rounded-xl border border-petroleum-700 bg-petroleum-950 p-3">
            <div className="flex items-center justify-between"><span>Vector chunks</span><span className="text-lg font-bold text-petroleum-300">{status.knowledge_base.chunks}</span></div>
            <p className="mt-2 text-xs text-emerald-100/60">ChromaDB collection count</p>
          </div>
        </div>
      )}
    </section>
  );
}
