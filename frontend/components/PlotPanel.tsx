"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import { createDemoPlot } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type PlotFigure = {
  data?: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
};

export function PlotPanel() {
  const [figure, setFigure] = useState<PlotFigure | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function loadPlot() {
    setError(null);
    try {
      const result = await createDemoPlot();
      setFigure(result.figure);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Plot request failed");
    }
  }

  return (
    <section className="rounded-2xl border border-petroleum-700 bg-petroleum-900/70 p-5 shadow-xl">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Interactive Plotly Graph</h2>
          <p className="text-sm text-emerald-100/70">Generate browser-rendered engineering plots from backend payloads.</p>
        </div>
        <button onClick={loadPlot} className="rounded-lg bg-petroleum-400 px-4 py-2 font-semibold text-petroleum-950">
          Demo IPR Plot
        </button>
      </div>
      {error && <p className="mt-4 rounded-lg bg-red-950 p-3 text-sm text-red-100">{error}</p>}
      {figure && (
        <div className="mt-4 overflow-hidden rounded-xl bg-white p-2 text-slate-900">
          <Plot data={figure.data ?? []} layout={{ autosize: true, ...(figure.layout ?? {}) }} className="h-[420px] w-full" useResizeHandler />
        </div>
      )}
    </section>
  );
}
