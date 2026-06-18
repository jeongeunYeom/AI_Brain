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
  const [loading, setLoading] = useState(false);

  async function loadPlot() {
    setLoading(true);
    setError(null);
    try {
      const result = await createDemoPlot();
      setFigure(result.figure);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Plot request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <h2 className="text-sm font-bold text-slate-900">Engineering plot</h2>
          <p className="mt-1 text-xs text-slate-500">Generate a quick browser-rendered IPR demo plot.</p>
        </div>
        <button
          onClick={loadPlot}
          disabled={loading}
          className="rounded-full bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Loading..." : "Demo IPR Plot"}
        </button>
      </div>

      {error && <p className="mt-3 rounded-xl bg-red-50 p-3 text-xs text-red-600">{error}</p>}

      {figure && (
        <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 p-2 text-slate-900">
          <Plot
            data={figure.data ?? []}
            layout={{
              autosize: true,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 56, r: 20, t: 48, b: 48 },
              ...(figure.layout ?? {})
            }}
            className="h-[360px] w-full"
            useResizeHandler
          />
        </div>
      )}
    </section>
  );
}
