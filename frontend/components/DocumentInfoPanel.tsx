"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

type DocumentRecord = {
  document_id: string;
  filename: string;
  pages: number;
  chunks: number;
  title?: string | null;
  document_type?: string | null;
  contents_pages?: number[];
};

export function DocumentInfoPanel() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);

  async function refresh() {
    const response = await fetch(`${API_BASE}/documents`, { cache: "no-store" });
    if (response.ok) setDocuments(await response.json());
  }

  useEffect(() => {
    void refresh();
  }, []);

  if (!documents.length) return null;

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3 text-slate-900 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-bold">Document Information</h2>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-500">{documents.length}</span>
      </div>

      <div className="mt-3 max-h-72 space-y-2 overflow-y-auto pr-1">
        {documents.map((document) => (
          <article key={document.document_id} className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-[11px] leading-4">
            <p className="truncate font-semibold text-indigo-700" title={document.title || document.filename}>{document.title || document.filename}</p>
            <p className="mt-1 truncate text-slate-500" title={document.filename}>file: {document.filename}</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              <span className="rounded-full bg-white px-2 py-0.5 text-slate-500 ring-1 ring-slate-200">{document.document_type || "unknown"}</span>
              <span className="rounded-full bg-white px-2 py-0.5 text-slate-500 ring-1 ring-slate-200">{document.pages} pages</span>
              <span className="rounded-full bg-white px-2 py-0.5 text-slate-500 ring-1 ring-slate-200">{document.chunks} chunks</span>
            </div>
            <p className="mt-2 truncate text-slate-400" title={document.contents_pages?.join(", ")}>contents: {document.contents_pages?.length ? document.contents_pages.join(", ") : "not detected"}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
