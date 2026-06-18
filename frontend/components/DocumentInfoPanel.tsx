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

function cleanTitle(document: DocumentRecord) {
  return (document.title || document.filename).replace(/\.[^/.]+$/, "");
}

export function DocumentInfoPanel() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);

  async function refresh() {
    const response = await fetch(`${API_BASE}/documents`, { cache: "no-store" });
    if (response.ok) setDocuments(await response.json());
  }

  useEffect(() => {
    void refresh();
  }, []);

  if (!documents.length) {
    return (
      <div className="rounded-2xl border border-dashed border-slate-200 bg-white p-3 text-xs text-slate-500 shadow-sm">
        No indexed documents yet.
      </div>
    );
  }

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-3 text-slate-900 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-bold">Recent Documents</h2>
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-500">{documents.length}</span>
      </div>

      <div className="mt-3 max-h-80 space-y-1.5 overflow-y-auto pr-1">
        {documents.slice(0, 20).map((document) => (
          <article key={document.document_id} className="rounded-xl px-2 py-2 text-xs hover:bg-slate-50">
            <div className="flex min-w-0 items-start gap-2">
              <span className="mt-0.5 shrink-0">📄</span>
              <div className="min-w-0 flex-1">
                <p className="truncate font-semibold text-slate-800" title={cleanTitle(document)}>{cleanTitle(document)}</p>
                <p className="mt-0.5 truncate text-[11px] text-slate-500" title={document.filename}>{document.pages} pages · {document.chunks} chunks</p>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
