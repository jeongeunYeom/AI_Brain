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
    <section className="rounded-2xl border border-petroleum-700 bg-petroleum-900/70 p-5 shadow-xl">
      <h2 className="text-xl font-semibold">Document Information</h2>
      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        {documents.map((document) => (
          <div key={document.document_id} className="rounded-xl border border-petroleum-700 bg-petroleum-950 p-4 text-sm">
            <p className="font-semibold text-petroleum-300">{document.title || document.filename}</p>
            <p className="mt-1 text-emerald-100/70">file: {document.filename}</p>
            <p className="mt-1 text-emerald-100/70">type: {document.document_type || "unknown"}</p>
            <p className="mt-1 text-emerald-100/70">pages: {document.pages} · chunks: {document.chunks}</p>
            <p className="mt-1 text-emerald-100/70">detected contents: {document.contents_pages?.length ? document.contents_pages.join(", ") : "not detected"}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
