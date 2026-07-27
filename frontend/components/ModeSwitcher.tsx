"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function ModeSwitcher() {
  const pathname = usePathname();
  const agentActive = pathname.startsWith("/agent");

  return (
    <nav className="fixed bottom-4 right-4 z-50 flex items-center gap-1 rounded-2xl border border-slate-200 bg-white/95 p-1 shadow-xl backdrop-blur">
      <Link
        href="/"
        className={`rounded-xl px-3 py-2 text-xs font-semibold transition ${
          !agentActive
            ? "bg-indigo-600 text-white"
            : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        RAG 질문
      </Link>
      <Link
        href="/agent"
        className={`rounded-xl px-3 py-2 text-xs font-semibold transition ${
          agentActive
            ? "bg-emerald-600 text-white"
            : "text-slate-600 hover:bg-slate-100"
        }`}
      >
        Agent 작업
      </Link>
    </nav>
  );
}
