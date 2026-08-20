"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/", icon: "⌂", label: "RAG 채팅", match: (path: string) => path === "/" },
  { href: "/agent", icon: "✦", label: "Agent 작업", match: (path: string) => path.startsWith("/agent") },
  { href: "/review", icon: "▧", label: "Figure Review", match: (path: string) => path.startsWith("/review") },
  { href: "/evaluation", icon: "◫", label: "평가", match: (path: string) => path.startsWith("/evaluation") },
];

export function AppIconRail() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-40 hidden w-[72px] flex-col items-center border-r border-slate-200 bg-white py-4 md:flex">
      <Link
        href="/"
        className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-sm font-black text-white shadow-sm"
        aria-label="Petroleum RAG Agent 홈"
      >
        AI
      </Link>
      <nav className="mt-6 flex flex-1 flex-col gap-2" aria-label="주요 메뉴">
        {NAV_ITEMS.map((item) => {
          const active = item.match(pathname);
          return (
            <Link
              key={item.href}
              href={item.href}
              title={item.label}
              aria-current={active ? "page" : undefined}
              className={`flex h-10 w-10 items-center justify-center rounded-xl text-lg transition ${
                active
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-slate-500 hover:bg-slate-100 hover:text-slate-900"
              }`}
            >
              {item.icon}
            </Link>
          );
        })}
      </nav>
      <span className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-900 text-xs font-bold text-white">
        JE
      </span>
    </aside>
  );
}

export function MobileModeTabs() {
  const pathname = usePathname();

  return (
    <nav className="flex items-center rounded-xl bg-slate-100 p-1 md:hidden" aria-label="작업 모드">
      {NAV_ITEMS.slice(0, 2).map((item) => {
        const active = item.match(pathname);
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={`rounded-lg px-3 py-1.5 text-xs font-bold transition ${
              active ? "bg-white text-indigo-700 shadow-sm" : "text-slate-500"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}
