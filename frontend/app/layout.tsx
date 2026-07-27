import type { Metadata } from "next";
import { ModeSwitcher } from "@/components/ModeSwitcher";
import "./globals.css";

export const metadata: Metadata = {
  title: "Petroleum Engineering AI Agent",
  description: "Local petroleum engineering RAG and approved workspace Agent",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>
        {children}
        <ModeSwitcher />
      </body>
    </html>
  );
}
