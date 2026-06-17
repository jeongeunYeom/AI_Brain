import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Petroleum Engineering AI Agent",
  description: "Local petroleum engineering RAG assistant with Ollama, ChromaDB, and Plotly"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
