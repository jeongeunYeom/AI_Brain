import { API_BASE, requestJson } from "./http";

export { API_BASE };

export type Source = {
  document: string;
  page: number | null;
  chunk_id: string;
  score: number | null;
  vector_score?: number | null;
  keyword_score?: number | null;
  excerpt: string;
  preview?: string | null;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  query_type?: string | null;
  model?: string | null;
  elapsed_seconds?: number | null;
};

export type ModelAnswer = {
  model: string;
  answer: string;
  elapsed_seconds: number;
};

export type ChatCompareResponse = {
  answers: ModelAnswer[];
  sources: Source[];
  query_type?: string | null;
  figures?: unknown[];
  retrieval_elapsed_seconds: number;
  shared_context: boolean;
};

export type SystemCheck = {
  ok: boolean;
  message?: string | null;
  model?: string;
  path?: string;
  base_url?: string;
};

export type SystemStatus = {
  ok: boolean;
  checks: {
    ollama: SystemCheck;
    text_model: SystemCheck;
    vision_model: SystemCheck;
    data_dir: SystemCheck;
    chroma: SystemCheck;
  };
  models: string[];
  knowledge_base: {
    documents: number;
    chunks: number;
  };
};

export type UploadJob = {
  job_id: string;
  step: number;
  total_steps: number;
  message: string;
  status: "running" | "completed" | "failed";
  error?: string | null;
};

export function getSystemStatus(): Promise<SystemStatus> {
  return requestJson("/system/checklist", { cache: "no-store" });
}

export function createUploadJob(): Promise<{ job_id: string }> {
  return requestJson("/jobs", { method: "POST" });
}

export function getUploadJob(jobId: string): Promise<UploadJob> {
  return requestJson(`/jobs/${jobId}`, { cache: "no-store" });
}

export function uploadDocument(
  file: File,
  analyzeFigures: boolean,
  jobId?: string,
) {
  const form = new FormData();
  form.append("file", file);

  const query = new URLSearchParams({
    analyze_figures: String(analyzeFigures),
  });
  if (jobId) query.set("job_id", jobId);

  return requestJson(`/documents/upload?${query}`, {
    method: "POST",
    body: form,
  });
}

export function askQuestion(
  question: string,
  model = "qwen3:8b",
): Promise<ChatResponse> {
  return requestJson("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, model }),
  });
}

export function askComparison(
  question: string,
  models = ["qwen3:8b", "gemma4:latest"],
): Promise<ChatCompareResponse> {
  return requestJson("/chat/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, models }),
  });
}

export function analyzeImage(file: File) {
  const form = new FormData();
  form.append("file", file);

  return requestJson("/vision/analyze", {
    method: "POST",
    body: form,
  });
}

export function createDemoPlot() {
  return requestJson("/plots", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: "Example IPR Curve",
      x_label: "Flow rate, q (STB/day)",
      y_label: "Bottomhole pressure, Pwf (psi)",
      x: [0, 200, 400, 600, 800, 1000],
      y: [3200, 2950, 2600, 2100, 1450, 700],
      chart_type: "line",
    }),
  });
}
