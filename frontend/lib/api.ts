export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000/api";

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

export async function getSystemStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE}/system/checklist`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export type UploadJob = {
  job_id: string;
  step: number;
  total_steps: number;
  message: string;
  status: "running" | "completed" | "failed";
  error?: string | null;
};

export async function createUploadJob(): Promise<{ job_id: string }> {
  const response = await fetch(`${API_BASE}/jobs`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function getUploadJob(jobId: string): Promise<UploadJob> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function uploadDocument(file: File, analyzeFigures: boolean, jobId?: string) {
  const form = new FormData();
  form.append("file", file);
  const suffix = jobId ? `&job_id=${jobId}` : "";
  const response = await fetch(`${API_BASE}/documents/upload?analyze_figures=${analyzeFigures}${suffix}`, {
    method: "POST",
    body: form
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function askQuestion(
  question: string,
  model = "qwen3:8b",
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, model })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function askComparison(
  question: string,
  models = ["qwen3:8b", "gemma4:latest"],
): Promise<ChatCompareResponse> {
  const response = await fetch(`${API_BASE}/chat/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, models })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function analyzeImage(file: File) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/vision/analyze`, { method: "POST", body: form });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function createDemoPlot() {
  const response = await fetch(`${API_BASE}/plots`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: "Example IPR Curve",
      x_label: "Flow rate, q (STB/day)",
      y_label: "Bottomhole pressure, Pwf (psi)",
      x: [0, 200, 400, 600, 800, 1000],
      y: [3200, 2950, 2600, 2100, 1450, 700],
      chart_type: "line"
    })
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
