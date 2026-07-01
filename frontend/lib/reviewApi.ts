import { API_BASE } from "@/lib/api";

export type ReviewStatus =
  | "valid"
  | "review_required"
  | "failed"
  | "ignored";

export type ReviewDocumentSummary = {
  document_id: string;
  filename: string;
  counts: Record<ReviewStatus, number>;
  total: number;
  figure_note_count: number;
  preview_count: number;
};

export type ReviewSummary = {
  documents: ReviewDocumentSummary[];
  totals: Record<string, number>;
  chroma_changed: boolean;
  automatic_reindex: boolean;
};

export type ReviewCandidate = {
  candidate_id: string;
  candidate_filename: string;
  document_id: string;
  document_name: string;
  page_number: number | null;
  image_index: number | null;
  status: ReviewStatus;
  classification: string;
  confidence: number | null;
  title: string | null;
  image_type: string | null;
  analysis: string | null;
  x_axis: string | null;
  x_axis_unit: string | null;
  y_axis: string | null;
  y_axis_unit: string | null;
  trend_summary: string | null;
  engineering_meaning: string | null;
  series_descriptions: string[];
  reference_lines: string[];
  validation_errors: string[];
  manual_review_reasons: string[];
  schema_valid: boolean;
  information_quality_passed: boolean;
  apply_ready: boolean;
  needs_reindex: boolean;
  rotation: number | null;
  pdf_crop_rotation: number | null;
  enhance: boolean;
  asset_filename: string;
  original_url: string | null;
  preview_url: string | null;
  preview_source: string | null;
  updated_at: string | null;
  editable: boolean;
};

export type ReviewCandidateList = {
  items: ReviewCandidate[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
};

function ensureOk(response: Response): Promise<Response> {
  if (response.ok) {
    return Promise.resolve(response);
  }
  return response.text().then((body) => {
    throw new Error(body || `HTTP ${response.status}`);
  });
}

export function resolveReviewUrl(value: string): string {
  if (/^https?:\/\//i.test(value)) {
    return value;
  }
  const origin = API_BASE.replace(/\/api$/, "");
  return value.startsWith("/api/")
    ? `${origin}${value}`
    : `${API_BASE}/${value.replace(/^\//, "")}`;
}

export async function getReviewSummary(): Promise<ReviewSummary> {
  const response = await ensureOk(
    await fetch(`${API_BASE}/review/summary`, {
      cache: "no-store",
    }),
  );
  return response.json();
}

export async function getReviewCandidates(params: {
  documentId?: string;
  status?: string;
  page?: string;
  query?: string;
  offset?: number;
  limit?: number;
}): Promise<ReviewCandidateList> {
  const search = new URLSearchParams();
  if (params.documentId) {
    search.set("document_id", params.documentId);
  }
  if (params.status && params.status !== "all") {
    search.set("status", params.status);
  }
  if (params.page) {
    search.set("page", params.page);
  }
  if (params.query) {
    search.set("q", params.query);
  }
  search.set("offset", String(params.offset ?? 0));
  search.set("limit", String(params.limit ?? 20));

  const response = await ensureOk(
    await fetch(
      `${API_BASE}/review/candidates?${search.toString()}`,
      { cache: "no-store" },
    ),
  );
  return response.json();
}

export async function updateReviewCandidate(
  candidateId: string,
  payload: Partial<ReviewCandidate> & {
    series_descriptions?: string[];
    reference_lines?: string[];
  },
): Promise<ReviewCandidate> {
  const response = await ensureOk(
    await fetch(
      `${API_BASE}/review/candidates/${candidateId}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    ),
  );
  return response.json();
}

export async function updateReviewRotation(
  candidateId: string,
  payload: {
    rotation: number | null;
    pdf_crop_rotation: number | null;
    enhance: boolean;
    regenerate: boolean;
  },
): Promise<ReviewCandidate> {
  const response = await ensureOk(
    await fetch(
      `${API_BASE}/review/candidates/${candidateId}/rotation`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    ),
  );
  return response.json();
}

export async function regenerateReviewPreview(
  candidateId: string,
): Promise<{
  preview_url: string;
  preview_source: string;
  rotation_applied: number;
}> {
  const response = await ensureOk(
    await fetch(
      `${API_BASE}/review/candidates/${candidateId}/preview`,
      { method: "POST" },
    ),
  );
  return response.json();
}
