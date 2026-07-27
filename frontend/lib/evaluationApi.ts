import { requestJson } from "./http";

export type BenchmarkSummary = {
  questions_total?: number;
  questions_completed?: number;
  infrastructure_errors?: number;
  initial_benchmark_pass_rate?: number | null;
  final_benchmark_pass_rate?: number | null;
  initial_validator_pass_rate?: number | null;
  final_validator_pass_rate?: number | null;
  rewrite_success_rate?: number | null;
  validator_detection_rate?: number | null;
  validator_false_positive_rate?: number | null;
  exact_refusal_rate?: number | null;
  preferred_page_hit_rate?: number | null;
  expected_document_hit_rate?: number | null;
  average_attempts?: number | null;
  average_retrieval_seconds?: number | null;
  average_generation_seconds?: number | null;
  average_total_seconds?: number | null;
  category_metrics?: Record<
    string,
    {
      total?: number;
      final_passed?: number;
      final_pass_rate?: number | null;
    }
  >;
};

export type BenchmarkAttempt = {
  attempt?: number | null;
  answer?: string | null;
  elapsed_seconds?: number | null;
  validation_passed?: boolean | null;
  errors?: string[];
  warnings?: string[];
  rule_ids?: string[];
};

export type BenchmarkResult = {
  id: string;
  category?: string | null;
  question?: string | null;
  model?: string | null;
  expected_behavior?: string | null;
  initial_validator_passed?: boolean | null;
  final_validator_passed?: boolean | null;
  initial_benchmark_passed?: boolean | null;
  final_benchmark_passed?: boolean | null;
  rewrite_success?: boolean | null;
  attempts?: number | null;
  final_status?: string | null;
  expected_document_hit?: boolean | null;
  preferred_page_hit?: boolean | null;
  source_pages?: number[];
  retrieval_seconds?: number | null;
  generation_seconds?: number | null;
  total_seconds?: number | null;
  initial_required_failures?: string[];
  initial_forbidden_hits?: string[];
  final_required_failures?: string[];
  final_forbidden_hits?: string[];
  initial_answer?: string | null;
  final_answer?: string | null;
  infrastructure_error?: string | null;
  attempt_details?: BenchmarkAttempt[];
};

export type BenchmarkRun = {
  benchmark_name?: string | null;
  benchmark_file?: string | null;
  run_id: string;
  created_at?: string | null;
  model?: string | null;
  api_url?: string | null;
  question_count?: number | null;
  wall_clock_seconds?: number | null;
  summary: BenchmarkSummary;
  results: BenchmarkResult[];
};

export type BenchmarkRunListItem = Pick<
  BenchmarkRun,
  | "run_id"
  | "created_at"
  | "model"
  | "question_count"
  | "wall_clock_seconds"
  | "summary"
>;

export function getBenchmarkRuns(
  limit = 50,
): Promise<BenchmarkRunListItem[]> {
  return requestJson(`/evaluation/runs?limit=${limit}`, {
    cache: "no-store",
  });
}

export function getLatestBenchmarkRun(): Promise<BenchmarkRun> {
  return requestJson("/evaluation/latest", { cache: "no-store" });
}

export function getBenchmarkRun(runId: string): Promise<BenchmarkRun> {
  return requestJson(
    `/evaluation/runs/${encodeURIComponent(runId)}`,
    { cache: "no-store" },
  );
}
