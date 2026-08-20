"use client";

import { useEffect, useMemo, useState } from "react";
import { AppIconRail } from "@/components/AppNavigation";
import type {
  BenchmarkResult,
  BenchmarkRun,
  BenchmarkRunListItem,
  BenchmarkSummary,
} from "@/lib/evaluationApi";
import {
  getBenchmarkRun,
  getBenchmarkRuns,
  getLatestBenchmarkRun,
} from "@/lib/evaluationApi";

type ResultFilter = "all" | "passed" | "rewritten" | "failed";

function percentage(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(value === 1 || value === 0 ? 0 : 1)}%`;
}

function seconds(value?: number | null): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(2)}s`;
}

function firstNumber(
  ...values: Array<number | null | undefined>
): number | null | undefined {
  return values.find(
    (value) => value !== null && value !== undefined,
  );
}

function summaryRate(
  summary: BenchmarkSummary,
  ...keys: Array<keyof BenchmarkSummary>
): number | null | undefined {
  for (const key of keys) {
    const value = summary[key];
    if (typeof value === "number") return value;
  }
  return undefined;
}

function answerPassed(
  result: BenchmarkResult,
  stage: "initial" | "final" = "final",
): boolean | null | undefined {
  if (stage === "initial") {
    return (
      result.initial_answer_passed ??
      result.initial_benchmark_passed
    );
  }

  return result.final_answer_passed ?? result.final_benchmark_passed;
}

function dateTime(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function MetricCard({
  label,
  value,
  description,
  emphasis = false,
}: {
  label: string;
  value: string;
  description: string;
  emphasis?: boolean;
}) {
  return (
    <div
      className={`rounded-2xl border p-4 shadow-sm ${
        emphasis
          ? "border-indigo-200 bg-gradient-to-br from-indigo-600 to-blue-600 text-white"
          : "border-slate-200 bg-white text-slate-900"
      }`}
    >
      <p
        className={`text-[11px] font-bold uppercase tracking-[0.16em] ${
          emphasis ? "text-indigo-100" : "text-slate-400"
        }`}
      >
        {label}
      </p>
      <p className="mt-2 text-3xl font-black tracking-tight">{value}</p>
      <p
        className={`mt-1 text-xs ${
          emphasis ? "text-indigo-100" : "text-slate-500"
        }`}
      >
        {description}
      </p>
    </div>
  );
}

function RateBar({
  label,
  value,
  caption,
}: {
  label: string;
  value?: number | null;
  caption: string;
}) {
  const safeValue = Math.max(0, Math.min(1, value ?? 0));

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-slate-700">{label}</p>
          <p className="text-xs text-slate-400">{caption}</p>
        </div>
        <span className="text-sm font-black text-slate-900">
          {percentage(value)}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-sky-500 transition-all"
          style={{ width: `${safeValue * 100}%` }}
        />
      </div>
    </div>
  );
}

function StatusBadge({
  passed,
  label,
}: {
  passed?: boolean | null;
  label?: string;
}) {
  if (passed === null || passed === undefined) {
    return (
      <span className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-semibold text-slate-500">
        {label ?? "N/A"}
      </span>
    );
  }

  return (
    <span
      className={`rounded-full px-2 py-1 text-[11px] font-semibold ${
        passed
          ? "bg-emerald-100 text-emerald-700"
          : "bg-red-100 text-red-700"
      }`}
    >
      {label ?? (passed ? "PASS" : "FAIL")}
    </span>
  );
}

function TextPanel({
  title,
  value,
  tone = "default",
}: {
  title: string;
  value?: string | null;
  tone?: "default" | "success";
}) {
  return (
    <section
      className={`rounded-2xl border p-4 ${
        tone === "success"
          ? "border-emerald-200 bg-emerald-50/60"
          : "border-slate-200 bg-white"
      }`}
    >
      <h3 className="text-sm font-bold text-slate-900">{title}</h3>
      <div className="mt-3 max-h-[420px] overflow-y-auto whitespace-pre-wrap text-sm leading-7 text-slate-700">
        {value?.trim() || "저장된 답변이 없습니다."}
      </div>
    </section>
  );
}

export default function EvaluationDashboardPage() {
  const [runs, setRuns] = useState<BenchmarkRunListItem[]>([]);
  const [run, setRun] = useState<BenchmarkRun | null>(null);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedResultId, setSelectedResultId] = useState("");
  const [filter, setFilter] = useState<ResultFilter>("all");
  const [busy, setBusy] = useState(true);
  const [message, setMessage] = useState<string | null>(null);

  async function loadInitial() {
    setBusy(true);
    setMessage(null);

    try {
      const [runList, latest] = await Promise.all([
        getBenchmarkRuns(),
        getLatestBenchmarkRun(),
      ]);
      setRuns(runList);
      setRun(latest);
      setSelectedRunId(latest.run_id);
      setSelectedResultId(latest.results[0]?.id ?? "");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Benchmark 결과를 불러오지 못했습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void loadInitial();
  }, []);

  async function selectRun(runId: string) {
    if (!runId || runId === selectedRunId) return;

    setBusy(true);
    setMessage(null);
    try {
      const next = await getBenchmarkRun(runId);
      setRun(next);
      setSelectedRunId(runId);
      setSelectedResultId(next.results[0]?.id ?? "");
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "선택한 실행 결과를 불러오지 못했습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  const filteredResults = useMemo(() => {
    const values = run?.results ?? [];

    if (filter === "passed") {
      return values.filter(
        (result) => answerPassed(result) === true,
      );
    }
    if (filter === "rewritten") {
      return values.filter((result) => result.rewrite_success === true);
    }
    if (filter === "failed") {
      return values.filter(
        (result) => answerPassed(result) === false,
      );
    }
    return values;
  }, [filter, run]);

  const selectedResult = useMemo<BenchmarkResult | null>(() => {
    return (
      filteredResults.find(
        (result) => result.id === selectedResultId,
      ) ??
      filteredResults[0] ??
      null
    );
  }, [filteredResults, run, selectedResultId]);

  const summary = run?.summary ?? {};
  const answerAccuracy = summaryRate(
    summary,
    "answer_accuracy",
    "final_benchmark_pass_rate",
  );
  const initialAccuracy = summaryRate(
    summary,
    "initial_answer_accuracy",
    "initial_benchmark_pass_rate",
  );
  const pageHitRate = summaryRate(
    summary,
    "retrieval_page_recall_at_k",
    "preferred_page_hit_rate",
  );
  const documentHitRate = summaryRate(
    summary,
    "retrieval_document_recall_at_k",
    "expected_document_hit_rate",
  );
  const categoryMetrics = Object.entries(
    summary.category_metrics ?? {},
  ).sort(([left], [right]) => left.localeCompare(right));

  const rewriteCount = (run?.results ?? []).filter(
    (result) => result.rewrite_success,
  ).length;

  return (
    <main className="min-h-screen bg-[#eef1f6] text-slate-900 md:pl-[72px]">
      <AppIconRail />
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -left-40 -top-40 h-96 w-96 rounded-full bg-indigo-200/35 blur-3xl" />
        <div className="absolute -right-40 top-1/3 h-96 w-96 rounded-full bg-sky-200/35 blur-3xl" />
      </div>

      <header className="sticky top-0 z-30 border-b border-white/70 bg-white/85 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-4 px-4 py-4">
          <div className="flex items-center gap-3">
            <a
              href="/"
              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-600 shadow-sm hover:bg-slate-50"
            >
              ← 채팅
            </a>
            <div>
              <h1 className="text-lg font-black tracking-tight">
                Benchmark Dashboard
              </h1>
              <p className="text-xs text-slate-500">
                Well Test Agent 정확도·재작성·검색 성능
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <a
              href="/review"
              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-600 shadow-sm hover:bg-slate-50"
            >
              Figure Review
            </a>
            <button
              type="button"
              onClick={() => void loadInitial()}
              disabled={busy}
              className="rounded-xl bg-indigo-600 px-4 py-2 text-xs font-bold text-white shadow-sm disabled:opacity-50"
            >
              {busy ? "불러오는 중…" : "새로고침"}
            </button>
          </div>
        </div>
      </header>

      <div className="relative mx-auto max-w-[1600px] space-y-5 px-4 py-6">
        {message && (
          <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {message}
          </div>
        )}

        <section className="rounded-3xl border border-white/80 bg-white/80 p-5 shadow-sm backdrop-blur">
          <div className="flex flex-col justify-between gap-4 lg:flex-row lg:items-end">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-indigo-500">
                Evaluation run
              </p>
              <h2 className="mt-1 text-2xl font-black tracking-tight">
                {run?.run_id ?? "저장된 결과 없음"}
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                {dateTime(run?.created_at)} · {run?.condition ?? run?.model ?? "condition unknown"} ·{" "}
                {run?.question_count ?? 0} questions
              </p>
            </div>

            <label className="min-w-[300px]">
              <span className="mb-1 block text-xs font-semibold text-slate-500">
                과거 실행 선택
              </span>
              <select
                value={selectedRunId}
                onChange={(event) => void selectRun(event.target.value)}
                disabled={busy || runs.length === 0}
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm font-semibold outline-none ring-indigo-200 focus:ring-2 disabled:opacity-50"
              >
                {runs.map((item) => (
                  <option key={item.run_id} value={item.run_id}>
                    {item.run_id} · {item.condition ?? item.model ?? "unknown"} ·{" "}
                    {percentage(
                      summaryRate(
                        item.summary,
                        "answer_accuracy",
                        "final_benchmark_pass_rate",
                      ),
                    )}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </section>

        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
          <MetricCard
            label="Answer accuracy"
            value={percentage(answerAccuracy)}
            description={`${summary.questions_completed ?? 0}/${
              summary.questions_total ?? 0
            } completed`}
            emphasis
          />
          <MetricCard
            label="Initial pass"
            value={percentage(initialAccuracy)}
            description="첫 답변 기준"
          />
          <MetricCard
            label="Rewrite success"
            value={percentage(
              run?.mode === "ollama-direct"
                ? null
                : summary.rewrite_success_rate,
            )}
            description={
              run?.mode === "ollama-direct"
                ? "단독 LLM에는 적용 안 됨"
                : `${rewriteCount}개 문항 자동 복구`
            }
          />
          <MetricCard
            label="Hallucination"
            value={percentage(summary.hallucination_rate)}
            description="금지 주장·잘못된 비거절 비율"
          />
          <MetricCard
            label="Page hit"
            value={percentage(pageHitRate)}
            description="권장 페이지 검색"
          />
          <MetricCard
            label="Average attempts"
            value={(summary.average_attempts ?? 0).toFixed(3)}
            description={`평균 ${seconds(summary.average_total_seconds)}`}
          />
        </section>

        <section className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-black">품질 지표</h2>
            <div className="mt-5 space-y-5">
              <RateBar
                label="답변 정확도"
                value={answerAccuracy}
                caption="내용·거절 행동 기준"
              />
              <RateBar
                label="환각률"
                value={summary.hallucination_rate}
                caption="낮을수록 좋음"
              />
              {run?.mode !== "ollama-direct" && (
                <>
                  <RateBar
                    label="최종 Benchmark 통과율"
                    value={summary.final_benchmark_pass_rate}
                    caption="답변과 검색 문서를 함께 평가"
                  />
                  <RateBar
                    label="최초 Benchmark 통과율"
                    value={summary.initial_benchmark_pass_rate}
                    caption="첫 생성 답변"
                  />
                  <RateBar
                    label="재작성 성공률"
                    value={summary.rewrite_success_rate}
                    caption="최초 실패 문항 중 복구 비율"
                  />
                  <RateBar
                    label="검증기 오류 감지율"
                    value={summary.validator_detection_rate}
                    caption="실제 오류를 검증기가 잡은 비율"
                  />
                </>
              )}
              <RateBar
                label="정확한 거절률"
                value={summary.exact_refusal_rate}
                caption="문서 근거가 없는 질문"
              />
              <RateBar
                label="예상 문서 적중률"
                value={documentHitRate}
                caption="정답 문서가 검색 근거에 포함"
              />
              <RateBar
                label="Figure 검색 정확도"
                value={summary.figure_retrieval_accuracy}
                caption="Figure 문항의 관련 이미지 반환"
              />
            </div>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-black">시간 및 안정성</h2>
            <dl className="mt-4 divide-y divide-slate-100">
              {[
                ["평균 검색 시간", seconds(summary.average_retrieval_seconds)],
                ["평균 생성 시간", seconds(summary.average_generation_seconds)],
                ["평균 전체 시간", seconds(summary.average_total_seconds)],
                [
                  "Validator 오탐률",
                  percentage(summary.validator_false_positive_rate),
                ],
                [
                  "Infrastructure errors",
                  String(summary.infrastructure_errors ?? 0),
                ],
                [
                  "Wall clock",
                  seconds(run?.wall_clock_seconds),
                ],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="flex items-center justify-between py-3"
                >
                  <dt className="text-sm text-slate-500">{label}</dt>
                  <dd className="text-sm font-black text-slate-900">
                    {value}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-black">카테고리별 결과</h2>
              <p className="text-xs text-slate-500">
                각 질문 유형의 답변 정확도
              </p>
            </div>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-bold text-slate-500">
              {categoryMetrics.length} categories
            </span>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5">
            {categoryMetrics.map(([category, metric]) => (
              <div
                key={category}
                className="rounded-2xl border border-slate-200 bg-slate-50 p-3"
              >
                <p className="truncate text-xs font-bold text-slate-600">
                  {category}
                </p>
                <div className="mt-2 flex items-end justify-between">
                  <span className="text-2xl font-black">
                    {percentage(
                      firstNumber(
                        metric.answer_accuracy,
                        metric.final_pass_rate,
                      ),
                    )}
                  </span>
                  <span className="text-xs text-slate-400">
                    {metric.answer_passed ?? metric.final_passed ?? 0}/
                    {metric.total ?? 0}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="grid gap-5 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,0.95fr)]">
          <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
            <div className="flex flex-col justify-between gap-3 border-b border-slate-200 p-4 sm:flex-row sm:items-center">
              <div>
                <h2 className="text-base font-black">문항별 결과</h2>
                <p className="text-xs text-slate-500">
                  행을 선택하면 답변과 검증 과정을 확인할 수 있어.
                </p>
              </div>

              <div className="flex flex-wrap gap-1 rounded-xl bg-slate-100 p-1">
                {(
                  [
                    ["all", "전체"],
                    ["passed", "최종 통과"],
                    ["rewritten", "재작성"],
                    ["failed", "실패"],
                  ] as Array<[ResultFilter, string]>
                ).map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setFilter(value)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-bold ${
                      filter === value
                        ? "bg-white text-indigo-700 shadow-sm"
                        : "text-slate-500"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div className="max-h-[760px] overflow-auto">
              <table className="w-full border-collapse text-left">
                <thead className="sticky top-0 bg-slate-50 text-[11px] uppercase tracking-wider text-slate-400">
                  <tr>
                    <th className="px-4 py-3">ID</th>
                    <th className="px-3 py-3">Category</th>
                    <th className="px-3 py-3">Initial</th>
                    <th className="px-3 py-3">Final</th>
                    <th className="px-3 py-3">Attempts</th>
                    <th className="px-4 py-3">Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {filteredResults.map((result) => (
                    <tr
                      key={result.id}
                      onClick={() => setSelectedResultId(result.id)}
                      className={`cursor-pointer transition hover:bg-indigo-50/60 ${
                        selectedResult?.id === result.id
                          ? "bg-indigo-50"
                          : ""
                      }`}
                    >
                      <td className="px-4 py-3 text-sm font-black text-slate-900">
                        {result.id}
                      </td>
                      <td className="px-3 py-3 text-xs text-slate-500">
                        {result.category ?? "—"}
                      </td>
                      <td className="px-3 py-3">
                        <StatusBadge
                          passed={answerPassed(result, "initial")}
                        />
                      </td>
                      <td className="px-3 py-3">
                        <StatusBadge
                          passed={answerPassed(result)}
                        />
                      </td>
                      <td className="px-3 py-3 text-sm font-bold">
                        {result.attempts ?? "—"}
                      </td>
                      <td className="px-4 py-3 text-xs text-slate-500">
                        {seconds(result.total_seconds)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {filteredResults.length === 0 && (
                <p className="p-8 text-center text-sm text-slate-400">
                  이 조건에 해당하는 문항이 없습니다.
                </p>
              )}
            </div>
          </div>

          <div className="space-y-4">
            {selectedResult ? (
              <>
                <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xl font-black">
                      {selectedResult.id}
                    </span>
                    <StatusBadge
                      passed={answerPassed(selectedResult)}
                    />
                    {selectedResult.rewrite_success && (
                      <StatusBadge passed label="REWRITTEN" />
                    )}
                    <span className="rounded-full bg-slate-100 px-2 py-1 text-[11px] font-semibold text-slate-500">
                      {selectedResult.category ?? "uncategorized"}
                    </span>
                  </div>

                  <p className="mt-3 text-sm font-semibold leading-6 text-slate-700">
                    {selectedResult.question ?? "질문 없음"}
                  </p>

                  <div className="mt-4 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                    <div className="rounded-xl bg-slate-50 p-3">
                      <p className="text-slate-400">Retrieval</p>
                      <p className="mt-1 font-black">
                        {seconds(selectedResult.retrieval_seconds)}
                      </p>
                    </div>
                    <div className="rounded-xl bg-slate-50 p-3">
                      <p className="text-slate-400">Generation</p>
                      <p className="mt-1 font-black">
                        {seconds(selectedResult.generation_seconds)}
                      </p>
                    </div>
                    <div className="rounded-xl bg-slate-50 p-3">
                      <p className="text-slate-400">Page hit</p>
                      <p className="mt-1 font-black">
                        {selectedResult.preferred_page_hit === null ||
                        selectedResult.preferred_page_hit === undefined
                          ? "N/A"
                          : selectedResult.preferred_page_hit
                            ? "YES"
                            : "NO"}
                      </p>
                    </div>
                    <div className="rounded-xl bg-slate-50 p-3">
                      <p className="text-slate-400">Source pages</p>
                      <p className="mt-1 truncate font-black">
                        {selectedResult.source_pages?.join(", ") || "—"}
                      </p>
                    </div>
                  </div>
                </section>

                <div className="grid gap-4 2xl:grid-cols-2">
                  <TextPanel
                    title="최초 답변"
                    value={selectedResult.initial_answer}
                  />
                  <TextPanel
                    title="최종 답변"
                    value={selectedResult.final_answer}
                    tone="success"
                  />
                </div>

                <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
                  <h3 className="text-base font-black">검증 및 재작성 과정</h3>
                  <div className="mt-4 space-y-3">
                    {(selectedResult.attempt_details ?? []).map(
                      (attempt, index) => (
                        <div
                          key={`${selectedResult.id}-${attempt.attempt ?? index}`}
                          className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-black">
                                Attempt {attempt.attempt ?? index + 1}
                              </span>
                              <StatusBadge
                                passed={attempt.validation_passed}
                                label={
                                  attempt.validation_passed
                                    ? "VALID"
                                    : "REWRITE"
                                }
                              />
                            </div>
                            <span className="text-xs font-semibold text-slate-400">
                              {seconds(attempt.elapsed_seconds)}
                            </span>
                          </div>

                          {(attempt.rule_ids?.length ?? 0) > 0 && (
                            <div className="mt-3 flex flex-wrap gap-1">
                              {attempt.rule_ids?.map((rule) => (
                                <span
                                  key={rule}
                                  className="rounded-md bg-white px-2 py-1 text-[11px] font-bold text-indigo-600 ring-1 ring-slate-200"
                                >
                                  {rule}
                                </span>
                              ))}
                            </div>
                          )}

                          {(attempt.errors?.length ?? 0) > 0 && (
                            <ul className="mt-3 space-y-1 text-xs leading-5 text-red-600">
                              {attempt.errors?.map((error) => (
                                <li key={error}>• {error}</li>
                              ))}
                            </ul>
                          )}
                        </div>
                      ),
                    )}

                    {(selectedResult.attempt_details?.length ?? 0) === 0 && (
                      <p className="rounded-xl bg-slate-50 p-4 text-sm text-slate-400">
                        연결된 Agent Run 로그가 없습니다.
                      </p>
                    )}
                  </div>
                </section>

                {[
                  ...(selectedResult.final_required_failures ?? []),
                  ...(selectedResult.final_forbidden_hits ?? []),
                ].length > 0 && (
                  <section className="rounded-3xl border border-red-200 bg-red-50 p-5">
                    <h3 className="font-black text-red-800">
                      최종 Benchmark 실패 원인
                    </h3>
                    <ul className="mt-3 space-y-1 text-xs text-red-700">
                      {[
                        ...(selectedResult.final_required_failures ?? []),
                        ...(selectedResult.final_forbidden_hits ?? []),
                      ].map((value) => (
                        <li key={value}>• {value}</li>
                      ))}
                    </ul>
                  </section>
                )}
              </>
            ) : (
              <div className="rounded-3xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-400 shadow-sm">
                확인할 문항을 선택해줘.
              </div>
            )}
          </div>
        </section>

        <footer className="pb-4 text-center text-xs text-slate-400">
          Read-only dashboard · Benchmark JSON과 Agent Run 로그만 조회 ·
          ChromaDB 및 Figure Note 변경 없음
        </footer>
      </div>
    </main>
  );
}
