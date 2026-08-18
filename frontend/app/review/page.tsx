"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { AppIconRail } from "@/components/AppNavigation";
import {
  getReviewCandidates,
  getReviewSummary,
  regenerateReviewPreview,
  resolveReviewUrl,
  ReviewCandidate,
  ReviewDocumentSummary,
  ReviewStatus,
  updateReviewCandidate,
  updateReviewRotation,
} from "@/lib/reviewApi";

const PAGE_SIZE = 12;
const STATUS_OPTIONS: Array<{
  value: "all" | ReviewStatus;
  label: string;
}> = [
  { value: "all", label: "전체 상태" },
  { value: "review_required", label: "검토 필요" },
  { value: "failed", label: "실패" },
  { value: "valid", label: "유효" },
  { value: "ignored", label: "무시" },
];

const STATUS_BADGE: Record<ReviewStatus, string> = {
  valid: "bg-emerald-100 text-emerald-700",
  review_required: "bg-amber-100 text-amber-700",
  failed: "bg-red-100 text-red-700",
  ignored: "bg-slate-200 text-slate-600",
};

function ImagePane({
  title,
  url,
  refreshKey,
}: {
  title: string;
  url: string | null;
  refreshKey: number;
}) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [url, refreshKey]);

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-3 py-2 text-xs font-semibold text-slate-500">
        {title}
      </div>
      <div className="flex h-64 items-center justify-center bg-slate-50 p-3">
        {!url || failed ? (
          <span className="text-xs text-slate-400">
            이미지를 표시할 수 없습니다.
          </span>
        ) : (
          <img
            src={`${resolveReviewUrl(url)}${
              url.includes("?") ? "&" : "?"
            }v=${refreshKey}`}
            alt={title}
            className="max-h-full max-w-full object-contain"
            onError={() => setFailed(true)}
          />
        )}
      </div>
    </div>
  );
}

function CountCard({
  label,
  value,
  description,
}: {
  label: string;
  value: number;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </p>
      <p className="mt-2 text-3xl font-bold text-slate-900">
        {value.toLocaleString()}
      </p>
      <p className="mt-1 text-xs text-slate-500">
        {description}
      </p>
    </div>
  );
}

export default function FigureReviewPage() {
  const [documents, setDocuments] = useState<
    ReviewDocumentSummary[]
  >([]);
  const [totals, setTotals] = useState<Record<string, number>>(
    {},
  );
  const [candidates, setCandidates] = useState<
    ReviewCandidate[]
  >([]);
  const [selected, setSelected] = useState<
    ReviewCandidate | null
  >(null);
  const [documentId, setDocumentId] = useState("");
  const [status, setStatus] = useState("review_required");
  const [pageFilter, setPageFilter] = useState("");
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [total, setTotal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(
    null,
  );
  const [refreshKey, setRefreshKey] = useState(
    Date.now(),
  );

  const pageNumber = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(
    1,
    Math.ceil(total / PAGE_SIZE),
  );

  async function loadSummary() {
    const summary = await getReviewSummary();
    setDocuments(summary.documents);
    setTotals(summary.totals);
  }

  async function loadCandidates(nextOffset = offset) {
    const result = await getReviewCandidates({
      documentId,
      status,
      page: pageFilter,
      query,
      offset: nextOffset,
      limit: PAGE_SIZE,
    });
    setCandidates(result.items);
    setTotal(result.total);
    setOffset(result.offset);

    if (selected) {
      const updated = result.items.find(
        (item) =>
          item.candidate_id === selected.candidate_id,
      );
      if (updated) {
        setSelected(updated);
      }
    }
  }

  useEffect(() => {
    setBusy(true);
    Promise.all([loadSummary(), loadCandidates(0)])
      .catch((error) => {
        setMessage(
          error instanceof Error
            ? error.message
            : "대시보드를 불러오지 못했습니다.",
        );
      })
      .finally(() => setBusy(false));
    // Initial load only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedDocument = useMemo(
    () =>
      documents.find(
        (document) =>
          document.document_id === documentId,
      ),
    [documents, documentId],
  );

  async function applyFilters(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      await loadCandidates(0);
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "검색에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function goToPage(nextPage: number) {
    const nextOffset = Math.max(
      0,
      (nextPage - 1) * PAGE_SIZE,
    );
    setBusy(true);
    try {
      await loadCandidates(nextOffset);
    } finally {
      setBusy(false);
    }
  }

  async function saveSelected() {
    if (!selected) {
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const updated = await updateReviewCandidate(
        selected.candidate_id,
        {
          status: selected.status,
          title: selected.title,
          analysis: selected.analysis,
          x_axis: selected.x_axis,
          x_axis_unit: selected.x_axis_unit,
          y_axis: selected.y_axis,
          y_axis_unit: selected.y_axis_unit,
          trend_summary: selected.trend_summary,
          engineering_meaning:
            selected.engineering_meaning,
          series_descriptions:
            selected.series_descriptions,
          reference_lines: selected.reference_lines,
        },
      );
      setSelected(updated);
      setMessage(
        "후보 JSON을 백업 후 저장했습니다. Chroma와 Figure Note에는 아직 반영되지 않았습니다.",
      );
      await Promise.all([
        loadSummary(),
        loadCandidates(offset),
      ]);
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "저장에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function saveRotation() {
    if (!selected) {
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const updated = await updateReviewRotation(
        selected.candidate_id,
        {
          rotation: selected.rotation,
          pdf_crop_rotation:
            selected.pdf_crop_rotation,
          enhance: selected.enhance,
          regenerate: true,
        },
      );
      setSelected(updated);
      setRefreshKey(Date.now());
      setMessage(
        "회전 설정을 백업 후 저장하고 preview를 다시 생성했습니다.",
      );
      await loadSummary();
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "회전 저장에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function regeneratePreview() {
    if (!selected) {
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const result = await regenerateReviewPreview(
        selected.candidate_id,
      );
      setSelected({
        ...selected,
        preview_url: result.preview_url,
        preview_source: result.preview_source,
      });
      setRefreshKey(Date.now());
      setMessage("Preview를 다시 생성했습니다.");
      await loadSummary();
    } catch (error) {
      setMessage(
        error instanceof Error
          ? error.message
          : "Preview 생성에 실패했습니다.",
      );
    } finally {
      setBusy(false);
    }
  }

  function updateSelected<K extends keyof ReviewCandidate>(
    key: K,
    value: ReviewCandidate[K],
  ) {
    setSelected((current) =>
      current ? { ...current, [key]: value } : current,
    );
  }

  return (
    <main className="min-h-screen bg-[#eef1f6] text-slate-900 md:pl-[72px]">
      <AppIconRail />
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-4 py-4">
          <div>
            <div className="flex items-center gap-3">
              <a
                href="/"
                className="rounded-xl border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-50"
              >
                ← 채팅
              </a>
              <div>
                <h1 className="text-lg font-bold">
                  Figure Review Dashboard
                </h1>
                <p className="text-xs text-slate-500">
                  후보 검수·회전·preview 관리
                </p>
              </div>
            </div>
          </div>
          <div className="rounded-full bg-amber-50 px-4 py-2 text-xs font-semibold text-amber-700">
            자동 재색인 꺼짐 · Chroma 변경 없음
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1500px] space-y-5 px-4 py-6">
        <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
          <CountCard
            label="전체 후보"
            value={totals.total ?? 0}
            description="현재 candidate JSON"
          />
          <CountCard
            label="검토 필요"
            value={totals.review_required ?? 0}
            description="사람 검수가 필요한 항목"
          />
          <CountCard
            label="실패"
            value={totals.failed ?? 0}
            description="분석 또는 스키마 실패"
          />
          <CountCard
            label="유효"
            value={totals.valid ?? 0}
            description="현재 valid 후보"
          />
          <CountCard
            label="Figure Notes"
            value={totals.figure_note_count ?? 0}
            description="저장된 note 파일"
          />
          <CountCard
            label="Previews"
            value={totals.preview_count ?? 0}
            description="생성된 표시용 이미지"
          />
        </section>

        <form
          onSubmit={applyFilters}
          className="grid gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm md:grid-cols-5"
        >
          <select
            value={documentId}
            onChange={(event) => {
              setDocumentId(event.target.value);
              setOffset(0);
            }}
            className="rounded-xl border border-slate-200 px-3 py-2 text-sm"
          >
            <option value="">전체 문서</option>
            {documents.map((document) => (
              <option
                key={document.document_id}
                value={document.document_id}
              >
                {document.filename} ({document.total})
              </option>
            ))}
          </select>

          <select
            value={status}
            onChange={(event) => {
              setStatus(event.target.value);
              setOffset(0);
            }}
            className="rounded-xl border border-slate-200 px-3 py-2 text-sm"
          >
            {STATUS_OPTIONS.map((option) => (
              <option
                key={option.value}
                value={option.value}
              >
                {option.label}
              </option>
            ))}
          </select>

          <input
            value={pageFilter}
            onChange={(event) =>
              setPageFilter(
                event.target.value.replace(/\D/g, ""),
              )
            }
            placeholder="페이지"
            className="rounded-xl border border-slate-200 px-3 py-2 text-sm"
          />

          <input
            value={query}
            onChange={(event) =>
              setQuery(event.target.value)
            }
            placeholder="제목·분석·분류 검색"
            className="rounded-xl border border-slate-200 px-3 py-2 text-sm"
          />

          <button
            disabled={busy}
            className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
          >
            검색
          </button>
        </form>

        {selectedDocument && (
          <div className="rounded-2xl border border-indigo-100 bg-indigo-50 px-4 py-3 text-sm text-indigo-800">
            {selectedDocument.filename} · valid{" "}
            {selectedDocument.counts.valid} · review{" "}
            {selectedDocument.counts.review_required} · failed{" "}
            {selectedDocument.counts.failed} · ignored{" "}
            {selectedDocument.counts.ignored}
          </div>
        )}

        {message && (
          <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 shadow-sm">
            {message}
          </div>
        )}

        <section className="space-y-4">
          {candidates.length === 0 && !busy ? (
            <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-12 text-center text-sm text-slate-500">
              조건에 맞는 후보가 없습니다.
            </div>
          ) : (
            candidates.map((candidate) => (
              <article
                key={candidate.candidate_id}
                className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"
              >
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-bold">
                      {candidate.title ??
                        candidate.asset_filename}
                    </p>
                    <p className="mt-1 truncate text-xs text-slate-500">
                      {candidate.document_name} · p.
                      {candidate.page_number ?? "?"} · fig{" "}
                      {candidate.image_index ?? "?"}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    {candidate.needs_reindex && (
                      <span className="rounded-full bg-violet-100 px-2 py-1 text-[11px] font-semibold text-violet-700">
                        검색 반영 대기
                      </span>
                    )}
                    <span
                      className={`rounded-full px-2 py-1 text-[11px] font-semibold ${
                        STATUS_BADGE[candidate.status]
                      }`}
                    >
                      {candidate.status}
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        setSelected(candidate)
                      }
                      className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white"
                    >
                      검수하기
                    </button>
                  </div>
                </div>

                <div className="grid gap-3 p-4 lg:grid-cols-2">
                  <ImagePane
                    title="원본 추출 이미지"
                    url={candidate.original_url}
                    refreshKey={0}
                  />
                  <ImagePane
                    title="보정 Preview"
                    url={candidate.preview_url}
                    refreshKey={refreshKey}
                  />
                </div>

                <div className="grid gap-3 border-t border-slate-100 px-4 py-3 text-xs text-slate-600 md:grid-cols-4">
                  <span>
                    분류:{" "}
                    <strong>
                      {candidate.classification || "-"}
                    </strong>
                  </span>
                  <span>
                    confidence:{" "}
                    <strong>
                      {candidate.confidence ?? "-"}
                    </strong>
                  </span>
                  <span>
                    rotation:{" "}
                    <strong>
                      {candidate.rotation ?? "auto"}
                    </strong>
                  </span>
                  <span>
                    PDF crop:{" "}
                    <strong>
                      {candidate.pdf_crop_rotation ??
                        "auto"}
                    </strong>
                  </span>
                </div>

                {(candidate.validation_errors.length > 0 ||
                  candidate.manual_review_reasons.length >
                    0) && (
                  <div className="border-t border-red-100 bg-red-50 px-4 py-3 text-xs text-red-700">
                    {[
                      ...candidate.validation_errors,
                      ...candidate.manual_review_reasons,
                    ].join(" · ")}
                  </div>
                )}
              </article>
            ))
          )}
        </section>

        <div className="flex items-center justify-center gap-3 py-3">
          <button
            type="button"
            disabled={busy || pageNumber <= 1}
            onClick={() => goToPage(pageNumber - 1)}
            className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm disabled:opacity-40"
          >
            이전
          </button>
          <span className="text-sm text-slate-500">
            {pageNumber} / {totalPages} · 총{" "}
            {total.toLocaleString()}개
          </span>
          <button
            type="button"
            disabled={
              busy || pageNumber >= totalPages
            }
            onClick={() => goToPage(pageNumber + 1)}
            className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm disabled:opacity-40"
          >
            다음
          </button>
        </div>
      </div>

      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="max-h-[94vh] w-full max-w-6xl overflow-y-auto rounded-3xl bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="sticky top-0 z-10 flex items-center justify-between border-b border-slate-200 bg-white px-5 py-4">
              <div>
                <h2 className="font-bold">
                  {selected.title ??
                    selected.asset_filename}
                </h2>
                <p className="text-xs text-slate-500">
                  {selected.document_name} · p.
                  {selected.page_number ?? "?"}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-100 text-xl"
              >
                ×
              </button>
            </div>

            <div className="grid gap-4 p-5 lg:grid-cols-2">
              <ImagePane
                title="원본"
                url={selected.original_url}
                refreshKey={0}
              />
              <ImagePane
                title="Preview"
                url={selected.preview_url}
                refreshKey={refreshKey}
              />
            </div>

            <div className="grid gap-4 border-t border-slate-200 p-5 lg:grid-cols-2">
              <div className="space-y-3">
                <label className="block text-xs font-semibold text-slate-600">
                  상태
                  <select
                    value={selected.status}
                    onChange={(event) =>
                      updateSelected(
                        "status",
                        event.target
                          .value as ReviewStatus,
                      )
                    }
                    className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm"
                  >
                    {STATUS_OPTIONS.filter(
                      (option) =>
                        option.value !== "all",
                    ).map((option) => (
                      <option
                        key={option.value}
                        value={option.value}
                      >
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>

                {[
                  ["title", "제목"],
                  ["analysis", "분석"],
                  ["x_axis", "X축"],
                  ["x_axis_unit", "X축 단위"],
                  ["y_axis", "Y축"],
                  ["y_axis_unit", "Y축 단위"],
                  ["trend_summary", "추세"],
                  [
                    "engineering_meaning",
                    "공학적 의미",
                  ],
                ].map(([key, label]) => (
                  <label
                    key={key}
                    className="block text-xs font-semibold text-slate-600"
                  >
                    {label}
                    <textarea
                      disabled={!selected.editable}
                      value={
                        String(
                          selected[
                            key as keyof ReviewCandidate
                          ] ?? "",
                        )
                      }
                      onChange={(event) =>
                        updateSelected(
                          key as keyof ReviewCandidate,
                          event.target
                            .value as never,
                        )
                      }
                      rows={
                        [
                          "analysis",
                          "trend_summary",
                          "engineering_meaning",
                        ].includes(key)
                          ? 3
                          : 1
                      }
                      className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm disabled:bg-slate-100"
                    />
                  </label>
                ))}

                <label className="block text-xs font-semibold text-slate-600">
                  계열 설명 — 줄바꿈으로 구분
                  <textarea
                    disabled={!selected.editable}
                    value={selected.series_descriptions.join(
                      "\n",
                    )}
                    onChange={(event) =>
                      updateSelected(
                        "series_descriptions",
                        event.target.value
                          .split("\n")
                          .filter(Boolean),
                      )
                    }
                    rows={4}
                    className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm disabled:bg-slate-100"
                  />
                </label>

                <label className="block text-xs font-semibold text-slate-600">
                  기준선 — 줄바꿈으로 구분
                  <textarea
                    disabled={!selected.editable}
                    value={selected.reference_lines.join(
                      "\n",
                    )}
                    onChange={(event) =>
                      updateSelected(
                        "reference_lines",
                        event.target.value
                          .split("\n")
                          .filter(Boolean),
                      )
                    }
                    rows={4}
                    className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm disabled:bg-slate-100"
                  />
                </label>

                {!selected.editable && (
                  <p className="rounded-xl bg-amber-50 p-3 text-xs text-amber-700">
                    이 후보에는 final_note_data가 없어
                    상태·회전만 수정할 수 있습니다.
                  </p>
                )}
              </div>

              <div className="space-y-4">
                <div className="rounded-2xl border border-slate-200 p-4">
                  <h3 className="text-sm font-bold">
                    Preview 회전
                  </h3>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    <label className="text-xs font-semibold text-slate-600">
                      추출 이미지 회전
                      <select
                        value={
                          selected.rotation ?? ""
                        }
                        onChange={(event) =>
                          updateSelected(
                            "rotation",
                            event.target.value === ""
                              ? null
                              : Number(
                                  event.target.value,
                                ),
                          )
                        }
                        className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm"
                      >
                        <option value="">
                          자동
                        </option>
                        {[0, 90, 180, 270].map(
                          (value) => (
                            <option
                              key={value}
                              value={value}
                            >
                              {value}°
                            </option>
                          ),
                        )}
                      </select>
                    </label>

                    <label className="text-xs font-semibold text-slate-600">
                      PDF crop 회전
                      <select
                        value={
                          selected.pdf_crop_rotation ??
                          ""
                        }
                        onChange={(event) =>
                          updateSelected(
                            "pdf_crop_rotation",
                            event.target.value === ""
                              ? null
                              : Number(
                                  event.target.value,
                                ),
                          )
                        }
                        className="mt-1 w-full rounded-xl border border-slate-200 px-3 py-2 text-sm"
                      >
                        <option value="">
                          자동
                        </option>
                        {[0, 90, 180, 270].map(
                          (value) => (
                            <option
                              key={value}
                              value={value}
                            >
                              {value}°
                            </option>
                          ),
                        )}
                      </select>
                    </label>
                  </div>

                  <label className="mt-3 flex items-center gap-2 text-xs font-semibold text-slate-600">
                    <input
                      type="checkbox"
                      checked={selected.enhance}
                      onChange={(event) =>
                        updateSelected(
                          "enhance",
                          event.target.checked,
                        )
                      }
                    />
                    어두운 추출 이미지 보정
                  </label>

                  <div className="mt-4 flex flex-wrap gap-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={saveRotation}
                      className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
                    >
                      회전 저장 + 재생성
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={regeneratePreview}
                      className="rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold"
                    >
                      Preview만 재생성
                    </button>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4 text-xs text-slate-600">
                  <p>
                    schema valid:{" "}
                    <strong>
                      {String(selected.schema_valid)}
                    </strong>
                  </p>
                  <p className="mt-1">
                    information quality:{" "}
                    <strong>
                      {String(
                        selected.information_quality_passed,
                      )}
                    </strong>
                  </p>
                  <p className="mt-1">
                    apply ready:{" "}
                    <strong>
                      {String(selected.apply_ready)}
                    </strong>
                  </p>
                  <p className="mt-1">
                    preview source:{" "}
                    <strong>
                      {selected.preview_source ??
                        "요청 시 생성"}
                    </strong>
                  </p>
                </div>

                <div className="rounded-2xl bg-amber-50 p-4 text-xs leading-5 text-amber-800">
                  저장은 candidate JSON에만 반영됩니다.
                  Figure Note·추출 청크·ChromaDB는 변경하지
                  않으며, 저장한 항목은 ‘검색 반영 대기’로
                  표시됩니다.
                </div>
              </div>
            </div>

            <div className="sticky bottom-0 flex items-center justify-end gap-2 border-t border-slate-200 bg-white px-5 py-4">
              <button
                type="button"
                onClick={() => setSelected(null)}
                className="rounded-xl border border-slate-200 px-4 py-2 text-sm font-semibold"
              >
                닫기
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={saveSelected}
                className="rounded-xl bg-slate-900 px-5 py-2 text-sm font-semibold text-white disabled:opacity-50"
              >
                후보 저장
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
