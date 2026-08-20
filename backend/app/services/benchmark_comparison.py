from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


COMPARISON_METRICS = (
    "answer_accuracy",
    "hallucination_rate",
    "exact_refusal_rate",
    "retrieval_document_recall_at_k",
    "retrieval_page_recall_at_k",
    "figure_answer_accuracy",
    "figure_retrieval_accuracy",
    "average_total_seconds",
)


class BenchmarkComparisonError(ValueError):
    pass


def build_benchmark_comparison(
    payloads: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    runs = [dict(payload) for payload in payloads]
    if len(runs) < 2:
        raise BenchmarkComparisonError(
            "비교표에는 최소 2개의 benchmark 실행 결과가 필요합니다."
        )

    conditions = [
        str(run.get("condition") or run.get("model") or "").strip()
        for run in runs
    ]
    if any(not condition for condition in conditions):
        raise BenchmarkComparisonError("모든 실행 결과에 condition 또는 model이 필요합니다.")
    if len(set(conditions)) != len(conditions):
        raise BenchmarkComparisonError("비교 조건 이름이 중복되었습니다.")

    question_counts = {
        int(run.get("question_count") or 0) for run in runs
    }
    if len(question_counts) != 1:
        raise BenchmarkComparisonError(
            "동일한 질문 수로 실행한 결과만 직접 비교할 수 있습니다."
        )

    result_id_sets = [
        tuple(
            str(result.get("id") or "")
            for result in (run.get("results") or [])
        )
        for run in runs
    ]
    populated_id_sets = [ids for ids in result_id_sets if ids]
    if populated_id_sets and (
        len(populated_id_sets) != len(runs)
        or len(set(populated_id_sets)) != 1
    ):
        raise BenchmarkComparisonError(
            "동일한 질문 ID로 실행한 결과만 직접 비교할 수 있습니다."
        )

    rows = [
        _comparison_row(run, condition)
        for run, condition in zip(runs, conditions)
    ]
    baseline = rows[0]
    for row in rows:
        row["answer_accuracy_delta_vs_baseline"] = _difference(
            row.get("answer_accuracy"),
            baseline.get("answer_accuracy"),
        )
        row["hallucination_rate_delta_vs_baseline"] = _difference(
            row.get("hallucination_rate"),
            baseline.get("hallucination_rate"),
        )

    return {
        "comparison_name": "well_test_benchmark_matrix",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_condition": baseline["condition"],
        "question_count": question_counts.pop(),
        "metrics": list(COMPARISON_METRICS),
        "rows": rows,
        "notes": [
            "delta는 첫 번째 조건을 기준으로 계산합니다.",
            "이 표는 기술 통계이며 통계적 유의성을 자동 판정하지 않습니다.",
        ],
    }


def _comparison_row(payload: Mapping[str, Any], condition: str) -> dict[str, Any]:
    summary = payload.get("summary") or {}
    row: dict[str, Any] = {
        "condition": condition,
        "mode": payload.get("mode") or "rag",
        "model": payload.get("model"),
        "run_id": payload.get("run_id"),
        "question_count": payload.get("question_count"),
        "infrastructure_errors": summary.get("infrastructure_errors"),
    }
    for metric in COMPARISON_METRICS:
        row[metric] = summary.get(metric)
    return row


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BenchmarkComparisonError",
    "COMPARISON_METRICS",
    "build_benchmark_comparison",
]
