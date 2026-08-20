import pytest

from app.services.benchmark_comparison import (
    BenchmarkComparisonError,
    build_benchmark_comparison,
)


def payload(condition: str, accuracy: float, hallucination_rate: float) -> dict:
    return {
        "condition": condition,
        "mode": "rag",
        "model": "qwen3:8b",
        "run_id": condition,
        "question_count": 32,
        "results": [{"id": f"WT-{index:03d}"} for index in range(1, 33)],
        "summary": {
            "answer_accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "exact_refusal_rate": 1.0,
            "retrieval_document_recall_at_k": 0.8,
            "retrieval_page_recall_at_k": 0.7,
            "figure_answer_accuracy": 0.75,
            "figure_retrieval_accuracy": 0.66,
            "average_total_seconds": 3.2,
            "infrastructure_errors": 0,
        },
    }


def test_builds_paper_comparison_and_baseline_deltas():
    comparison = build_benchmark_comparison(
        [
            payload("qwen3_baseline", 0.5, 0.3),
            payload("qwen3_rag", 0.8, 0.1),
        ]
    )

    assert comparison["baseline_condition"] == "qwen3_baseline"
    assert comparison["question_count"] == 32
    assert comparison["rows"][1]["answer_accuracy_delta_vs_baseline"] == pytest.approx(0.3)
    assert comparison["rows"][1]["hallucination_rate_delta_vs_baseline"] == pytest.approx(-0.2)


def test_rejects_different_question_counts():
    left = payload("left", 0.5, 0.3)
    right = payload("right", 0.8, 0.1)
    right["question_count"] = 16

    with pytest.raises(BenchmarkComparisonError, match="동일한 질문 수"):
        build_benchmark_comparison([left, right])


def test_rejects_different_question_ids():
    left = payload("left", 0.5, 0.3)
    right = payload("right", 0.8, 0.1)
    right["results"][-1]["id"] = "WT-999"

    with pytest.raises(BenchmarkComparisonError, match="동일한 질문 ID"):
        build_benchmark_comparison([left, right])
