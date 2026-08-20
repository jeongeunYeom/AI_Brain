from scripts.run_well_test_benchmark import (
    build_summary,
    direct_ollama_answer,
    figure_retrieval_hit,
)


def test_summary_exposes_paper_metrics():
    results = [
        {
            "category": "flow_regime",
            "question_type": "text",
            "expected_behavior": "answer",
            "infrastructure_error": None,
            "initial_benchmark_passed": False,
            "final_benchmark_passed": True,
            "initial_validator_passed": False,
            "final_validator_passed": True,
            "initial_answer_passed": False,
            "final_answer_passed": True,
            "hallucination_detected": False,
            "rewrite_success": True,
            "expected_document_hit": True,
            "preferred_page_hit": True,
            "figure_retrieval_hit": None,
            "attempts": 2,
            "retrieval_seconds": 1.0,
            "generation_seconds": 2.0,
            "total_seconds": 3.0,
            "final_answer": "answer",
        },
        {
            "category": "refusal",
            "question_type": "hallucination",
            "expected_behavior": "refuse",
            "infrastructure_error": None,
            "initial_benchmark_passed": False,
            "final_benchmark_passed": False,
            "initial_validator_passed": True,
            "final_validator_passed": True,
            "initial_answer_passed": False,
            "final_answer_passed": False,
            "hallucination_detected": True,
            "rewrite_success": False,
            "expected_document_hit": None,
            "preferred_page_hit": None,
            "figure_retrieval_hit": None,
            "attempts": 1,
            "retrieval_seconds": 0.5,
            "generation_seconds": 1.0,
            "total_seconds": 1.5,
            "final_answer": "unsupported number",
        },
    ]

    summary = build_summary(results)

    assert summary["answer_accuracy"] == 0.5
    assert summary["initial_answer_accuracy"] == 0.0
    assert summary["hallucination_rate"] == 0.5
    assert summary["retrieval_document_recall_at_k"] == 1.0
    assert summary["average_total_seconds"] == 2.25
    assert summary["category_metrics"]["flow_regime"]["answer_accuracy"] == 1.0


def test_figure_retrieval_requires_a_preferred_page_hit():
    item = {"question_type": "figure", "preferred_pages": [219]}

    assert figure_retrieval_hit(item, [{"page": 219}]) is True
    assert figure_retrieval_hit(item, [{"page": 220}]) is False
    assert figure_retrieval_hit(item, []) is False
    assert figure_retrieval_hit({"question_type": "text"}, []) is None


def test_direct_ollama_answer_uses_chat_endpoint(monkeypatch):
    captured = {}

    def fake_http_json(method, url, *, payload=None, timeout=0):
        captured.update(
            method=method,
            url=url,
            payload=payload,
            timeout=timeout,
        )
        return {"message": {"content": "direct answer"}}

    monkeypatch.setattr(
        "scripts.run_well_test_benchmark.http_json",
        fake_http_json,
    )

    answer, elapsed = direct_ollama_answer(
        "http://localhost:11434/",
        model="qwen3:8b",
        question="What is radial flow?",
        timeout=12.0,
    )

    assert answer == "direct answer"
    assert elapsed >= 0
    assert captured == {
        "method": "POST",
        "url": "http://localhost:11434/api/chat",
        "payload": {
            "model": "qwen3:8b",
            "stream": False,
            "messages": [
                {"role": "user", "content": "What is radial flow?"}
            ],
        },
        "timeout": 12.0,
    }
