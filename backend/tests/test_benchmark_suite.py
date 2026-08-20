import json
from pathlib import Path

import pytest

from app.services.benchmark_suite import (
    BenchmarkSuiteValidationError,
    build_benchmark_manifest,
    materialize_benchmark_items,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_benchmark_has_paper_minimum_and_balanced_types():
    raw = json.loads(
        (PROJECT_ROOT / "evaluation" / "well_test_agent_benchmark.json").read_text(
            encoding="utf-8"
        )
    )

    items = materialize_benchmark_items(raw)
    manifest = build_benchmark_manifest(items)

    assert manifest["question_count"] == 32
    assert manifest["concept_group_count"] == 16
    assert manifest["variant_count"] == 16
    assert manifest["question_type_counts"] == {
        "figure": 12,
        "hallucination": 6,
        "text": 14,
    }


def test_variant_inherits_evaluation_rules_from_base():
    base = {
        "id": "WT-001",
        "category": "flow_regime",
        "question": "base",
        "expected_behavior": "answer",
        "required_patterns": ["radial"],
        "forbidden_patterns": [],
        "preferred_pages": [1],
    }
    variants = [
        {
            "id": f"WT-{index:03d}",
            "variant_of": "WT-001",
            "question": f"variant {index}",
        }
        for index in range(2, 31)
    ]

    items = materialize_benchmark_items([base, *variants])

    assert items[-1]["required_patterns"] == ["radial"]
    assert items[-1]["concept_group"] == "WT-001"
    assert items[-1]["is_variant"] is True


def test_rejects_suite_below_minimum_question_count():
    with pytest.raises(BenchmarkSuiteValidationError, match="최소 30개"):
        materialize_benchmark_items([], minimum_questions=30)
