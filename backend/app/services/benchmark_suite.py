from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


BENCHMARK_MINIMUM_QUESTIONS = 30

_ALLOWED_BEHAVIORS = {"answer", "partial_answer", "refuse"}
_ALLOWED_QUESTION_TYPES = {"text", "figure", "hallucination"}
_FIGURE_CATEGORIES = {
    "figure_mixing",
    "figure_semantics",
    "numeric_extraction",
    "plot_type",
    "rft_comparison",
    "rft_supercharging",
}


class BenchmarkSuiteValidationError(ValueError):
    pass


def infer_question_type(item: Mapping[str, Any]) -> str:
    explicit = str(item.get("question_type") or "").strip().lower()
    if explicit:
        return explicit
    if str(item.get("expected_behavior") or "") == "refuse":
        return "hallucination"
    if str(item.get("category") or "") == "false_premise":
        return "hallucination"
    if str(item.get("category") or "") in _FIGURE_CATEGORIES:
        return "figure"
    return "text"


def materialize_benchmark_items(
    raw_items: Iterable[Mapping[str, Any]],
    *,
    minimum_questions: int = BENCHMARK_MINIMUM_QUESTIONS,
) -> list[dict[str, Any]]:
    items = [dict(item) for item in raw_items]
    if len(items) < minimum_questions:
        raise BenchmarkSuiteValidationError(
            f"Benchmark에는 최소 {minimum_questions}개 질문이 필요합니다. "
            f"현재 {len(items)}개입니다."
        )

    ids = [str(item.get("id") or "").strip() for item in items]
    missing_ids = [index + 1 for index, value in enumerate(ids) if not value]
    if missing_ids:
        raise BenchmarkSuiteValidationError(
            f"Benchmark ID가 없는 항목이 있습니다: {missing_ids}"
        )
    duplicates = sorted(
        value for value, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise BenchmarkSuiteValidationError(
            f"중복 Benchmark ID가 있습니다: {', '.join(duplicates)}"
        )

    by_id = {item_id: item for item_id, item in zip(ids, items)}
    resolved: list[dict[str, Any]] = []
    for item_id, item in zip(ids, items):
        variant_of = str(item.get("variant_of") or "").strip()
        if variant_of:
            base = by_id.get(variant_of)
            if base is None:
                raise BenchmarkSuiteValidationError(
                    f"{item_id}의 원본 질문을 찾지 못했습니다: {variant_of}"
                )
            if base.get("variant_of"):
                raise BenchmarkSuiteValidationError(
                    f"Variant는 다른 variant를 상속할 수 없습니다: {item_id}"
                )
            materialized = {**base, **item}
        else:
            materialized = dict(item)

        materialized["id"] = item_id
        materialized["question_type"] = infer_question_type(materialized)
        materialized["concept_group"] = str(
            materialized.get("concept_group") or variant_of or item_id
        )
        materialized["is_variant"] = bool(variant_of)
        _validate_materialized_item(materialized)
        resolved.append(materialized)

    return resolved


def build_benchmark_manifest(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    item_list = list(items)
    question_types = Counter(str(item["question_type"]) for item in item_list)
    categories = Counter(str(item.get("category") or "unknown") for item in item_list)
    concept_groups = {
        str(item.get("concept_group") or item.get("id")) for item in item_list
    }
    return {
        "question_count": len(item_list),
        "concept_group_count": len(concept_groups),
        "variant_count": sum(bool(item.get("is_variant")) for item in item_list),
        "question_type_counts": dict(sorted(question_types.items())),
        "category_counts": dict(sorted(categories.items())),
    }


def _validate_materialized_item(item: Mapping[str, Any]) -> None:
    item_id = str(item["id"])
    if not str(item.get("question") or "").strip():
        raise BenchmarkSuiteValidationError(f"질문이 비어 있습니다: {item_id}")
    behavior = str(item.get("expected_behavior") or "")
    if behavior not in _ALLOWED_BEHAVIORS:
        raise BenchmarkSuiteValidationError(
            f"지원하지 않는 expected_behavior입니다: {item_id} ({behavior})"
        )
    question_type = str(item.get("question_type") or "")
    if question_type not in _ALLOWED_QUESTION_TYPES:
        raise BenchmarkSuiteValidationError(
            f"지원하지 않는 question_type입니다: {item_id} ({question_type})"
        )
    for field in ("required_patterns", "forbidden_patterns", "preferred_pages"):
        if not isinstance(item.get(field), list):
            raise BenchmarkSuiteValidationError(
                f"{field}는 배열이어야 합니다: {item_id}"
            )


__all__ = [
    "BENCHMARK_MINIMUM_QUESTIONS",
    "BenchmarkSuiteValidationError",
    "build_benchmark_manifest",
    "infer_question_type",
    "materialize_benchmark_items",
]
