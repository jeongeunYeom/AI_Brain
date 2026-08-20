from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


STRICT_REFUSAL = "제공된 문서 근거로는 확인할 수 없습니다."
REVIEW_REQUIRED_TEXT = "사람의 검토가 필요합니다."

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)

_NEGATION_RE = re.compile(
    r"(?:"
    r"\bnot\b|\bno\b|\bnever\b|\bincorrect\b|\bfalse\b|"
    r"\bdoes\s+not\b|\bdo\s+not\b|\bis\s+not\b|"
    r"않|아니|없|틀|잘못|정확하지"
    r")",
    re.IGNORECASE,
)


@dataclass
class BenchmarkEvaluation:
    passed: bool
    answer_passed: bool
    hallucination_detected: bool
    behavior_passed: bool
    expected_document_hit: bool | None
    preferred_page_hit: bool | None
    required_failures: list[str]
    forbidden_hits: list[str]
    source_pages: list[int]
    source_documents: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(value: str) -> str:
    value = (value or "").translate(_DASH_TRANSLATION).lower()
    value = value.replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def normalize_document(value: str) -> str:
    name = Path(str(value or "")).name
    name = re.sub(r"^[0-9a-f]{64}_", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    return normalize_text(name)


def _source_payloads(
    sources: Iterable[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    return [
        source
        for source in (sources or [])
        if isinstance(source, Mapping)
    ]


def _source_pages(
    sources: list[Mapping[str, Any]],
) -> list[int]:
    pages: set[int] = set()
    for source in sources:
        page = source.get("page")
        if page is None:
            continue
        try:
            pages.add(int(page))
        except (TypeError, ValueError):
            continue
    return sorted(pages)


def _source_documents(
    sources: list[Mapping[str, Any]],
) -> list[str]:
    documents = {
        str(source.get("document") or "").strip()
        for source in sources
        if str(source.get("document") or "").strip()
    }
    return sorted(documents)


def _sentence_chunks(value: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+|[\n;]+",
            normalize_text(value),
        )
        if part.strip()
    ]


def _required_failures(
    patterns: Iterable[str],
    answer: str,
) -> list[str]:
    normalized = normalize_text(answer)
    return [
        pattern
        for pattern in patterns
        if re.search(
            pattern,
            normalized,
            re.IGNORECASE | re.DOTALL,
        )
        is None
    ]


def _forbidden_hits(
    patterns: Iterable[str],
    answer: str,
) -> list[str]:
    hits: list[str] = []

    for pattern in patterns:
        for sentence in _sentence_chunks(answer):
            if re.search(
                pattern,
                sentence,
                re.IGNORECASE | re.DOTALL,
            ) is None:
                continue
            if _NEGATION_RE.search(sentence):
                continue
            hits.append(pattern)
            break

    return hits


def evaluate_benchmark_answer(
    item: Mapping[str, Any],
    answer: str,
    *,
    sources: Iterable[Mapping[str, Any]] | None = None,
) -> BenchmarkEvaluation:
    source_list = _source_payloads(sources)
    source_pages = _source_pages(source_list)
    source_documents = _source_documents(source_list)

    expected_behavior = str(
        item.get("expected_behavior") or "answer"
    )
    stripped_answer = str(answer or "").strip()

    if expected_behavior == "refuse":
        behavior_passed = stripped_answer == STRICT_REFUSAL
    elif expected_behavior == "partial_answer":
        behavior_passed = (
            stripped_answer != STRICT_REFUSAL
            and REVIEW_REQUIRED_TEXT not in stripped_answer
        )
    else:
        behavior_passed = (
            bool(stripped_answer)
            and stripped_answer != STRICT_REFUSAL
            and REVIEW_REQUIRED_TEXT not in stripped_answer
        )

    required_failures = _required_failures(
        item.get("required_patterns") or [],
        stripped_answer,
    )
    forbidden_hits = _forbidden_hits(
        item.get("forbidden_patterns") or [],
        stripped_answer,
    )

    expected_document = item.get("expected_document")
    if expected_document:
        expected_key = normalize_document(
            str(expected_document)
        )
        expected_document_hit = any(
            expected_key in normalize_document(document)
            or normalize_document(document) in expected_key
            for document in source_documents
        )
    else:
        expected_document_hit = None

    preferred_pages = {
        int(page)
        for page in (item.get("preferred_pages") or [])
    }
    if preferred_pages:
        preferred_page_hit = bool(
            preferred_pages.intersection(source_pages)
        )
    else:
        preferred_page_hit = None

    answer_passed = (
        behavior_passed
        and not required_failures
        and not forbidden_hits
    )
    passed = (
        answer_passed
        and expected_document_hit is not False
    )
    hallucination_detected = bool(forbidden_hits) or (
        expected_behavior == "refuse" and not behavior_passed
    )

    return BenchmarkEvaluation(
        passed=passed,
        answer_passed=answer_passed,
        hallucination_detected=hallucination_detected,
        behavior_passed=behavior_passed,
        expected_document_hit=expected_document_hit,
        preferred_page_hit=preferred_page_hit,
        required_failures=required_failures,
        forbidden_hits=forbidden_hits,
        source_pages=source_pages,
        source_documents=source_documents,
    )
