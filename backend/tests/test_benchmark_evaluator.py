from app.services.benchmark_evaluator import (
    STRICT_REFUSAL,
    evaluate_benchmark_answer,
)


WT1 = {
    "id": "WT-001",
    "expected_behavior": "answer",
    "expected_document": "Well_Test_Analysis.pdf",
    "preferred_pages": [219],
    "required_patterns": [
        (
            r"(wellbore\s*storage|유정\s*저장).{0,240}"
            r"(pressure|압력).{0,100}(derivative|미분)"
            r".{0,100}(겹|overlap)"
        ),
        (
            r"(radial\s*flow|방사\s*유동).{0,240}"
            r"(derivative|미분).{0,100}"
            r"(plateau|평탄|수평|일정)"
        ),
    ],
    "forbidden_patterns": [
        (
            r"(radial\s*flow|방사\s*유동).{0,180}"
            r"(pressure|압력).{0,120}"
            r"(unit[- ]?slope|단위\s*기울기)"
        )
    ],
}

SOURCES = [
    {
        "document": "Well_Test_Analysis.pdf",
        "page": 219,
    }
]


def test_correct_answer_passes():
    answer = (
        "Wellbore storage에서는 pressure와 derivative가 "
        "서로 겹쳐 unit-slope diagonal을 따른다. "
        "Radial flow에서는 derivative가 일정해져 "
        "수평 plateau를 형성한다."
    )
    result = evaluate_benchmark_answer(
        WT1,
        answer,
        sources=SOURCES,
    )
    assert result.passed is True
    assert result.preferred_page_hit is True
    assert result.expected_document_hit is True


def test_affirmative_forbidden_claim_fails():
    answer = (
        "Wellbore storage에서는 pressure와 derivative가 "
        "서로 겹친다. Radial flow에서는 pressure가 "
        "unit-slope를 따른다. Radial flow derivative는 "
        "plateau를 형성한다."
    )
    result = evaluate_benchmark_answer(
        WT1,
        answer,
        sources=SOURCES,
    )
    assert result.passed is False
    assert result.forbidden_hits


def test_negated_correction_does_not_trigger_forbidden():
    answer = (
        "Wellbore storage에서는 pressure와 derivative가 "
        "서로 겹친다. Radial flow에서는 pressure가 "
        "unit-slope를 따르지 않는다. Radial flow에서는 "
        "derivative가 수평 plateau를 형성한다."
    )
    result = evaluate_benchmark_answer(
        WT1,
        answer,
        sources=SOURCES,
    )
    assert result.forbidden_hits == []


def test_exact_refusal_behavior():
    item = {
        "expected_behavior": "refuse",
        "expected_document": None,
        "preferred_pages": [],
        "required_patterns": [],
        "forbidden_patterns": [],
    }
    passed = evaluate_benchmark_answer(
        item,
        STRICT_REFUSAL,
        sources=[],
    )
    failed = evaluate_benchmark_answer(
        item,
        "문서에는 없지만 14,000 m3/day입니다.",
        sources=[],
    )
    assert passed.passed is True
    assert failed.passed is False


def test_missing_expected_document_fails():
    answer = (
        "Wellbore storage에서는 pressure와 derivative가 "
        "겹친다. Radial flow derivative는 plateau를 "
        "형성한다."
    )
    result = evaluate_benchmark_answer(
        WT1,
        answer,
        sources=[
            {
                "document": "Other_Document.pdf",
                "page": 219,
            }
        ],
    )
    assert result.passed is False
    assert result.expected_document_hit is False
