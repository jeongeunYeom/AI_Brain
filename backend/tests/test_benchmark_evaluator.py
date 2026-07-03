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



def test_open_hyphen_circle_matches_supercharged_rule():
    item = {
        "expected_behavior": "answer",
        "expected_document": "Well_Test_Analysis.pdf",
        "preferred_pages": [441],
        "required_patterns": [
            r"supercharg",
            r"open[- ]?circle|제외|제거",
        ],
        "forbidden_patterns": [],
    }
    result = evaluate_benchmark_answer(
        item,
        "Supercharged points는 open-circle points로 표시된다.",
        sources=[
            {
                "document": "Well_Test_Analysis.pdf",
                "page": 441,
            }
        ],
    )
    assert result.passed is True


def test_shared_rft_unit_is_accepted():
    item = {
        "expected_behavior": "answer",
        "expected_document": "Well_Test_Analysis.pdf",
        "preferred_pages": [440],
        "required_patterns": [
            r"0\.29",
            r"0\.37",
            r"0\.42",
            r"psi\s*/\s*ft",
        ],
        "forbidden_patterns": [],
    }
    result = evaluate_benchmark_answer(
        item,
        "압력 기울기는 0.29, 0.37, 0.42 psi/ft이다.",
        sources=[
            {
                "document": "Well_Test_Analysis.pdf",
                "page": 440,
            }
        ],
    )
    assert result.passed is True



def test_wt007_comparison_sentence_is_not_false_positive():
    item = {
        "id": "WT-007",
        "expected_behavior": "answer",
        "expected_document": "Well_Test_Analysis.pdf",
        "preferred_pages": [439, 440],
        "required_patterns": [
            r"0\.34",
            r"0\.29",
            r"0\.37",
            r"0\.42",
            r"psi\s*/\s*ft|psi\s*per\s*ft",
            (
                r"(pressure\s*discontinu|압력\s*불연속|"
                r"permeability\s*barrier|투과(?:율|성)\s*장벽|"
                r"differential\s*depletion|차별적\s*고갈|"
                r"여러.{0,60}(gradient|구배|기울기)|"
                r"압력\s*분포.{0,80}(변화|달라))"
            ),
        ],
        "forbidden_patterns": [
            (
                r"(figure\s*2|appraisal)"
                r"(?![^.\n]{0,320}"
                r"(figure\s*3|after\s*significant\s*production|"
                r"rft\s*survey\s*after|생산\s*후))"
                r"[^.\n]{0,320}"
                r"0\.29[^.\n]{0,140}"
                r"0\.37[^.\n]{0,140}"
                r"0\.42"
            ),
        ],
    }

    answer = (
        "Figure 2 Appraisal RFT는 0.34 psi/ft이다. "
        "Figure 3 RFT Survey after Significant Production은 "
        "0.29, 0.37, 0.42 psi/ft이며 투과성 장벽으로 "
        "구분된다. 두 조사를 비교하면 Appraisal은 0.34 "
        "psi/ft이고 생산 후 조사는 0.29, 0.37, 0.42 "
        "psi/ft이다."
    )

    result = evaluate_benchmark_answer(
        item,
        answer,
        sources=[
            {
                "document": "Well_Test_Analysis.pdf",
                "page": 439,
            },
            {
                "document": "Well_Test_Analysis.pdf",
                "page": 440,
            },
        ],
    )

    assert result.passed is True
    assert result.forbidden_hits == []


def test_wt007_real_appraisal_misassignment_still_fails():
    item = {
        "id": "WT-007",
        "expected_behavior": "answer",
        "expected_document": "Well_Test_Analysis.pdf",
        "preferred_pages": [439, 440],
        "required_patterns": [],
        "forbidden_patterns": [
            (
                r"(figure\s*2|appraisal)"
                r"(?![^.\n]{0,320}"
                r"(figure\s*3|after\s*significant\s*production|"
                r"rft\s*survey\s*after|생산\s*후))"
                r"[^.\n]{0,320}"
                r"0\.29[^.\n]{0,140}"
                r"0\.37[^.\n]{0,140}"
                r"0\.42"
            ),
        ],
    }

    answer = (
        "Figure 2 Appraisal RFT에는 "
        "0.29, 0.37, 0.42 psi/ft 세 구간이 있다."
    )

    result = evaluate_benchmark_answer(
        item,
        answer,
        sources=[
            {
                "document": "Well_Test_Analysis.pdf",
                "page": 439,
            }
        ],
    )

    assert result.passed is False
    assert result.forbidden_hits
