from app.services.engineering_validator import (
    EngineeringValidator,
    STRICT_REFUSAL,
)


SOURCES = [
    {
        "document": "Heriot-Watt_University_-_Well_Test_Analysis.pdf",
        "page": 219,
        "chunk_id": "p219:c0",
    }
]


def test_correct_regime_answer_passes():
    answer = (
        "Wellbore storage에서는 pressure와 pressure derivative가 "
        "겹치며 unit-slope diagonal을 따른다. "
        "Radial flow의 middle-time region에서는 pressure derivative가 "
        "수평 plateau를 형성한다. "
        "[Well Test Analysis, p.219]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        "Wellbore storage와 radial flow를 구분해줘.",
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is True
    assert result.errors == []


def test_affirmative_radial_pressure_unit_slope_is_blocked():
    answer = (
        "Radial flow에서는 pressure가 unit-slope를 따른다. "
        "Wellbore storage에서는 pressure와 derivative가 겹치며 "
        "unit-slope를 따른다. "
        "Radial flow derivative는 plateau를 형성한다. "
        "[Well Test Analysis, p.219]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        "Wellbore storage와 radial flow를 구분해줘.",
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is False
    assert "WT-RADIAL-001" in result.rule_ids


def test_negated_bad_claim_is_not_blocked():
    answer = (
        "Radial flow에서 pressure가 unit-slope를 따르는 것은 아니다. "
        "Wellbore storage에서는 pressure와 derivative가 겹치며 "
        "unit-slope를 따른다. "
        "Radial flow에서는 derivative가 수평 plateau를 형성한다. "
        "[Well Test Analysis, p.219]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        "잘못된 설명을 검토해줘. Radial flow에서 pressure가 "
        "unit-slope를 따른다.",
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is True


def test_missing_radial_plateau_fails():
    answer = (
        "Wellbore storage에서는 pressure와 derivative가 겹치며 "
        "unit-slope를 따른다. [Well Test Analysis, p.219]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        "Wellbore storage와 radial flow를 구분해줘.",
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is False
    assert "WT-RADIAL-003" in result.rule_ids


def test_no_sources_requires_exact_refusal():
    validator = EngineeringValidator()

    passed = validator.validate_well_test_answer(
        "Johansen Formation 값을 알려줘.",
        STRICT_REFUSAL,
        retrieved_sources=[],
    )
    failed = validator.validate_well_test_answer(
        "Johansen Formation 값을 알려줘.",
        "주입률은 14,000 m3/day입니다.",
        retrieved_sources=[],
    )

    assert passed.passed is True
    assert failed.passed is False
    assert "WT-EVIDENCE-001" in failed.rule_ids


def test_missing_citation_is_warning_not_error():
    answer = (
        "Wellbore storage에서는 pressure와 derivative가 겹치며 "
        "unit-slope를 따른다. Radial flow에서는 derivative가 "
        "수평 plateau를 형성한다."
    )
    result = EngineeringValidator().validate_well_test_answer(
        "Wellbore storage와 radial flow를 구분해줘.",
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is True
    assert result.warnings



def test_radial_derivative_unit_slope_is_blocked():
    answer = (
        "Wellbore storage에서는 pressure와 derivative가 겹치며 "
        "unit-slope를 따른다. "
        "Radial flow에서는 pressure derivative가 unit-slope를 "
        "따르면서 plateau를 형성한다. "
        "[Well Test Analysis, p.219]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        "Wellbore storage와 radial flow를 구분해줘.",
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is False
    assert "WT-RADIAL-005" in result.rule_ids
    assert "WT-RADIAL-001" not in result.rule_ids


def test_pressure_derivative_is_not_bare_pressure():
    validator = EngineeringValidator()
    assert validator._mentions_derivative(
        "pressure derivative가 plateau를 형성한다"
    )
    assert not validator._mentions_pressure(
        "pressure derivative가 plateau를 형성한다"
    )



def test_supported_rft_comparison_refusal_is_blocked():
    sources = [
        {
            "document": "Well_Test_Analysis.pdf",
            "page": 440,
            "excerpt": (
                "title: Appraisal Well RFT Survey\n"
                "title: RFT Survey after Significant Production"
            ),
        }
    ]
    result = EngineeringValidator().validate_well_test_answer(
        (
            "Appraisal Well RFT Survey와 RFT Survey after "
            "Significant Production을 비교해줘."
        ),
        STRICT_REFUSAL,
        retrieved_sources=sources,
    )
    assert result.passed is False
    assert "WT-RFT-001" in result.rule_ids



def test_unit_slope_radial_association_is_blocked():
    answer = (
        "Unit-slope diagonal은 방사형 유동에서 나타난다. "
        "압력과 도함수 응답은 겹친다. "
        "[Well Test Analysis, p.219]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        (
            "Log-log diagnostic plot에서 unit-slope diagonal은 "
            "어떤 유동 구간을 의미하는지 설명해줘."
        ),
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is False
    assert "WT-UNIT-002" in result.rule_ids


def test_rft_comparison_requires_all_gradients():
    sources = [
        {
            "document": "Well_Test_Analysis.pdf",
            "page": 440,
            "excerpt": (
                "Appraisal Well RFT Survey. "
                "RFT Survey after Significant Production."
            ),
        }
    ]
    answer = (
        "Figure 3은 0.29, 0.37, 0.42 psi/ft를 보인다. "
        "[Well Test Analysis, p.440]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        (
            "Appraisal Well RFT Survey와 RFT Survey after "
            "Significant Production을 비교해줘."
        ),
        answer,
        retrieved_sources=sources,
    )
    assert result.passed is False
    assert "WT-RFT-002" in result.rule_ids


def test_open_circle_supercharged_answer_passes():
    answer = (
        "Supercharged points는 open-circle points로 표시된다. "
        "[Well Test Analysis, p.441]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        (
            "RFT 그래프에서 supercharged points가 어떻게 "
            "표시되거나 처리되었는지 설명해줘."
        ),
        answer,
        retrieved_sources=SOURCES,
    )
    assert "WT-RFT-008" not in result.rule_ids


def test_fracture_omission_is_blocked():
    answer = (
        "Wellbore storage는 unit-slope를 보이고 radial flow는 "
        "derivative plateau를 보인다. "
        "[Well Test Analysis, p.219]"
    )
    result = EngineeringValidator().validate_well_test_answer(
        (
            "Wellbore storage, radial flow, fracture flow의 "
            "pressure derivative 특징을 설명해줘."
        ),
        answer,
        retrieved_sources=SOURCES,
    )
    assert result.passed is False
    assert "WT-FRACTURE-001" in result.rule_ids
