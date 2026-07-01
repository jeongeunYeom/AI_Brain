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
