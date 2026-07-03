from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


STRICT_REFUSAL = "제공된 문서 근거로는 확인할 수 없습니다."

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
    r"않(?:다|는다|습니다|음)?|아니(?:다|며|고|라고)?|"
    r"없(?:다|습니다|음)?|틀(?:리|렸|린)|잘못|정확하지\s*않"
    r")",
    re.IGNORECASE,
)

_CITATION_RE = re.compile(
    r"\[[^\]]*(?:p\.?\s*\d+|page\s*\d+)[^\]]*\]",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)


class EngineeringValidator:
    """Deterministic first-pass validator for Well Test answers."""

    def validate_well_test_answer(
        self,
        question: str,
        answer: str,
        *,
        retrieved_sources: Iterable[Mapping[str, Any]] | None = None,
    ) -> ValidationResult:
        question_n = self._normalize(question)
        answer_n = self._normalize(answer)
        sources = list(retrieved_sources or [])

        errors: list[str] = []
        warnings: list[str] = []
        rule_ids: list[str] = []

        if not sources:
            if answer.strip() != STRICT_REFUSAL:
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-EVIDENCE-001",
                    "검색 근거가 없는데 고정 거절문 대신 내용을 생성했습니다.",
                )
            return ValidationResult(
                passed=not errors,
                errors=errors,
                warnings=warnings,
                rule_ids=rule_ids,
            )

        asks_rft_comparison = (
            "rft" in question_n
            and (
                "significant production" in question_n
                or "생산" in question_n
            )
            and (
                "appraisal" in question_n
                or "비교" in question_n
            )
        )
        source_text = " ".join(
            str(source.get("excerpt") or "")
            for source in sources
        ).lower()
        has_rft_comparison_evidence = (
            "appraisal well rft survey" in source_text
            and (
                "rft survey after significant production"
                in source_text
            )
        )
        if (
            asks_rft_comparison
            and has_rft_comparison_evidence
            and answer.strip() == STRICT_REFUSAL
        ):
            self._add_error(
                errors,
                rule_ids,
                "WT-RFT-001",
                "RFT 전후 비교 근거가 검색됐지만 전체 답변을 거절했습니다.",
            )

        for sentence in self._sentences(answer_n):
            if not self._mentions_radial_flow(sentence):
                continue

            if (
                self._mentions_derivative(sentence)
                and self._mentions_unit_slope(sentence)
                and not self._is_negated_or_corrective(sentence)
            ):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RADIAL-005",
                    "Radial flow에서 pressure derivative가 unit-slope를 따른다고 설명했습니다.",
                )

            if (
                self._mentions_pressure(sentence)
                and self._mentions_unit_slope(sentence)
                and not self._is_negated_or_corrective(sentence)
            ):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RADIAL-001",
                    "Radial flow에서 pressure가 unit-slope를 따른다고 설명했습니다.",
                )

            if (
                self._mentions_pressure(sentence)
                and self._mentions_derivative(sentence)
                and self._mentions_overlap(sentence)
                and not self._is_negated_or_corrective(sentence)
            ):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RADIAL-002",
                    "Radial flow에서 pressure와 derivative가 겹친다고 설명했습니다.",
                )

        asks_regime_comparison = (
            self._mentions_wellbore_storage(question_n)
            and self._mentions_radial_flow(question_n)
        )
        asks_false_radial_claim = (
            self._mentions_radial_flow(question_n)
            and self._mentions_unit_slope(question_n)
            and self._mentions_pressure(question_n)
        )

        if asks_regime_comparison or asks_false_radial_claim:
            if not self._has_wbs_overlap(answer_n):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-WBS-001",
                    "Wellbore storage에서 pressure와 derivative가 겹친다는 설명이 없습니다.",
                )
            if not self._has_wbs_unit_slope(answer_n):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-WBS-002",
                    "Wellbore storage의 unit-slope 설명이 없습니다.",
                )
            if not self._has_radial_plateau(answer_n):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RADIAL-003",
                    "Radial flow의 pressure derivative plateau 설명이 없습니다.",
                )

        asks_plateau = bool(
            re.search(
                r"(derivative|미분|도함수).{0,80}(plateau|평탄|수평)|"
                r"(plateau|평탄|수평).{0,80}(derivative|미분|도함수)",
                question_n,
                re.IGNORECASE,
            )
        )
        if asks_plateau and not self._has_radial_plateau(answer_n):
            self._add_error(
                errors,
                rule_ids,
                "WT-RADIAL-004",
                "Derivative plateau를 radial flow 또는 MTR과 연결하지 않았습니다.",
            )


        asks_unit_slope_meaning = bool(
            re.search(
                r"unit[- ]?slope|단위\s*기울기",
                question_n,
                re.IGNORECASE,
            )
            and re.search(
                r"의미|어떤\s*유동|무슨\s*구간|설명|meaning|which\s*flow",
                question_n,
                re.IGNORECASE,
            )
        )
        if asks_unit_slope_meaning:
            if not self._has_wbs_overlap(answer_n):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-UNIT-001",
                    "Unit-slope를 wellbore storage의 pressure-derivative overlap과 연결하지 않았습니다.",
                )

            misleading_unit_slope = any(
                self._mentions_unit_slope(sentence)
                and re.search(
                    r"radial\s*flow|방사\s*유동|방사형|"
                    r"middle[- ]?time|\bmtr\b|중기|"
                    r"final\s*stage|last\s*stage|마지막\s*단계",
                    sentence,
                    re.IGNORECASE,
                )
                and not self._is_negated_or_corrective(sentence)
                for sentence in self._sentences(answer_n)
            )
            if misleading_unit_slope:
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-UNIT-002",
                    "Unit-slope를 radial flow, MTR 또는 마지막 유동 단계의 특징으로 설명했습니다.",
                )

        def has_number(value: str) -> bool:
            return re.search(
                rf"(?<!\d){re.escape(value)}(?!\d)",
                answer_n,
            ) is not None

        has_psi_per_ft = bool(
            re.search(
                r"psi\s*/\s*ft|psi\s*per\s*ft",
                answer_n,
                re.IGNORECASE,
            )
        )

        if asks_rft_comparison and has_rft_comparison_evidence:
            if not (has_number("0.34") and has_psi_per_ft):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RFT-002",
                    "Appraisal RFT의 0.34 psi/ft 압력 기울기 설명이 없습니다.",
                )

            after_values_present = all(
                has_number(value)
                for value in ("0.29", "0.37", "0.42")
            )
            if not (after_values_present and has_psi_per_ft):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RFT-003",
                    "생산 후 RFT의 0.29, 0.37, 0.42 psi/ft 기울기를 모두 설명하지 않았습니다.",
                )

            if re.search(
                r"(정확한|specific).{0,80}"
                r"(기울기|gradient).{0,100}"
                r"(확인할\s*수\s*없|not\s*available|not\s*provided)",
                answer_n,
                re.IGNORECASE,
            ):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RFT-004",
                    "문서에 기울기 수치가 있는데도 정확한 값이 없다고 설명했습니다.",
                )

        asks_figure_two_three = bool(
            "rft" in question_n
            and re.search(
                r"figure\s*2|도\s*2",
                question_n,
                re.IGNORECASE,
            )
            and re.search(
                r"figure\s*3|도\s*3",
                question_n,
                re.IGNORECASE,
            )
        )
        if asks_figure_two_three:
            figure_two_ok = bool(
                re.search(
                    r"figure\s*2.{0,500}0\.34",
                    answer_n,
                    re.IGNORECASE,
                )
            )
            figure_three_ok = bool(
                re.search(
                    r"figure\s*3.{0,700}0\.29"
                    r".{0,160}0\.37.{0,160}0\.42",
                    answer_n,
                    re.IGNORECASE,
                )
            )
            if not figure_two_ok:
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RFT-005",
                    "Figure 2에 Appraisal RFT의 0.34 psi/ft를 배치하지 않았습니다.",
                )
            if not figure_three_ok:
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RFT-006",
                    "Figure 3에 생산 후 0.29, 0.37, 0.42 psi/ft를 배치하지 않았습니다.",
                )
            if re.search(
                r"figure\s*31|well\s*13a|211/19a-7",
                answer_n,
                re.IGNORECASE,
            ):
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RFT-007",
                    "Figure 2·3 질문에 다른 Figure 또는 다른 유정 정보를 혼합했습니다.",
                )

        asks_supercharged = "supercharg" in question_n
        if asks_supercharged:
            has_display_or_handling = bool(
                re.search(
                    r"open[- ]?circle|개방된\s*원|빈\s*원|"
                    r"exclude|excluded|remove|removed|"
                    r"disregard|제외|제거",
                    answer_n,
                    re.IGNORECASE,
                )
            )
            if not has_display_or_handling:
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-RFT-008",
                    "Supercharged point의 표시 또는 제외·제거 방식을 설명하지 않았습니다.",
                )

        asks_fracture_flow = bool(
            re.search(
                r"fracture\s*flow|균열\s*유동",
                question_n,
                re.IGNORECASE,
            )
        )
        if asks_fracture_flow:
            fracture_handled = bool(
                re.search(
                    r"(fracture|균열).{0,260}"
                    r"(확인|근거|linear|half[- ]?slope|"
                    r"1/2|절반|제공된\s*문서)",
                    answer_n,
                    re.IGNORECASE,
                )
            )
            if not fracture_handled:
                self._add_error(
                    errors,
                    rule_ids,
                    "WT-FRACTURE-001",
                    "Fracture flow를 설명하거나 해당 항목의 문서 근거 부족을 명시하지 않았습니다.",
                )

        if not _CITATION_RE.search(answer):
            warnings.append(
                "검색 근거가 있지만 답변 본문에 문서·페이지 인용 표지가 없습니다."
            )

        return ValidationResult(
            passed=not errors,
            errors=errors,
            warnings=warnings,
            rule_ids=rule_ids,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        value = (value or "").translate(_DASH_TRANSLATION).lower()
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _sentences(value: str) -> list[str]:
        return [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+|[\n;]+", value)
            if item.strip()
        ]

    @staticmethod
    def _is_negated_or_corrective(sentence: str) -> bool:
        return bool(_NEGATION_RE.search(sentence))

    @staticmethod
    def _mentions_wellbore_storage(value: str) -> bool:
        return bool(
            re.search(
                r"wellbore\s*storage|유정\s*저장|웰보어\s*스토리지",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _mentions_radial_flow(value: str) -> bool:
        return bool(
            re.search(
                r"radial\s*flow|방사\s*유동|방사형",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _mentions_pressure(value: str) -> bool:
        return bool(
            re.search(
                r"pressure(?!\s*derivative)|"
                r"압력(?!\s*(?:derivative|미분))",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _mentions_derivative(value: str) -> bool:
        return bool(
            re.search(
                r"pressure\s*derivative|derivative|미분|도함수",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _mentions_unit_slope(value: str) -> bool:
        return bool(
            re.search(
                r"unit[- ]?slope|단위\s*기울기",
                value,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _mentions_overlap(value: str) -> bool:
        return bool(
            re.search(
                r"overlap|coincid|겹(?:치|쳐|친|침|친다|칩니다)",
                value,
                re.IGNORECASE,
            )
        )

    def _has_wbs_overlap(self, answer: str) -> bool:
        return bool(
            re.search(
                r"(wellbore\s*storage|유정\s*저장|웰보어\s*스토리지)"
                r".{0,220}(pressure|압력).{0,120}"
                r"(derivative|미분|도함수).{0,100}(overlap|coincid|겹)",
                answer,
                re.IGNORECASE,
            )
            or re.search(
                r"(pressure|압력).{0,120}(derivative|미분|도함수).{0,100}"
                r"(overlap|coincid|겹).{0,220}"
                r"(wellbore\s*storage|유정\s*저장|웰보어\s*스토리지)",
                answer,
                re.IGNORECASE,
            )
        )

    def _has_wbs_unit_slope(self, answer: str) -> bool:
        return bool(
            re.search(
                r"(wellbore\s*storage|유정\s*저장|웰보어\s*스토리지)"
                r".{0,260}(unit[- ]?slope|단위\s*기울기)",
                answer,
                re.IGNORECASE,
            )
            or re.search(
                r"(unit[- ]?slope|단위\s*기울기).{0,260}"
                r"(wellbore\s*storage|유정\s*저장|웰보어\s*스토리지)",
                answer,
                re.IGNORECASE,
            )
        )

    def _has_radial_plateau(self, answer: str) -> bool:
        return bool(
            re.search(
                r"(radial\s*flow|방사\s*유동|방사형|middle[- ]?time|\bmtr\b|중기)"
                r".{0,260}(pressure\s*derivative|derivative|미분|도함수)"
                r".{0,120}(plateau|constant|평탄|수평|일정)",
                answer,
                re.IGNORECASE,
            )
            or re.search(
                r"(pressure\s*derivative|derivative|미분|도함수)"
                r".{0,120}(plateau|constant|평탄|수평|일정)"
                r".{0,260}(radial\s*flow|방사\s*유동|방사형|middle[- ]?time|\bmtr\b|중기)",
                answer,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _add_error(
        errors: list[str],
        rule_ids: list[str],
        rule_id: str,
        message: str,
    ) -> None:
        if message not in errors:
            errors.append(message)
        if rule_id not in rule_ids:
            rule_ids.append(rule_id)
