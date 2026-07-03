from __future__ import annotations

import asyncio
import json

from app.core.config import Settings
from app.models.schemas import FigureReference, Source
from app.services.engineering_validator import ValidationResult
from app.services.qa import QAService
from app.services.query_router import QueryType


HIT = {
    "id": "well-test:p219:c0",
    "text": (
        "Wellbore storage dominated flow: pressure and "
        "pressure derivative overlap on a unit-slope diagonal. "
        "Middle-time radial flow: pressure derivative forms "
        "a constant plateau."
    ),
    "metadata": {
        "document": "Well_Test_Analysis.pdf",
        "page": 219,
    },
    "score": 0.95,
    "vector_score": 0.92,
    "keyword_score": 0.88,
}

SOURCE = Source(
    document="Well_Test_Analysis.pdf",
    page=219,
    chunk_id="well-test:p219:c0",
    score=0.95,
    vector_score=0.92,
    keyword_score=0.88,
    excerpt=HIT["text"],
    preview=HIT["text"],
)


class FakeVectorStore:
    pass


class FakeOllama:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.messages = []

    async def chat(self, messages, model=None):
        self.messages.append(
            [dict(message) for message in messages]
        )
        answer = self.answers[
            min(self.calls, len(self.answers) - 1)
        ]
        self.calls += 1
        return answer


def make_service(tmp_path, answers):
    settings = Settings(data_dir=tmp_path)
    ollama = FakeOllama(answers)
    service = QAService(
        settings,
        FakeVectorStore(),
        ollama,
    )
    return service, ollama


def prepared_payload():
    return {
        "aggregate": None,
        "query_type": QueryType.LOCAL_FACT_SEARCH,
        "hits": [HIT],
        "sources": [SOURCE],
        "figures": [
            FigureReference(
                document="Well_Test_Analysis.pdf",
                page=219,
                title="Type Curve",
                image_type="graph",
                filename="figure.png",
                url="/api/figures/figure.png",
            )
        ],
        "messages": [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": (
                    "Question plus one fixed retrieved chunk."
                ),
            },
        ],
        "retrieval_elapsed_seconds": 0.25,
    }


def install_prepared(service, payload):
    state = {"calls": 0}

    def fake_prepare(question, top_k):
        state["calls"] += 1
        return payload

    service._prepare_generation = fake_prepare
    return state


def load_run(tmp_path):
    files = list(
        (tmp_path / "agent_runs").glob("*.json")
    )
    assert len(files) == 1
    return json.loads(
        files[0].read_text(encoding="utf-8")
    )


def test_failed_draft_is_rewritten_with_one_retrieval(
    tmp_path,
):
    bad = (
        "Radial flow에서는 pressure가 unit-slope를 따른다. "
        "Wellbore storage에서는 pressure와 derivative가 "
        "겹치며 unit-slope를 따른다. "
        "Radial flow derivative는 plateau를 형성한다. "
        "[Well Test Analysis, p.219]"
    )
    good = (
        "Wellbore storage에서는 pressure와 pressure "
        "derivative가 겹치며 unit-slope diagonal을 따른다. "
        "Radial flow에서는 pressure derivative가 수평 "
        "plateau를 형성한다. "
        "[Well Test Analysis, p.219]"
    )
    service, ollama = make_service(
        tmp_path,
        [bad, good],
    )
    prepared = install_prepared(
        service,
        prepared_payload(),
    )

    response = asyncio.run(
        service.answer(
            "Wellbore storage와 radial flow를 구분해줘.",
            model="qwen3:8b",
            benchmark_id="WT-001",
        )
    )

    assert prepared["calls"] == 1
    assert ollama.calls == 2
    assert "pressure derivative가 수평" in response.answer
    assert ollama.messages[0] != ollama.messages[1]

    record = load_run(tmp_path)
    assert record["benchmark_id"] == "WT-001"
    assert record["initial_passed"] is False
    assert record["final_passed"] is True
    assert record["final_status"] == "completed"
    assert len(record["attempts"]) == 2
    assert record["retrieved_sources"][0]["page"] == 219


def test_persistent_failure_stops_after_three_generations(
    tmp_path,
):
    bad = (
        "Radial flow에서는 pressure가 unit-slope를 따른다. "
        "Wellbore storage에서는 pressure와 derivative가 "
        "겹치며 unit-slope를 따른다. "
        "Radial flow derivative는 plateau를 형성한다. "
        "[Well Test Analysis, p.219]"
    )
    service, ollama = make_service(
        tmp_path,
        [bad, bad, bad],
    )
    prepared = install_prepared(
        service,
        prepared_payload(),
    )

    response = asyncio.run(
        service.answer(
            "Wellbore storage와 radial flow를 구분해줘."
        )
    )

    assert prepared["calls"] == 1
    assert ollama.calls == 3
    assert "사람의 검토가 필요합니다" in response.answer

    record = load_run(tmp_path)
    assert record["final_passed"] is False
    assert record["final_status"] == "review_required"
    assert len(record["attempts"]) == 3


def test_no_hits_logs_exact_refusal_without_generation(
    tmp_path,
):
    service, ollama = make_service(
        tmp_path,
        ["should not be used"],
    )
    payload = prepared_payload()
    payload.update(
        {
            "hits": [],
            "sources": [],
            "figures": [],
            "messages": [],
        }
    )
    prepared = install_prepared(service, payload)

    response = asyncio.run(
        service.answer(
            "Johansen Formation의 주입률을 알려줘.",
            benchmark_id="WT-011",
        )
    )

    assert prepared["calls"] == 1
    assert ollama.calls == 0
    assert response.answer == (
        "제공된 문서 근거로는 확인할 수 없습니다."
    )

    record = load_run(tmp_path)
    assert record["benchmark_id"] == "WT-011"
    assert record["final_status"] == "completed"
    assert record["final_passed"] is True
    assert len(record["attempts"]) == 1



def test_rewrite_prompt_discards_previous_answer(tmp_path):
    service, _ = make_service(
        tmp_path,
        ["unused"],
    )
    previous = "DO_NOT_COPY_THIS_BAD_ANSWER"
    validation = service.engineering_validator.validate_well_test_answer(
        "Wellbore storage와 radial flow를 구분해줘.",
        (
            "Radial flow에서는 pressure가 unit-slope를 따른다. "
            "[Well Test Analysis, p.219]"
        ),
        retrieved_sources=[
            {
                "document": "Well_Test_Analysis.pdf",
                "page": 219,
            }
        ],
    )
    messages = service._build_rewrite_messages(
        question=(
            "다음 설명이 맞는지 검토해줘. "
            "Radial flow에서 pressure가 unit-slope를 따른다."
        ),
        previous_answer=previous,
        validation=validation,
        original_messages=prepared_payload()["messages"],
    )
    prompt = messages[-1]["content"]
    assert previous not in prompt
    assert "완전히 폐기" in prompt
    assert "해당 설명은 틀렸습니다." in prompt
    assert "pressure derivative가 일정해져 수평 plateau" in prompt



def test_general_rewrite_has_no_false_premise_framing(tmp_path):
    service, _ = make_service(tmp_path, ["unused"])
    validation = service.engineering_validator.validate_well_test_answer(
        "Wellbore storage와 radial flow를 구분해줘.",
        (
            "Radial flow에서는 pressure가 unit-slope를 따른다. "
            "[Well Test Analysis, p.219]"
        ),
        retrieved_sources=[
            {
                "document": "Well_Test_Analysis.pdf",
                "page": 219,
                "excerpt": HIT["text"],
            }
        ],
    )
    messages = service._build_rewrite_messages(
        question=(
            "Wellbore storage와 radial flow를 "
            "pressure derivative로 구분해줘."
        ),
        previous_answer="bad",
        validation=validation,
        original_messages=prepared_payload()["messages"],
    )
    prompt = messages[-1]["content"]
    assert "판정 1문장" not in prompt
    assert "올바르게 수정한 문장" not in prompt
    assert "'판정', '원래 답변'" in prompt


def test_rft_comparison_routes_to_figure_retrieval(tmp_path):
    service, _ = make_service(tmp_path, ["unused"])
    assert service._is_figure_question(
        "Appraisal Well RFT Survey와 RFT Survey "
        "after Significant Production의 압력-심도 "
        "분포를 비교해줘."
    )


def test_strict_refusal_hides_irrelevant_display_sources(tmp_path):
    service, ollama = make_service(
        tmp_path,
        ["제공된 문서 근거로는 확인할 수 없습니다."],
    )
    prepared = install_prepared(
        service,
        prepared_payload(),
    )

    response = asyncio.run(
        service.answer(
            "Johansen Formation의 주입률을 알려줘."
        )
    )

    assert prepared["calls"] == 1
    assert ollama.calls == 1
    assert response.sources == []
    assert response.figures == []

    record = load_run(tmp_path)
    assert len(record["retrieved_sources"]) == 1



def test_rft_rewrite_prompt_contains_numeric_requirements(tmp_path):
    service, _ = make_service(tmp_path, ["unused"])
    validation = ValidationResult(
        passed=False,
        errors=["Appraisal gradient missing."],
        rule_ids=["WT-RFT-002", "WT-RFT-003"],
    )
    messages = service._build_rewrite_messages(
        question=(
            "Appraisal Well RFT Survey와 RFT Survey after "
            "Significant Production을 비교해줘."
        ),
        previous_answer="bad",
        validation=validation,
        original_messages=prepared_payload()["messages"],
    )
    prompt = messages[-1]["content"]
    assert "0.34 psi/ft" in prompt
    assert "0.29" in prompt
    assert "0.37" in prompt
    assert "0.42 psi/ft" in prompt


def test_fracture_rewrite_prompt_requires_partial_handling(tmp_path):
    service, _ = make_service(tmp_path, ["unused"])
    validation = ValidationResult(
        passed=False,
        errors=["Fracture flow missing."],
        rule_ids=["WT-FRACTURE-001"],
    )
    messages = service._build_rewrite_messages(
        question=(
            "Wellbore storage, radial flow, fracture flow의 "
            "특징을 설명해줘."
        ),
        previous_answer="bad",
        validation=validation,
        original_messages=prepared_payload()["messages"],
    )
    prompt = messages[-1]["content"]
    assert "Fracture flow" in prompt
    assert "그 항목만" in prompt
