from types import SimpleNamespace

from app.services.qa import QAService
from app.services.query_router import QueryType


class FakeCollection:
    def __init__(self, rows=None):
        self.rows = rows or []

    def get(self, include=None):
        return {
            "ids": [row[0] for row in self.rows],
            "documents": [row[1] for row in self.rows],
            "metadatas": [row[2] for row in self.rows],
        }


class FakeVectorStore:
    def __init__(self, hits, collection_rows=None):
        self.hits = hits
        self.calls = []
        self.collection = FakeCollection(collection_rows)

    def hybrid_search(self, question, top_k, score_threshold, prefer_metadata=None, keyword_weight=0.45):
        self.calls.append({
            "question": question,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "prefer_metadata": prefer_metadata,
            "keyword_weight": keyword_weight,
        })
        return [dict(hit) for hit in self.hits[:top_k]]


def make_service(hits, collection_rows=None):
    settings = SimpleNamespace(top_k=10, similarity_threshold=0.35)
    return QAService(settings, FakeVectorStore(hits, collection_rows), None)


def hit(chunk_id, text, score, document="Well_Test.pdf", page=1):
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {"document": document, "page": page},
        "score": score,
        "vector_score": score,
        "keyword_score": score,
    }


def test_log_log_figure_note_beats_learning_outcomes():
    service = make_service([
        hit("learning", "LEARNING OUTCOMES Describe the use of the log-log plot", 0.92),
        hit(
            "figure",
            "[Extracted figure notes] [Figure Note Metadata] title: Log-Log Plot "
            "x_axis: Equivalent Time y_axis: Delta P series_descriptions: cyan rises "
            "trend_summary: upper rises and lower declines",
            0.38,
        ),
    ])
    results = service._retrieve(
        "Log-Log Plot Equivalent Time Delta P",
        QueryType.LOCAL_FACT_SEARCH,
        2,
        original_question="Log-Log Plot에서 Equivalent Time과 Delta P의 두 계열은 어떤 추세를 보이는가?",
    )
    assert results[0]["id"] == "figure"
    assert results[0]["is_figure_note"] is True
    assert "Figure Note Metadata" in service.vector_store.calls[0]["question"]


def test_rft_figure_note_beats_general_chapter_text():
    service = make_service([
        hit("general", "RFT pressure gradients and supercharged points are discussed.", 0.88),
        hit(
            "figure",
            "[Figure Note Metadata] title: Appraisal Well RFT Survey "
            "analysis: pressure-depth graph reference_lines: 0.34 psi/ft "
            "engineering_meaning: supercharged tests are offset points",
            0.42,
        ),
    ])
    results = service._retrieve(
        "Appraisal Well RFT Survey supercharged",
        QueryType.LOCAL_FACT_SEARCH,
        2,
        original_question="Appraisal Well RFT Survey에서 표시된 압력 기울기와 supercharged test의 의미를 설명해줘.",
    )
    assert results[0]["id"] == "figure"


def test_strong_literal_anchor_locks_retrieval_to_matching_document_and_neighbors():
    wrong = hit(
        "otherdoc:p68:c2",
        "Pressure distribution and RFT pressure gradient in a reservoir.",
        0.96,
        document="Reservoir_Engineering.pdf",
        page=68,
    )
    rows = [
        (
            "targetdoc:p439:c0",
            "Figure 2 shows an offshore appraisal well. An oil gradient of 0.34 psi/ft "
            "is indicated and two supercharged points are excluded.",
            {"document": "Well_Test_Analysis.pdf", "page": 439},
        ),
        (
            "targetdoc:p440:c1",
            "[Extracted figure notes] [Figure Note Metadata] "
            "title: Appraisal Well RFT Survey analysis: pressure-depth graph "
            "reference_lines: 0.34 psi/ft engineering_meaning: supercharged tests are offset points",
            {"document": "Well_Test_Analysis.pdf", "page": 440},
        ),
        (
            "otherdoc:p68:c2",
            "Pressure distribution and RFT pressure gradient in a reservoir.",
            {"document": "Reservoir_Engineering.pdf", "page": 68},
        ),
    ]
    service = make_service([wrong], collection_rows=rows)
    results = service._retrieve(
        "Appraisal Well RFT Survey supercharged test",
        QueryType.LOCAL_FACT_SEARCH,
        5,
        original_question="Appraisal Well RFT Survey에서 표시된 압력 기울기와 supercharged test의 의미를 설명해줘.",
    )

    assert results[0]["id"] == "targetdoc:p440:c1"
    assert all(str(result["id"]).startswith("targetdoc:") for result in results)
    assert any(result["id"] == "targetdoc:p439:c0" for result in results)
    assert results[0]["strong_phrase_matches"] >= 1
    assert results[0]["figure_rank_score"] > results[-1]["figure_rank_score"]


def test_non_figure_question_keeps_normal_retrieval_path():
    service = make_service([
        hit("first", "Formation pressure definition", 0.8),
        hit("second", "Other text", 0.5),
    ])
    results = service._retrieve(
        "formation pressure definition",
        QueryType.LOCAL_FACT_SEARCH,
        1,
        original_question="formation pressure definition",
    )
    assert results[0]["id"] == "first"
    assert service.vector_store.calls[0]["top_k"] == 1
    assert service.vector_store.calls[0]["keyword_weight"] == 0.45


def test_numeric_gradient_preceding_page_is_promoted_for_rft_question():
    rows = [
        (
            "targetdoc:p439:c0",
            "Figure 2 shows an offshore appraisal well in the Gulf of Campeche. "
            "An oil gradient of 0.34 psi/ft is clearly indicated and two supercharged "
            "points were eliminated from consideration.",
            {"document": "Well_Test_Analysis.pdf", "page": 439},
        ),
        (
            "targetdoc:p440:c1",
            "[Figure Note Metadata] title: Appraisal Well RFT Survey "
            "analysis: pressure-depth graph engineering_meaning: pressure gradient analysis",
            {"document": "Well_Test_Analysis.pdf", "page": 440},
        ),
        (
            "targetdoc:p441:c2",
            "[Figure Note Metadata] title: Interpreted RFT Data "
            "series_descriptions: supercharged points and mud gradient",
            {"document": "Well_Test_Analysis.pdf", "page": 441},
        ),
    ]
    service = make_service([], collection_rows=rows)
    results = service._retrieve(
        "Appraisal Well RFT Survey supercharged test",
        QueryType.LOCAL_FACT_SEARCH,
        5,
        original_question=(
            "Appraisal Well RFT Survey에서 표시된 압력 기울기와 "
            "supercharged test의 의미를 설명해줘."
        ),
    )

    ids = [item["id"] for item in results]
    assert "targetdoc:p439:c0" in ids[:2]
    support = next(item for item in results if item["id"] == "targetdoc:p439:c0")
    assert support["has_numeric_gradient"] is True
    assert support["anchor_preceding"] == 1.0
    assert support["mentions_supercharged"] is True


def test_partial_evidence_policy_does_not_require_full_refusal():
    from app.services.qa import SYSTEM_PROMPT

    assert "partial evidence" in SYSTEM_PROMPT.lower()
    assert "Do not refuse the entire question" in SYSTEM_PROMPT


def test_type_curve_shape_question_promotes_unit_slope_and_derivative_plateau():
    rows = [
        (
            "targetdoc:p219:c0",
            "3.3.3 tD/CD Type Curve Including the Derivative. At very early time "
            "the pressure and logarithmic derivative overlay on the unit slope diagonal. "
            "The beginning of the middle time region is indicated by the derivative becoming constant.",
            {"document": "Well_Test_Analysis.pdf", "page": 219},
        ),
        (
            "targetdoc:p219:c1",
            "The middle time region is associated with the terminology derivative plateau. Figure 22.",
            {"document": "Well_Test_Analysis.pdf", "page": 219},
        ),
        (
            "targetdoc:p286:c1",
            "Early time analysis may be carried out by type curve matching.",
            {"document": "Well_Test_Analysis.pdf", "page": 286},
        ),
    ]
    service = make_service(
        [
            hit(
                "targetdoc:p286:c1",
                "Early time analysis may be carried out by type curve matching.",
                0.95,
                page=286,
            )
        ],
        collection_rows=rows,
    )

    results = service._retrieve(
        "Type curve matching early-time wellbore storage middle-time radial flow",
        QueryType.LOCAL_FACT_SEARCH,
        5,
        original_question=(
            "Type curve matching 그림에서 early-time wellbore storage 구간과 "
            "middle-time radial flow 구간은 곡선 형태로 어떻게 구분되는가?"
        ),
    )

    assert results[0]["id"] in {"targetdoc:p219:c0", "targetdoc:p219:c1"}
    assert any(item["id"] == "targetdoc:p219:c0" for item in results[:2])
    assert any(float(item.get("special_type_curve") or 0) > 0 for item in results[:2])
    expanded = service.vector_store.calls[0]["question"].lower()
    assert "unit slope diagonal" in expanded
    assert "derivative plateau" in expanded


def test_rft_before_after_comparison_promotes_both_figure_notes():
    rows = [
        (
            "targetdoc:p440:c1",
            "[Extracted figure notes] [Figure Note Metadata] "
            "title: Appraisal Well RFT Survey image_path: before.jpeg "
            "analysis: appraisal pressure gradient before significant production. "
            "Figure 3 RFT survey after significant production",
            {"document": "Well_Test_Analysis.pdf", "page": 440},
        ),
        (
            "targetdoc:p440:c2",
            "[Extracted figure notes] [Figure Note Metadata] "
            "title: RFT Survey after Significant Production image_path: after.jpeg "
            "trend_summary: three gradients separated by permeability barriers.",
            {"document": "Well_Test_Analysis.pdf", "page": 440},
        ),
        (
            "targetdoc:p441:c2",
            "[Figure Note Metadata] title: Interpreted RFT Data",
            {"document": "Well_Test_Analysis.pdf", "page": 441},
        ),
    ]
    service = make_service([], collection_rows=rows)

    results = service._retrieve(
        "RFT Survey before significant production after significant production",
        QueryType.LOCAL_FACT_SEARCH,
        5,
        original_question=(
            "RFT Survey before significant production과 after significant production을 "
            "비교하면 pressure trend와 barrier 해석이 어떻게 달라지는가?"
        ),
    )

    ids = [item["id"] for item in results]
    assert "targetdoc:p440:c1" in ids[:3]
    assert "targetdoc:p440:c2" in ids[:3]
    before = next(item for item in results if item["id"] == "targetdoc:p440:c1")
    after = next(item for item in results if item["id"] == "targetdoc:p440:c2")
    assert before["comparison_before"] == 1.0
    assert before["comparison_after"] == 1.0
    assert after["comparison_after"] == 1.0
