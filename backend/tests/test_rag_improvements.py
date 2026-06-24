import asyncio
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from app.core.config import Settings
from app.models.schemas import Source
from app.services.document_processor import DocumentProcessor
from app.services.qa import QAService
from app.services.query_router import QueryType, classify_query
from app.services.vector_store import VectorStore
from scripts.evaluate_rag import evaluate_questions, has_valid_archie_formula, judge_result, save_reports
from scripts.reprocess_figure_notes import matches_document, reprocess_notes


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def get(self, include=None, limit=None, offset=0):
        rows = self.rows[offset : offset + limit] if limit is not None else self.rows[offset:]
        return {
            "ids": [row["id"] for row in rows],
            "documents": [row["text"] for row in rows],
            "metadatas": [row["metadata"] for row in rows],
        }


def make_store(rows):
    store = VectorStore.__new__(VectorStore)
    store.collection = FakeCollection(rows)
    return store


def test_aggregate_query_classification_variants():
    assert classify_query("전 문서에서 BHP를 전부 찾아 표로 정리해줘") == QueryType.AGGREGATE_ANALYSIS
    assert classify_query("전체에서 찾아 문서별로 정리해줘") == QueryType.AGGREGATE_ANALYSIS


def test_aggregate_keyword_search_batches_dedupes_and_limits_per_document():
    rows = [
        {"id": "a:p1:c0", "text": "Bottomhole pressure definition A", "metadata": {"document_id": "a", "document": "a.pdf", "page": 1}},
        {"id": "a:p1:c1", "text": "BHP duplicate same page", "metadata": {"document_id": "a", "document": "a.pdf", "page": 1}},
        {"id": "a:p2:c0", "text": "bottom-hole pressure page two", "metadata": {"document_id": "a", "document": "a.pdf", "page": 2}},
        {"id": "b:p5:c0", "text": "bottom hole pressure other document", "metadata": {"document_id": "b", "document": "b.pdf", "page": 5}},
    ]
    hits = make_store(rows).aggregate_keyword_search(
        "Bottomhole pressure에 관한 내용을 찾은 모든 문서와 페이지를 표로 정리해줘",
        batch_size=2,
        max_results=10,
        max_per_document=1,
    )

    assert len(hits) == 2
    assert {hit["metadata"]["document"] for hit in hits} == {"a.pdf", "b.pdf"}
    assert len({(hit["metadata"]["document"], hit["metadata"]["page"]) for hit in hits}) == 2


def test_figure_note_filters_logo_or_decorative_image(tmp_path: Path):
    logo = tmp_path / "logo.png"
    Image.new("RGB", (220, 80), "white").save(logo)

    result = DocumentProcessor.classify_image_candidate(logo)

    assert result["should_analyze"] is False
    assert result["image_type"] in {"logo", "decorative"}


def test_graph_question_refuses_without_figure_evidence():
    class FakeVectorStore:
        def hybrid_search(self, *args, **kwargs):
            return [
                {
                    "id": "doc:p1:c0",
                    "text": "Pressure changes with depth in a paragraph.",
                    "metadata": {"document": "doc.pdf", "page": 1},
                    "score": 0.8,
                }
            ]

    class FakeOllama:
        async def chat(self, messages):
            return "should not be called"

    service = QAService(Settings(), FakeVectorStore(), FakeOllama())
    response = asyncio.run(service.answer("압력 관련 그래프 하나를 찾아 x축과 y축을 설명해줘"))

    assert response.sources == []
    assert "그래프 근거" in response.answer


def test_vector_store_init_does_not_load_sentence_transformer(monkeypatch, tmp_path: Path):
    def fail(*args, **kwargs):
        raise AssertionError("SentenceTransformer should be lazy")

    monkeypatch.setattr("app.services.vector_store.SentenceTransformer", fail)

    VectorStore(Settings(data_dir=tmp_path))


def test_evaluation_continues_after_question_failure_and_writes_json(tmp_path: Path):
    class FakeService:
        def __init__(self):
            self.last_debug = {}
            self.calls = 0

        async def answer(self, question):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")

            class Response:
                answer = "제공된 문서 근거로는 확인할 수 없습니다."
                sources = []

            self.last_debug = {
                "query_type": "local_fact_search",
                "search_question": question,
                "retrieved_count": 0,
                "context_sources": [],
            }
            return Response()

    results = asyncio.run(evaluate_questions(FakeService(), ["bad", "good"]))
    json_path, md_path = save_reports(results, tmp_path)

    assert results[0]["error"]
    assert results[1]["error"] is None
    assert json_path.exists()
    assert md_path.exists()


def test_final_source_ids_are_renumbered_in_answer_order():
    class FakeVectorStore:
        def hybrid_search(self, *args, **kwargs):
            return [
                {"id": "a", "text": "ignored", "metadata": {"document": "a.pdf", "page": 1}, "score": 0.9},
                {"id": "b", "text": "usable b", "metadata": {"document": "b.pdf", "page": 2}, "score": 0.8},
                {"id": "c", "text": "ignored", "metadata": {"document": "c.pdf", "page": 3}, "score": 0.7},
                {"id": "d", "text": "ignored", "metadata": {"document": "d.pdf", "page": 4}, "score": 0.6},
                {"id": "e", "text": "usable e", "metadata": {"document": "e.pdf", "page": 5}, "score": 0.5},
                {"id": "f", "text": "ignored", "metadata": {"document": "f.pdf", "page": 6}, "score": 0.4},
                {"id": "g", "text": "ignored", "metadata": {"document": "g.pdf", "page": 7}, "score": 0.3},
                {"id": "h", "text": "ignored", "metadata": {"document": "h.pdf", "page": 8}, "score": 0.2},
                {"id": "i", "text": "usable i", "metadata": {"document": "i.pdf", "page": 9}, "score": 0.1},
            ]

    class FakeOllama:
        async def chat(self, messages):
            return "첫 문장 [S2]. 둘째 문장 [S9]. 셋째 문장 [S5]. 없는 출처 [S99]."

    service = QAService(Settings(), FakeVectorStore(), FakeOllama())
    response = asyncio.run(service.answer("Archie"))

    assert response.answer == "첫 문장 [S1]. 둘째 문장 [S2]. 셋째 문장 [S3]. 없는 출처 ."
    assert [source.document for source in response.sources] == ["b.pdf", "i.pdf", "e.pdf"]
    assert service.last_debug["source_id_mapping"] == {"S2": "S1", "S9": "S2", "S5": "S3"}


def test_aggregate_context_limit_records_debug():
    settings = Settings(aggregate_context_max_chunks=2, aggregate_context_max_characters=30)
    service = QAService(settings, object(), object())
    hits = [
        {"text": "a" * 10},
        {"text": "b" * 10},
        {"text": "c" * 10},
    ]

    limited = service._limit_aggregate_context(hits)

    assert len(limited) == 2
    assert service.last_debug["context_chunks_sent_to_llm"] == 2
    assert service.last_debug["dropped_by_context_limit"] == 1


def test_graph_evidence_requires_structured_note_and_existing_image(tmp_path: Path):
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (400, 300), "white").save(image_path)
    service = QAService(Settings(data_dir=tmp_path), object(), object())

    legacy = {"text": "[Extracted figure notes]\nFigure 1: pressure graph", "metadata": {}}
    structured = {
        "text": (
            "[Figure Note Metadata]\n"
            f"Image Path: {image_path}\n"
            "Image Type: graph\n"
            "Confidence: 0.8\n"
        ),
        "metadata": {},
    }

    assert service._has_figure_evidence(legacy) is False
    assert service._has_figure_evidence(structured) is True


def test_archie_formula_semantic_check_rejects_wrong_scope():
    wrong = "$$S_w = \\frac{R_w}{R_t} \\cdot F^{1/n}$$"
    right = "$$S_w = \\left(\\frac{F R_w}{R_t}\\right)^{1/n}$$"

    assert has_valid_archie_formula(wrong) is False
    assert has_valid_archie_formula(right) is True


def test_evaluator_rejects_invented_mud_weight_symbols():
    checks = judge_result(
        question="Mud weight를 psi/ft 또는 SG로 변환하는 관계식을 문서에서 찾아 설명해줘.",
        answer="$$R_{\\text{psi/ft}} = R_{\\text{ppg}} \\times 0.052$$",
        sources=[{"excerpt": "conversion table"}],
        debug={},
        elapsed_seconds=1,
        error=None,
    )

    assert checks["mud_conversion_no_invented_r_symbols"]["ok"] is False


def test_reprocess_figure_notes_cli_entrypoints(tmp_path: Path):
    env = os.environ.copy()
    env["DATA_DIR"] = str(tmp_path)
    backend_root = Path(__file__).resolve().parents[1]

    commands = [
        [sys.executable, "scripts/reprocess_figure_notes.py", "--help"],
        [sys.executable, "scripts/reprocess_figure_notes.py"],
        [sys.executable, "-m", "scripts.reprocess_figure_notes", "--document", "missing.pdf"],
    ]

    for command in commands:
        result = subprocess.run(
            command,
            cwd=backend_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr

    assert not list(tmp_path.rglob("*.bak.md"))


def test_reprocess_figure_notes_skip_reasons_and_document_normalization(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    settings.figure_notes_dir.mkdir(parents=True, exist_ok=True)
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    note = settings.figure_notes_dir / "Heriot_Watt_University_Well_Test_Analysis.md"
    note.write_text("legacy note", encoding="utf-8")
    (settings.figure_notes_dir / "other.md").write_text("legacy note", encoding="utf-8")

    assert matches_document(note, "", "heriot-watt university well test analysis")
    counts = reprocess_notes(
        settings=settings,
        document="Heriot-Watt University - Well Test Analysis",
        apply=False,
        update_chroma=False,
    )

    assert counts["scanned"] == 2
    assert counts["document_matches"] == 1
    assert counts["skipped_missing_image"] == 1
    assert counts["skipped_document_mismatch"] == 1
    assert counts["processed"] == 0
    assert counts["failed"] == 0


def test_evaluator_rejects_rt_as_mud_weight_pressure_gradient():
    checks = judge_result(
        question="Mud weight를 psi/ft 또는 SG로 변환하는 관계식을 문서에서 찾아 설명해줘.",
        answer="$$R_t = MW \\times 0.052$$",
        sources=[{"excerpt": "conversion table"}],
        debug={},
        elapsed_seconds=1,
        error=None,
    )

    assert checks["mud_conversion_no_invented_r_symbols"]["ok"] is False


def test_mud_window_fallback_ignores_contents_and_uses_two_sources():
    service = QAService(Settings(), object(), object())
    sources = {
        "S1": Source(
            document="toc.pdf",
            page=160,
            chunk_id="toc",
            excerpt="Table of Contents kick lost circulation pore pressure fracture pressure",
        ),
        "S2": Source(
            document="drilling.pdf",
            page=10,
            chunk_id="kick",
            excerpt="If hydrostatic pressure is below pore pressure, formation fluid can enter the wellbore as an influx or kick.",
        ),
        "S3": Source(
            document="drilling.pdf",
            page=11,
            chunk_id="loss",
            excerpt="If pressure exceeds the fracture pressure or fracture gradient, formation breakdown and lost circulation can occur.",
        ),
    }

    answer, used = service._mud_window_fallback_answer(sources)

    assert [source.chunk_id for source in used] == ["kick", "loss"]
    assert "[S1]" in answer and "[S2]" in answer


def test_mud_window_fallback_refuses_contents_only():
    service = QAService(Settings(), object(), object())
    answer, used = service._mud_window_fallback_answer(
        {
            "S1": Source(
                document="toc.pdf",
                page=160,
                chunk_id="toc",
                excerpt="Table of Contents kick lost circulation pore pressure fracture pressure",
            )
        }
    )

    assert used == []
    assert "확인할 수 없습니다" in answer
