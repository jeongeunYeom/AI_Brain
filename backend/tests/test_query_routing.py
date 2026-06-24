import asyncio
import pytest

pytest.importorskip("pydantic")

from app.core.config import Settings
from app.services.qa import QAService
from app.services.query_router import QueryType, classify_query
from app.services.vector_store import expand_query_terms


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def hybrid_search(self, question, top_k, score_threshold, prefer_metadata=None, keyword_weight=0.45):
        self.calls.append({
            "question": question,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "prefer_metadata": prefer_metadata,
            "keyword_weight": keyword_weight,
        })
        return self.hits


class FakeOllama:
    def __init__(self, answer: str | None = None):
        self.messages = None
        self.answer = answer or "근거에 따르면 이 문서는 Contents에 나열된 색인과 표준을 제공합니다. [S1]"

    async def chat(self, messages):
        self.messages = messages
        return self.answer


def test_classifies_document_overview_and_aggregate():
    assert classify_query("이 문서의 목적이 무엇인지 설명해줘") == QueryType.DOCUMENT_OVERVIEW
    assert classify_query("이 문서에서 가장 많이 참조되는 저자 TOP 20") == QueryType.AGGREGATE_ANALYSIS
    assert classify_query("Bottomhole pressure에 관한 내용을 찾은 모든 문서와 페이지를 표로 정리해줘") == QueryType.AGGREGATE_ANALYSIS
    assert classify_query("압력 관련 그래프 하나를 찾아 x축과 y축을 설명해줘") == QueryType.GRAPH_ANALYSIS
    assert classify_query("SPE Symbols Standard에서 k의 의미") == QueryType.INDEX_LOOKUP


def test_document_overview_prioritizes_contents_chunks():
    hits = [
        {
            "id": "doc:p5:c0",
            "text": "Contents Master Author Index Master Subject Index SPE Symbols Standard SI Metric Conversion Factors",
            "metadata": {"document": "doc.pdf", "page": 5, "is_contents": True},
            "score": 0.7,
            "vector_score": 0.4,
            "keyword_score": 0.8,
        }
    ]
    ollama = FakeOllama()
    service = QAService(Settings(), FakeVectorStore(hits), ollama)

    response = asyncio.run(service.answer("이 문서의 목적이 무엇인지 설명해줘"))

    assert response.query_type == QueryType.DOCUMENT_OVERVIEW.value
    assert response.sources[0].page == 5
    assert "해상 플랫폼" not in response.answer
    assert "[S1]" in response.answer
    assert "Retrieved chunks" in ollama.messages[1]["content"]
    assert "Document: doc.pdf" in ollama.messages[1]["content"]


def test_low_score_results_refuse_answer():
    service = QAService(Settings(similarity_threshold=0.9), FakeVectorStore([]), FakeOllama())

    response = asyncio.run(service.answer("없는 내용 질문"))

    assert response.answer == "제공된 문서 근거로는 확인할 수 없습니다."
    assert response.sources == []


def test_aggregate_analysis_does_not_call_llm():
    ollama = FakeOllama()
    hits = [
        {
            "id": "doc:p1:c0",
            "text": "Most referenced author list",
            "metadata": {"document": "doc.pdf", "page": 1},
            "score": 0.7,
        }
    ]
    service = QAService(Settings(), FakeVectorStore(hits), ollama)

    response = asyncio.run(service.answer("이 문서에서 가장 많이 참조되는 저자 TOP 20을 뽑아줘"))

    assert response.query_type == QueryType.AGGREGATE_ANALYSIS.value
    assert response.sources
    assert "제한" in ollama.messages[1]["content"] or "limited" in ollama.messages[1]["content"].lower()


def test_refusal_answer_drops_sources():
    hits = [
        {
            "id": "doc:p9:c0",
            "text": "Unrelated petroleum production forecast.",
            "metadata": {"document": "forecast.pdf", "page": 9},
            "score": 0.7,
        }
    ]
    ollama = FakeOllama("제공된 문서 근거로는 확인할 수 없습니다.")
    service = QAService(Settings(), FakeVectorStore(hits), ollama)

    response = asyncio.run(service.answer("문서에 나오는 특정 유전의 2027년 생산량을 알려줘."))

    assert response.sources == []


def test_only_cited_source_ids_are_returned():
    hits = [
        {
            "id": "doc:p1:c0",
            "text": "Archie equation text",
            "metadata": {"document": "archie.pdf", "page": 1},
            "score": 0.9,
        },
        {
            "id": "doc:p2:c0",
            "text": "Uncited text",
            "metadata": {"document": "archie.pdf", "page": 2},
            "score": 0.8,
        },
    ]
    ollama = FakeOllama("Archie 방정식은 이 청크에 근거합니다. [S1] 존재하지 않는 출처는 제거됩니다. [S99]")
    service = QAService(Settings(), FakeVectorStore(hits), ollama)

    response = asyncio.run(service.answer("Archie 방정식을 설명해줘"))

    assert "[S1]" in response.answer
    assert "[S99]" not in response.answer
    assert [source.page for source in response.sources] == [1]


def test_technical_term_expansion_keeps_core_terms():
    terms = {term.lower() for term in expand_query_terms("Archie 방정식을 제시하고, 식에 포함된 각 변수와 적용 조건을 설명해줘")}

    assert "archie" in terms
    assert "archie's equation" in terms
    assert "formation resistivity factor" in terms
    assert "설명해줘" not in terms
