import asyncio
import pytest

pytest.importorskip("pydantic")

from app.core.config import Settings
from app.services.qa import QAService
from app.services.query_router import QueryType, classify_query


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
    def __init__(self):
        self.messages = None

    async def chat(self, messages):
        self.messages = messages
        return "근거에 따르면 이 문서는 Contents에 나열된 색인과 표준을 제공합니다. [doc.pdf, p.5]"


def test_classifies_document_overview_and_aggregate():
    assert classify_query("이 문서의 목적이 무엇인지 설명해줘") == QueryType.DOCUMENT_OVERVIEW
    assert classify_query("이 문서에서 가장 많이 참조되는 저자 TOP 20") == QueryType.AGGREGATE_ANALYSIS
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
    assert "Retrieved chunks" in ollama.messages[1]["content"]


def test_low_score_results_refuse_answer():
    service = QAService(Settings(similarity_threshold=0.9), FakeVectorStore([]), FakeOllama())

    response = asyncio.run(service.answer("없는 내용 질문"))

    assert response.answer == "제공된 문서 근거로는 확인할 수 없습니다."
    assert response.sources == []


def test_aggregate_analysis_does_not_call_llm():
    ollama = FakeOllama()
    service = QAService(Settings(), FakeVectorStore([]), ollama)

    response = asyncio.run(service.answer("이 문서에서 가장 많이 참조되는 저자 TOP 20을 뽑아줘"))

    assert response.query_type == QueryType.AGGREGATE_ANALYSIS.value
    assert "전체 문서 분석 기능" in response.answer
    assert ollama.messages is None
