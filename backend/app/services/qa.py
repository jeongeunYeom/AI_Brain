import re

from app.core.config import Settings
from app.models.schemas import ChatResponse, Source
from app.services.ollama import OllamaClient
from app.services.query_router import QueryType, classify_query
from app.services.vector_store import VectorStore


SYSTEM_PROMPT = """You are a strict evidence-only petroleum engineering RAG agent.
Rules:
1. Use ONLY the retrieved chunks provided in the prompt.
2. Do NOT infer from the document title, general petroleum-engineering knowledge, or model memory.
3. If the retrieved chunks do not explicitly support an answer, reply exactly: "제공된 문서 근거로는 확인할 수 없습니다."
4. Every factual sentence must be supported by a cited source marker like [document, p.page].
5. Do not mention topics that are not present in the retrieved chunks.
6. Answer in Korean unless the source terminology is English.
"""


class QAService:
    def __init__(self, settings: Settings, vector_store: VectorStore, ollama: OllamaClient):
        self.settings = settings
        self.vector_store = vector_store
        self.ollama = ollama

    async def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        query_type = classify_query(question)
        if query_type == QueryType.AGGREGATE_ANALYSIS:
            return self._aggregate_response(question, query_type)

        hits = self._retrieve(question, query_type, top_k or self.settings.top_k)
        if not hits:
            return ChatResponse(
                answer="제공된 문서 근거로는 확인할 수 없습니다.",
                sources=[],
                query_type=query_type.value,
            )

        context_blocks = []
        sources = []
        for hit in hits:
            metadata = hit["metadata"]
            score = float(hit.get("score") or 0.0)
            context_blocks.append(
                f"Source: {metadata.get('document')} p.{metadata.get('page')} chunk {hit['id']} score={score:.3f}\n{hit['text']}"
            )
            preview = self._preview(str(hit["text"]))
            sources.append(Source(
                document=str(metadata.get("document")),
                page=int(metadata["page"]) if metadata.get("page") is not None else None,
                chunk_id=str(hit["id"]),
                score=score,
                vector_score=float(hit.get("vector_score") or 0.0),
                keyword_score=float(hit.get("keyword_score") or 0.0),
                excerpt=str(hit["text"]),
                preview=preview,
            ))

        user_prompt = (
            f"Query type: {query_type.value}\n"
            f"Question:\n{question}\n\n"
            "Retrieved chunks. You must not use any information outside these chunks:\n"
            + "\n\n---\n\n".join(context_blocks)
        )
        answer = await self.ollama.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        return ChatResponse(answer=answer, sources=sources, query_type=query_type.value)

    def _retrieve(self, question: str, query_type: QueryType, top_k: int) -> list[dict]:
        if query_type == QueryType.DOCUMENT_OVERVIEW:
            hits = self.vector_store.hybrid_search(
                question,
                top_k=top_k,
                score_threshold=0.0,
                prefer_metadata={"is_contents": True},
                keyword_weight=0.55,
            )
            if hits:
                contents = [hit for hit in hits if hit.get("metadata", {}).get("is_contents")]
                titles = [hit for hit in hits if hit.get("metadata", {}).get("is_title_page")]
                others = [hit for hit in hits if hit not in contents and hit not in titles]
                ordered = contents + titles + others
                return ordered[:top_k]
        keyword_weight = 0.75 if query_type == QueryType.INDEX_LOOKUP else 0.45
        threshold = max(0.05, self.settings.similarity_threshold if query_type == QueryType.LOCAL_FACT_SEARCH else self.settings.similarity_threshold * 0.6)
        return self.vector_store.hybrid_search(
            question,
            top_k=top_k,
            score_threshold=threshold,
            prefer_metadata=None,
            keyword_weight=keyword_weight,
        )

    def _aggregate_response(self, question: str, query_type: QueryType) -> ChatResponse:
        # Avoid sending huge whole-document context to the LLM. These tasks need a dedicated Python scanner.
        if re.search(r"top\s*\d+|가장 많이|빈도|통계", question.lower()):
            message = "전체 문서 분석 기능이 필요합니다. 이 유형은 긴 context를 LLM에 보내지 않고 Python 기반 전체 chunk 스캔/빈도 분석 함수로 처리해야 합니다."
        else:
            message = "제공된 문서 근거로는 확인할 수 없습니다."
        return ChatResponse(answer=message, sources=[], query_type=query_type.value)

    def _preview(self, text: str, limit: int = 700) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:limit]
