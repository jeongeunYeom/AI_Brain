import re
from pathlib import Path

from app.core.config import Settings
from app.models.schemas import ChatResponse, Source
from app.services.ollama import OllamaClient
from app.services.query_router import QueryType, classify_query
from app.services.vector_store import VectorStore



SHA256_PREFIX = re.compile(r"^[0-9a-fA-F]{64}_")
SOURCE_REF = re.compile(r"\[S(\d+)\]")
REFUSAL_MARKERS = (
    "제공된 문서 근거로는 확인할 수 없습니다",
    "제공된 문서 근거로는 확인할 수 없어 답변할 수 없습니다",
)
GRAPH_REFUSAL = "제공된 문서에서 분석 가능한 그래프 또는 그림 근거를 찾지 못했습니다."


def clean_document_name(value: object) -> str:
    """Remove a stored SHA-256 prefix from a source filename for display/citation."""
    filename = Path(str(value or "Unknown document")).name
    return SHA256_PREFIX.sub("", filename)


SYSTEM_PROMPT = """You are a strict evidence-only petroleum engineering RAG agent.
Rules:
1. Use ONLY the retrieved chunks provided in the prompt.
2. Do NOT infer from the document title, general petroleum-engineering knowledge, or model memory.
3. If the retrieved chunks do not explicitly support an answer, reply exactly: "제공된 문서 근거로는 확인할 수 없습니다."
4. Cite sources only with provided source IDs such as [S1], [S2]. Do not cite document names, page numbers, or invented markers.
5. Every factual sentence must be supported by one or more valid [S#] markers.
6. Do not mention topics that are not present in the retrieved chunks.
7. When the user asks for formulas, use only formulas present in the chunks. Do not reconstruct or guess formulas.
8. Write standalone formulas as separate Markdown LaTeX blocks using $$ ... $$ and inline variables as $R_t$ or $S_w$.
9. Explain units only when units are explicitly present in the chunks. If not present, say "문서에 단위가 명시되지 않음".
10. Never concatenate multiple equations on one line.
11. Answer in Korean unless the source terminology is English.
"""


class QAService:
    def __init__(self, settings: Settings, vector_store: VectorStore, ollama: OllamaClient):
        self.settings = settings
        self.vector_store = vector_store
        self.ollama = ollama

    async def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        query_type = classify_query(question)

        search_question = self._build_search_question(question)
        hits = self._retrieve(
            search_question,
            query_type,
            top_k or self.settings.top_k,
        )

        if query_type == QueryType.GRAPH_ANALYSIS:
            hits = [hit for hit in hits if self._has_figure_evidence(hit)]
            if not hits:
                return ChatResponse(
                    answer=GRAPH_REFUSAL,
                    sources=[],
                    query_type=query_type.value,
                )

        if not hits:
            return ChatResponse(
                answer=self._refusal_for_question(question),
                sources=[],
                query_type=query_type.value,
            )

        context_blocks = []
        sources_by_id: dict[str, Source] = {}
        for source_index, hit in enumerate(hits, start=1):
            source_id = f"S{source_index}"
            metadata = hit["metadata"]
            score = float(hit.get("score") or 0.0)
            document_name = clean_document_name(metadata.get("document"))
            page_number = metadata.get("page")

            context_blocks.append(
                f"[{source_id}]\n"
                f"Document: {document_name}\n"
                f"Page: {page_number}\n"
                f"Chunk ID: {hit['id']}\n"
                f"Score: {score:.3f}\n"
                f"Content:\n{hit['text']}"
            )

            preview = self._preview(str(hit["text"]))
            sources_by_id[source_id] = Source(
                document=document_name,
                page=(
                    int(page_number)
                    if page_number is not None
                    else None
                ),
                chunk_id=str(hit["id"]),
                score=score,
                vector_score=float(
                    hit.get("vector_score") or 0.0
                ),
                keyword_score=float(
                    hit.get("keyword_score") or 0.0
                ),
                excerpt=str(hit["text"]),
                preview=preview,
            )

        user_prompt = (
            f"Query type: {query_type.value}\n"
            f"Question:\n{question}\n\n"
            "Retrieved chunks. Use only these source IDs. Cite only [S1], [S2], ... from this list.\n"
            "If no listed source explicitly supports the answer, refuse instead of guessing.\n"
            + self._query_type_instruction(query_type)
            + "\n\n---\n\n".join(context_blocks)
        )
        answer = await self.ollama.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        answer, sources = self._filter_answer_sources(answer, sources_by_id, question)
        return ChatResponse(answer=answer, sources=sources, query_type=query_type.value)


    def _build_search_question(self, question: str) -> str:
        """Build a compact retrieval query while preserving the full user question for generation."""
        normalized = re.sub(r"[’`]", "'", question)

        # Mixed Korean/English technical questions are searched primarily by
        # their English engineering terms. This prevents instruction words
        # such as "설명해줘", "변수", and "적용 조건" from diluting an exact
        # keyword like Archie, Kick, BHP, or Wyllie.
        english_terms = re.findall(
            r"[A-Za-z][A-Za-z0-9_.-]*(?:'s)?",
            normalized,
        )

        ignored_terms = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }

        compact_terms: list[str] = []
        seen: set[str] = set()

        for term in english_terms:
            term = re.sub(r"'s$", "", term, flags=re.IGNORECASE)
            key = term.lower()

            if key in ignored_terms or key in seen:
                continue

            seen.add(key)
            compact_terms.append(term)

        if compact_terms:
            return " ".join(compact_terms)

        # Korean-only questions keep their technical content but remove common
        # request phrasing that does not help document retrieval.
        cleaned = question
        request_patterns = [
            r"업로드한\s*문서만\s*근거로",
            r"문서\s*근거로",
            r"문서명과\s*페이지\s*번호를\s*표시해줘",
            r"사용한\s*문서명과\s*페이지\s*번호를\s*표시해줘",
            r"근거가\s*없으면\s*추측하지\s*말고\s*없다고\s*말해줘",
            r"설명해\s*줘",
            r"설명해줘",
            r"제시하고",
            r"제시해\s*줘",
            r"제시해줘",
            r"각\s*변수",
            r"적용\s*조건",
            r"문서에서\s*찾아",
            r"문서에서\s*찾아줘",
            r"정리해\s*줘",
            r"정리해줘",
            r"알려\s*줘",
            r"알려줘",
        ]

        for pattern in request_patterns:
            cleaned = re.sub(
                pattern,
                " ",
                cleaned,
                flags=re.IGNORECASE,
            )

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or question

    def _retrieve(self, question: str, query_type: QueryType, top_k: int) -> list[dict]:
        if query_type == QueryType.GRAPH_ANALYSIS:
            return self.vector_store.hybrid_search(
                f"{question} figure graph plot chart axis x-axis y-axis trend caption extracted figure notes",
                top_k=max(top_k * 2, 12),
                score_threshold=0.05,
                prefer_metadata=None,
                keyword_weight=0.75,
            )

        if query_type == QueryType.AGGREGATE_ANALYSIS:
            hits = self.vector_store.hybrid_search(
                question,
                top_k=max(top_k * 6, 30),
                score_threshold=0.02,
                prefer_metadata=None,
                keyword_weight=0.75,
            )
            return self._dedupe_document_pages(hits, max_total=max(top_k * 4, 20))

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
        hits = self.vector_store.hybrid_search(
            question,
            top_k=top_k,
            score_threshold=threshold,
            prefer_metadata=None,
            keyword_weight=keyword_weight,
        )
        if hasattr(self.vector_store, "expand_with_neighbors"):
            return self.vector_store.expand_with_neighbors(
                hits,
                max_total=min(max(top_k + 6, top_k), 18),
            )
        return hits

    def _preview(self, text: str, limit: int = 700) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:limit]

    def _query_type_instruction(self, query_type: QueryType) -> str:
        if query_type == QueryType.AGGREGATE_ANALYSIS:
            return (
                "For aggregate questions, summarize only the searched sources. "
                "State that the table is limited to retrieved evidence. "
                "Use table columns: 문서명, 페이지, 정의 또는 관련 내용, 문서 간 차이, 출처 ID.\n"
            )
        if query_type == QueryType.GRAPH_ANALYSIS:
            return (
                "For graph or figure questions, answer only from figure notes, image analysis, captions, "
                "or chunks explicitly discussing a figure/graph. Do not infer axes, units, or trends from text-only evidence.\n"
            )
        return ""

    def _filter_answer_sources(
        self,
        answer: str,
        sources_by_id: dict[str, Source],
        question: str,
    ) -> tuple[str, list[Source]]:
        if self._is_refusal(answer):
            return answer, []

        used_ids: list[str] = []

        def replace_invalid(match: re.Match[str]) -> str:
            source_id = f"S{match.group(1)}"
            if source_id not in sources_by_id:
                return ""
            if source_id not in used_ids:
                used_ids.append(source_id)
            return f"[{source_id}]"

        cleaned_answer = SOURCE_REF.sub(replace_invalid, answer)

        if not used_ids:
            return self._refusal_for_question(question), []

        return cleaned_answer, [sources_by_id[source_id] for source_id in used_ids]

    def _is_refusal(self, answer: str) -> bool:
        normalized = re.sub(r"\s+", " ", answer).strip()
        return any(marker in normalized for marker in REFUSAL_MARKERS) or GRAPH_REFUSAL in normalized

    def _refusal_for_question(self, question: str) -> str:
        if re.search(r"2027년\s*생산량|2027.*production", question, re.IGNORECASE):
            return "제공된 문서 근거로는 해당 유전의 2027년 생산량을 확인할 수 없습니다."
        return "제공된 문서 근거로는 확인할 수 없습니다."

    def _has_figure_evidence(self, hit: dict) -> bool:
        text = str(hit.get("text") or "").lower()
        return any(
            marker in text
            for marker in [
                "extracted figure notes",
                "image analysis",
                "figure ",
                "fig.",
                "graph",
                "plot",
                "caption",
                "x-axis",
                "y-axis",
                "x축",
                "y축",
            ]
        )

    def _dedupe_document_pages(self, hits: list[dict], max_total: int) -> list[dict]:
        deduped: list[dict] = []
        seen: set[tuple[str, object]] = set()

        for hit in hits:
            metadata = hit.get("metadata") or {}
            key = (
                str(metadata.get("document_id") or metadata.get("document") or ""),
                metadata.get("page"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
            if len(deduped) >= max_total:
                break

        return deduped
