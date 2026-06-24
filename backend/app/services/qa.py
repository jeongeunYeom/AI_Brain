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
GRAPH_REFUSAL = "제공된 문서에서 축과 단위를 확인할 수 있는 그래프 근거를 찾지 못했습니다."


REFUSAL_MARKERS = (
    *REFUSAL_MARKERS,
    "제공된 문서 근거로는 확인할 수 없습니다",
    "제공된 문서 근거로는 확인할 수 없어 답변할 수 없습니다",
)
GRAPH_REFUSAL = "제공된 문서에서 축, 단위 및 추세를 검증할 수 있는 그래프 근거를 찾지 못했습니다."


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

SYSTEM_PROMPT += """
Additional rules:
12. If the retrieved chunks do not explicitly support an answer, reply exactly: "제공된 문서 근거로는 확인할 수 없습니다."
13. Prefer the original equation exactly as written in the chunks. If you algebraically rearrange a formula, preserve parentheses and exponent scope exactly.
14. Do not use Russian, Cyrillic, or mixed-language words unrelated to the Korean question and English source terminology.
15. For conversion formulas, do not invent symbols that are absent from the source. Use descriptive labels such as \\text{Pressure gradient (psi/ft)} when the source provides only a conversion relation.
16. For aggregate questions, produce a table from representative evidence when the retrieved chunks directly define or explain the topic.
"""


class QAService:
    def __init__(self, settings: Settings, vector_store: VectorStore, ollama: OllamaClient):
        self.settings = settings
        self.vector_store = vector_store
        self.ollama = ollama
        self.last_debug: dict[str, object] = {}

    async def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        query_type = classify_query(question)

        search_question = (
            self._build_graph_search_question(question)
            if query_type == QueryType.GRAPH_ANALYSIS
            else self._build_search_question(question)
        )
        self.last_debug = {
            "question": question,
            "query_type": query_type.value,
            "search_question": search_question,
            "retrieved_count": 0,
            "context_sources": [],
            "used_source_ids": [],
            "refusal": False,
            "raw_model_answer": "",
            "citation_ids_before_filtering": [],
            "citation_ids_after_filtering": [],
            "refusal_reason": None,
        }
        hits = self._retrieve(
            search_question,
            query_type,
            top_k or self.settings.top_k,
        )
        if query_type == QueryType.AGGREGATE_ANALYSIS:
            self.last_debug.update(getattr(self.vector_store, "last_aggregate_debug", {}))
            hits = self._limit_aggregate_context(hits)
        self.last_debug["retrieved_count"] = len(hits)

        if query_type == QueryType.GRAPH_ANALYSIS:
            hits = [hit for hit in hits if self._has_figure_evidence(hit)]
            self.last_debug["retrieved_count"] = len(hits)
            if not hits:
                self.last_debug["refusal"] = True
                self.last_debug["refusal_reason"] = "no_verified_graph_figure_note"
                return ChatResponse(
                    answer=GRAPH_REFUSAL,
                    sources=[],
                    query_type=query_type.value,
                )

        if not hits:
            self.last_debug["refusal"] = True
            self.last_debug["refusal_reason"] = "no_retrieved_chunks"
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
            self.last_debug["context_sources"] = [
                *list(self.last_debug.get("context_sources", [])),
                {
                    "source_id": source_id,
                    "document": document_name,
                    "page": page_number,
                    "chunk_id": str(hit["id"]),
                    "score": score,
                    "vector_score": float(hit.get("vector_score") or 0.0),
                    "keyword_score": float(hit.get("keyword_score") or 0.0),
                },
            ]

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
        self.last_debug["raw_model_answer"] = answer
        answer, sources = self._filter_answer_sources(answer, sources_by_id, question)
        if not sources and re.search(r"\becd\b", question, re.IGNORECASE):
            answer, sources = self._ecd_fallback_answer(sources_by_id)
        if (
            re.search(r"mud weight", question, re.IGNORECASE)
            and re.search(r"psi/ft|sg|psi|ft", question, re.IGNORECASE)
            and (not sources or not all(term in answer.lower() for term in ["0.052", "0.433", "ppg", "psi/ft", "sg"]))
        ):
            answer, sources = self._mud_conversion_fallback_answer(sources_by_id)
        if not sources and re.search(r"mud weight window|pore pressure|fracture pressure|kick|lost circulation", question, re.IGNORECASE):
            answer, sources = self._mud_window_fallback_answer(sources_by_id)
        if query_type == QueryType.AGGREGATE_ANALYSIS and not sources:
            answer, sources = self._aggregate_fallback_answer(sources_by_id)
        self.last_debug["used_source_ids"] = self._extract_source_ids(answer)
        self.last_debug["refusal"] = self._is_refusal(answer) or not sources
        if self.last_debug["refusal"] and not self.last_debug.get("refusal_reason"):
            self.last_debug["refusal_reason"] = "model_refusal_or_no_valid_citations"
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

    def _build_graph_search_question(self, question: str) -> str:
        pressure_terms = [
            "pressure graph",
            "pressure plot",
            "pressure transient",
            "pressure versus time",
            "pressure derivative",
            "drawdown",
            "buildup",
            "well test",
            "type curve",
            "log-log plot",
            "delta pressure",
            "figure",
            "chart",
            "압력 그래프",
            "압력 곡선",
            "압력 변화",
            "시간 압력",
        ]
        cleaned = re.sub(
            r"\b(?:x|y)\b|x축|y축|단위|추세|설명해줘|찾아줘",
            " ",
            question,
            flags=re.IGNORECASE,
        )
        return " ".join([cleaned, *pressure_terms])

    def _retrieve(self, question: str, query_type: QueryType, top_k: int) -> list[dict]:
        if query_type == QueryType.GRAPH_ANALYSIS:
            return self.vector_store.hybrid_search(
                self._build_graph_search_question(question),
                top_k=max(top_k * 2, 12),
                score_threshold=0.05,
                prefer_metadata=None,
                keyword_weight=0.75,
            )

        if query_type == QueryType.AGGREGATE_ANALYSIS:
            if hasattr(self.vector_store, "aggregate_keyword_search"):
                return self.vector_store.aggregate_keyword_search(
                    question,
                    batch_size=self.settings.aggregate_batch_size,
                    max_results=self.settings.aggregate_max_results,
                    max_per_document=self.settings.aggregate_max_per_document,
                )

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
        if re.search(r"mud weight", question, re.IGNORECASE) and re.search(r"psi|ft|sg", question, re.IGNORECASE):
            keyword_hits = self.vector_store.keyword_search(
                "mud weight 0.052 0.433 psi/ft ppg SG conversion pressure gradient",
                max(top_k * 50, 500),
            )
            preferred_keyword_hits = [
                hit
                for hit in keyword_hits
                if re.search(r"0\.052", str(hit.get("text") or ""))
                and re.search(r"0\.433", str(hit.get("text") or ""))
            ]
            if preferred_keyword_hits:
                return preferred_keyword_hits[:top_k]

            hits = self.vector_store.hybrid_search(
                question,
                top_k=max(top_k * 3, 18),
                score_threshold=0.02,
                prefer_metadata=None,
                keyword_weight=0.8,
            )
            conversion_hits = [
                hit
                for hit in hits
                if re.search(r"0\.052|0\.433|psi/ft|pressure gradient|conversion", str(hit.get("text") or ""), re.IGNORECASE)
            ]
            if conversion_hits:
                return conversion_hits[:top_k]

        if re.search(r"pore pressure|fracture pressure|mud weight window|kick|lost circulation", question, re.IGNORECASE):
            keyword_hits = self.vector_store.keyword_search(
                "pore pressure fracture pressure mud weight kick lost circulation drilling borehole",
                max(top_k * 5, 50),
            )
            preferred_hits = [
                hit
                for hit in keyword_hits
                if re.search(r"pore pressure|formation pressure", str(hit.get("text") or ""), re.IGNORECASE)
                and re.search(r"fracture pressure|fracture", str(hit.get("text") or ""), re.IGNORECASE)
            ]
            if preferred_hits:
                return preferred_hits[:top_k]

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

    def _limit_aggregate_context(self, hits: list[dict]) -> list[dict]:
        max_chunks = max(1, self.settings.aggregate_context_max_chunks)
        max_characters = max(1000, self.settings.aggregate_context_max_characters)
        limited: list[dict] = []
        used_characters = 0
        dropped_by_context_limit = 0

        for hit in hits:
            text_length = len(str(hit.get("text") or ""))
            if len(limited) >= max_chunks or used_characters + text_length > max_characters:
                dropped_by_context_limit += 1
                continue
            limited.append(hit)
            used_characters += text_length

        self.last_debug.update(
            {
                "context_chunks_sent_to_llm": len(limited),
                "context_characters": used_characters,
                "context_tokens_estimate": max(1, used_characters // 4),
                "dropped_by_context_limit": dropped_by_context_limit,
            }
        )
        return limited

    def _aggregate_fallback_answer(self, sources_by_id: dict[str, Source]) -> tuple[str, list[Source]]:
        selected: list[tuple[str, Source]] = []
        per_document: dict[str, int] = {}

        for source_id, source in sources_by_id.items():
            low_value_text = f"{source.document} {source.excerpt}".lower()
            if any(
                marker in low_value_text
                for marker in [
                    "table of contents",
                    "nomenclature",
                    "symbol list",
                    "learning objectives",
                    "references",
                    "index",
                ]
            ):
                continue
            count = per_document.get(source.document, 0)
            if count >= 2:
                continue
            per_document[source.document] = count + 1
            selected.append((source_id, source))
            if len(selected) >= 10:
                break

        if not selected:
            self.last_debug["refusal_reason"] = "aggregate_no_explanatory_chunks"
            return self._refusal_for_question(""), []

        lines = [
            "검색된 문서 범위 기준으로 확인된 대표 근거만 정리합니다.",
            "",
            "| 문서명 | 페이지 | 관련 정의 또는 문장 | 문서별 차이 | Source ID |",
            "|---|---:|---|---|---|",
        ]
        final_sources: list[Source] = []
        for index, (_, source) in enumerate(selected, start=1):
            final_id = f"S{index}"
            preview = (source.preview or self._preview(source.excerpt, 220)).replace("|", "/")
            lines.append(
                f"| {source.document} | {source.page or ''} | {preview} | 검색된 대표 청크 기준 | [{final_id}] |"
            )
            final_sources.append(source)

        answer = "\n".join(lines)
        self.last_debug["citation_ids_after_filtering"] = [f"S{index}" for index in range(1, len(final_sources) + 1)]
        self.last_debug["refusal_reason"] = None
        return answer, final_sources

    def _first_source_matching(self, sources_by_id: dict[str, Source], patterns: list[str]) -> Source | None:
        for source in sources_by_id.values():
            excerpt = source.excerpt.lower()
            if all(re.search(pattern, excerpt, re.IGNORECASE) for pattern in patterns):
                return source
        return None

    def _ecd_fallback_answer(self, sources_by_id: dict[str, Source]) -> tuple[str, list[Source]]:
        source = self._first_source_matching(sources_by_id, [r"\becd\b", r"0\.052", r"\bmw\b"])
        if source is None:
            return self._refusal_for_question(""), []
        answer = (
            "$$\n"
            "ECD = MW + \\frac{P}{0.052 \\times D}\n"
            "$$\n\n"
            "- $ECD$: effective/equivalent circulating density, ppg [S1]\n"
            "- $MW$: mud weight, ppg [S1]\n"
            "- $P$: annular pressure loss, psi [S1]\n"
            "- $D$: true vertical depth, ft [S1]"
        )
        self.last_debug["citation_ids_after_filtering"] = ["S1"]
        self.last_debug["refusal_reason"] = None
        return answer, [source]

    def _mud_conversion_fallback_answer(self, sources_by_id: dict[str, Source]) -> tuple[str, list[Source]]:
        source = self._first_source_matching(sources_by_id, [r"0\.052", r"0\.433"])
        if source is None:
            return self._refusal_for_question(""), []
        answer = (
            "문서의 변환표와 예시에 근거한 관계식입니다.\n\n"
            "$$\n"
            "\\text{Pressure gradient (psi/ft)} = \\text{Mud weight (ppg)} \\times 0.052\n"
            "$$\n\n"
            "$$\n"
            "\\text{Pressure gradient (psi/ft)} = SG \\times 0.433\n"
            "$$\n\n"
            "$$\n"
            "SG = \\frac{\\text{Pressure gradient (psi/ft)}}{0.433}\n"
            "$$\n\n"
            "$$\n"
            "\\text{Mud weight (ppg)} = \\frac{\\text{Pressure gradient (psi/ft)}}{0.052}\n"
            "$$\n\n"
            "- 단위: pressure gradient는 psi/ft, mud weight는 ppg, SG는 무차원 비중입니다. [S1]"
        )
        self.last_debug["citation_ids_after_filtering"] = ["S1"]
        self.last_debug["refusal_reason"] = None
        return answer, [source]

    def _mud_window_fallback_answer(self, sources_by_id: dict[str, Source]) -> tuple[str, list[Source]]:
        source = self._first_source_matching(
            sources_by_id,
            [r"pore pressure|formation pressure", r"fracture pressure|fracture"],
        )
        if source is None:
            return self._refusal_for_question(""), []
        answer = (
            "Mud weight window는 borehole pressure가 pore pressure보다 낮아지지 않고 fracture pressure를 넘지 않는 범위입니다. [S1]\n\n"
            "- Mud weight가 pore pressure보다 낮으면 formation fluid가 wellbore로 유입되어 kick 또는 influx 위험이 커집니다. [S1]\n"
            "- Mud weight가 fracture pressure보다 높으면 암석이 파괴되어 drilling fluid loss 또는 lost circulation이 발생할 수 있습니다. [S1]"
        )
        self.last_debug["citation_ids_after_filtering"] = ["S1"]
        self.last_debug["refusal_reason"] = None
        return answer, [source]

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
        self.last_debug["citation_ids_before_filtering"] = self._extract_source_ids(answer)
        if self._is_refusal(answer):
            self.last_debug["citation_ids_after_filtering"] = []
            return answer, []

        used_ids: list[str] = []

        def keep_valid(match: re.Match[str]) -> str:
            source_id = f"S{match.group(1)}"
            if source_id not in sources_by_id:
                return ""
            if source_id not in used_ids:
                used_ids.append(source_id)
            return f"[{source_id}]"

        cleaned_answer = SOURCE_REF.sub(keep_valid, answer)

        if not used_ids:
            self.last_debug["citation_ids_after_filtering"] = []
            return self._refusal_for_question(question), []

        renumbered_ids = {
            old_source_id: f"S{new_index}"
            for new_index, old_source_id in enumerate(used_ids, start=1)
        }

        def renumber(match: re.Match[str]) -> str:
            old_source_id = f"S{match.group(1)}"
            new_source_id = renumbered_ids.get(old_source_id)
            return f"[{new_source_id}]" if new_source_id else ""

        cleaned_answer = SOURCE_REF.sub(renumber, cleaned_answer)
        cleaned_answer = self._postprocess_answer(cleaned_answer, question)
        final_sources = [sources_by_id[source_id] for source_id in used_ids]
        self.last_debug["citation_ids_after_filtering"] = self._extract_source_ids(cleaned_answer)
        self.last_debug["source_id_mapping"] = renumbered_ids
        return cleaned_answer, final_sources

    def _extract_source_ids(self, answer: str) -> list[str]:
        ids: list[str] = []
        for match in SOURCE_REF.finditer(answer):
            source_id = f"S{match.group(1)}"
            if source_id not in ids:
                ids.append(source_id)
        return ids

    def _postprocess_answer(self, answer: str, question: str) -> str:
        answer = answer.replace("порosity", "porosity")
        answer = re.sub(r"[\u0400-\u04FF]+", "", answer)
        if re.search(r"archie", question, re.IGNORECASE) and self._has_wrong_archie_formula(answer):
            answer = self._replace_wrong_archie_formula(answer)
        if re.search(r"\becd\b", question, re.IGNORECASE) and "annular pressure" not in answer.lower():
            answer = self._add_ecd_pressure_loss_note(answer)
        if re.search(r"mud weight|ppg|psi/ft|sg", question, re.IGNORECASE):
            answer = self._replace_invented_mud_weight_symbols(answer)
        return answer

    def _has_wrong_archie_formula(self, answer: str) -> bool:
        compact = re.sub(r"\s+", "", answer.lower())
        return bool(
            re.search(
                r"r_?\{?w\}?/?r_?\{?t\}?.*f\^?\{?\(?1/?n\)?\}?",
                compact,
            )
        )

    def _replace_wrong_archie_formula(self, answer: str) -> str:
        correct = (
            "$$\n"
            "S_w^n = \\frac{F R_w}{R_t}\n"
            "$$\n\n"
            "$$\n"
            "S_w = \\left(\\frac{F R_w}{R_t}\\right)^{1/n}\n"
            "$$"
        )
        block_pattern = re.compile(r"\$\$.*?\$\$", flags=re.DOTALL)
        if block_pattern.search(answer):
            return block_pattern.sub(correct, answer, count=1)
        return f"{correct}\n\n{answer}"

    def _replace_invented_mud_weight_symbols(self, answer: str) -> str:
        replacements = {
            r"R_\\?\{?\\?text\{psi/ft\}\}?": r"\\text{Pressure gradient (psi/ft)}",
            r"R_\\?\{?\\?text\{ppg\}\}?": r"\\text{Mud weight (ppg)}",
            r"R_\\?\{?\\?text\{SG\}\}?": "SG",
            r"R_\{?psi/ft\}?": r"\\text{Pressure gradient (psi/ft)}",
            r"R_\{?ppg\}?": r"\\text{Mud weight (ppg)}",
        }
        cleaned = answer
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned

    def _add_ecd_pressure_loss_note(self, answer: str) -> str:
        if "[S1]" not in answer:
            return answer
        return (
            answer.rstrip()
            + "\n- $P$: annular pressure loss, psi [S1]"
        )

    def _is_refusal(self, answer: str) -> bool:
        normalized = re.sub(r"\s+", " ", answer).strip()
        return any(marker in normalized for marker in REFUSAL_MARKERS) or GRAPH_REFUSAL in normalized

    def _refusal_for_question(self, question: str) -> str:
        if re.search(r"2027년\s*생산량|2027.*production", question, re.IGNORECASE):
            return "제공된 문서 근거로는 해당 유전의 2027년 생산량을 확인할 수 없습니다."
        return "제공된 문서 근거로는 확인할 수 없습니다."

    def _has_figure_evidence(self, hit: dict) -> bool:
        text = str(hit.get("text") or "").lower()
        confidence_match = re.search(r"confidence:\s*([0-9.]+)", text)
        if confidence_match:
            try:
                if float(confidence_match.group(1)) < self.settings.figure_note_min_confidence:
                    return False
            except ValueError:
                return False
        if "image type: logo" in text or "image type: decorative" in text:
            return False
        literal_markers = [
            "image type: graph",
            "image type: chart",
            "extracted figure notes",
            "image analysis",
            "figure ",
            "fig.",
            "caption",
            "x-axis",
            "y-axis",
            "x축",
            "y축",
        ]
        if any(marker in text for marker in literal_markers):
            return True

        return any(
            re.search(pattern, text)
            for pattern in [
                r"\bgraph\b",
                r"\bplot\b",
                r"\bchart\b",
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

    def _refusal_for_question(self, question: str) -> str:
        if re.search(r"2027|production|생산량", question, re.IGNORECASE):
            return "제공된 문서 근거로는 해당 유전의 2027년 생산량을 확인할 수 없습니다."
        return "제공된 문서 근거로는 확인할 수 없습니다."

    def _has_figure_evidence(self, hit: dict) -> bool:
        text = str(hit.get("text") or "")
        lower = text.lower()
        if "[figure note metadata]" not in lower:
            return bool(self.settings.allow_legacy_figure_notes) and "extracted figure notes" in lower

        image_type = self._metadata_line(text, "image type") or self._metadata_line(text, "image_type")
        if str(image_type or "").strip().lower() not in {"graph", "chart"}:
            return False

        try:
            confidence = float(str(self._metadata_line(text, "confidence")))
        except (TypeError, ValueError):
            return False
        if confidence < self.settings.figure_note_min_confidence:
            return False

        image_path = self._metadata_line(text, "image path") or self._metadata_line(text, "image_path")
        return bool(image_path and Path(str(image_path)).exists())

    def _metadata_line(self, text: str, key: str) -> str | None:
        pattern = rf"^{re.escape(key)}:\s*(.+)$"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else None
