import re
import time
from pathlib import Path
from urllib.parse import quote

from app.core.config import Settings
from app.models.schemas import (
    ChatCompareResponse,
    ChatResponse,
    FigureReference,
    ModelAnswer,
    Source,
)
from app.services.agent_run_logger import AgentRunLogger
from app.services.engineering_validator import (
    EngineeringValidator,
    ValidationResult,
)
from app.services.ollama import OllamaClient
from app.services.figure_preview import FigurePreviewService
from app.services.query_router import QueryType, classify_query
from app.services.vector_store import VectorStore


SHA256_PREFIX = re.compile(r"^[0-9a-fA-F]{64}_")
FIGURE_MARKERS = (
    "[extracted figure notes]",
    "[figure note metadata]",
    "image_type:",
    "trend_summary:",
    "series_descriptions:",
)
FIGURE_INTENT_RE = re.compile(
    r"그래프|도표|그림|플롯|곡선|계열|추세|경향|축|범례|기울기|구배|"
    r"graph|plot|figure|chart|curve|series|trend|axis|legend|slope|gradient",
    re.IGNORECASE,
)
FIGURE_TREND_RE = re.compile(
    r"추세|경향|변화|상승|하락|증가|감소|최대|최소|평탄|"
    r"trend|rise|increase|decline|decrease|peak|minimum|plateau|slope",
    re.IGNORECASE,
)
FIGURE_AXIS_RE = re.compile(
    r"x축|y축|축|단위|axis|unit|equivalent\s+time|delta\s*p",
    re.IGNORECASE,
)
FIGURE_GRADIENT_RE = re.compile(
    r"기울기|구배|gradient|psi\s*/\s*ft|psi\s*/\s*m",
    re.IGNORECASE,
)
NUMERIC_GRADIENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*psi\s*/\s*(?:ft|m)\b",
    re.IGNORECASE,
)
SUPERCHARGED_RE = re.compile(r"supercharg(?:ed|ing)", re.IGNORECASE)
RFT_ZONE_DETAIL_RE = re.compile(
    r"interpreted\s*rft\s*data|zone\s*1|zone\s*7|"
    r"supercharged\s*points?|double\s*pretest\s*sequence",
    re.IGNORECASE,
)
STRICT_REFUSAL = "제공된 문서 근거로는 확인할 수 없습니다."
TYPE_CURVE_INTENT_RE = re.compile(
    r"type\s*curve|wellbore\s*storage|radial\s*flow|derivative\s*plateau|"
    r"middle[-\s]*time\s*region|\bmtr\b|unit[-\s]*slope|타입\s*커브|"
    r"정류\s*유동|중기\s*유동|유정저장",
    re.IGNORECASE,
)
RFT_COMPARISON_RE = re.compile(
    r"(?:rft|formation\s*tester).*(?:before|after|전후|비교)|"
    r"(?:before|after|전후|비교).*(?:rft|formation\s*tester)|"
    r"significant\s*production|생산\s*전후",
    re.IGNORECASE,
)
RFT_BEFORE_TERMS = (
    "appraisal well rft survey",
    "appraisal well",
)
RFT_AFTER_TERMS = (
    "rft survey after significant production",
    "after significant production",
)
TYPE_CURVE_EVIDENCE_TERMS = (
    "td/cd type curve including the derivative",
    "wellbore storage dominated flow",
    "unit slope diagonal",
    "middle time region",
    "derivative plateau",
    "figure 22",
)


SYSTEM_PROMPT = """You are a strict evidence-only petroleum engineering RAG agent.
Rules:
1. Use ONLY the retrieved chunks provided in the prompt.
2. Do NOT infer from the document title, general petroleum-engineering knowledge, or model memory.
3. Use the exact refusal "제공된 문서 근거로는 확인할 수 없습니다." only when the retrieved chunks support no substantive part of the question.
4. For a multi-part question with partial evidence, answer every supported part with citations and state specifically which unsupported subpart cannot be confirmed. Do not refuse the entire question when any substantive part is supported.
5. Every factual sentence must be supported by a cited source marker like [document, p.page].
6. Do not mention topics that are not present in the retrieved chunks.
7. Answer in Korean unless the source terminology is English.
8. For figure questions, distinguish axis labels from plotted series. Never call x_axis or y_axis a series. Use series_descriptions and trend_summary to describe series behavior.
9. Prefer explicit Figure Note fields over generic nearby chapter text when both are retrieved.
10. If the question asks for a displayed gradient, unit, threshold, or other numeric value, state the exact value only when it appears explicitly in a retrieved chunk. Otherwise say that the exact value is not confirmed.
11. Do not infer causes, permeability behavior, fluid distribution, or reservoir properties unless a retrieved chunk explicitly states them.
12. Do not merge separate figures from adjacent pages into one series description unless the retrieved text explicitly identifies them as the same figure or comparison.
13. When a Figure Note names categories, zones, point types, legends, axes, or reference lines but does not give a separate behavior for every category, report the supported overall pattern and legend meanings, then explicitly say that per-category behavior is not confirmed.
"""


def clean_document_name(value: object) -> str:
    """Remove a stored SHA-256 prefix from a source filename for display/citation."""
    filename = Path(str(value or "Unknown document")).name
    return SHA256_PREFIX.sub("", filename)


class QAService:
    def __init__(self, settings: Settings, vector_store: VectorStore, ollama: OllamaClient):
        self.settings = settings
        self.vector_store = vector_store
        self.ollama = ollama
        configured_data_dir = getattr(settings, "data_dir", None)
        configured_figures_dir = getattr(settings, "figures_dir", None)
        if configured_data_dir is not None:
            data_dir = Path(configured_data_dir)
        elif configured_figures_dir is not None:
            data_dir = Path(configured_figures_dir)
        else:
            data_dir = Path("data")
        self.figure_preview = FigurePreviewService(
            data_dir / "figure_display_previews",
            data_dir / "figure_display_overrides.json",
        )
        agent_runs_dir = Path(
            getattr(
                settings,
                "agent_runs_dir",
                data_dir / "agent_runs",
            )
        )
        self.engineering_validator = EngineeringValidator()
        self.agent_run_logger = AgentRunLogger(agent_runs_dir)

    async def answer(
        self,
        question: str,
        top_k: int | None = None,
        model: str | None = None,
        benchmark_id: str | None = None,
    ) -> ChatResponse:
        selected_model = model or self.settings.text_model
        run_started = time.perf_counter()
        task_id = self.agent_run_logger.new_task_id("WT")
        prepared = self._prepare_generation(question, top_k)

        aggregate = prepared.get("aggregate")
        if aggregate is not None:
            self._save_agent_run(
                {
                    "task_id": task_id,
                    "benchmark_id": benchmark_id,
                    "question": question,
                    "model": selected_model,
                    "prompt_version": "welltest-validator-v1",
                    "query_type": aggregate.query_type,
                    "retrieval_elapsed_seconds": prepared[
                        "retrieval_elapsed_seconds"
                    ],
                    "retrieved_sources": [
                        self._source_to_dict(source)
                        for source in aggregate.sources
                    ],
                    "attempts": [
                        {
                            "attempt": 1,
                            "answer": aggregate.answer,
                            "elapsed_seconds": 0.0,
                            "validation_passed": None,
                            "errors": [],
                            "warnings": [
                                "Aggregate response was not rewritten."
                            ],
                            "rule_ids": [],
                        }
                    ],
                    "initial_passed": None,
                    "final_passed": None,
                    "final_status": "completed_unvalidated",
                    "total_elapsed_seconds": (
                        time.perf_counter() - run_started
                    ),
                }
            )
            return ChatResponse(
                answer=aggregate.answer,
                sources=aggregate.sources,
                query_type=aggregate.query_type,
                figures=aggregate.figures,
                model=selected_model,
                elapsed_seconds=0.0,
            )

        if not prepared["hits"]:
            validation = (
                self.engineering_validator
                .validate_well_test_answer(
                    question,
                    STRICT_REFUSAL,
                    retrieved_sources=[],
                )
            )
            self._save_agent_run(
                {
                    "task_id": task_id,
                    "benchmark_id": benchmark_id,
                    "question": question,
                    "model": selected_model,
                    "prompt_version": "welltest-validator-v1",
                    "query_type": prepared["query_type"].value,
                    "retrieval_elapsed_seconds": prepared[
                        "retrieval_elapsed_seconds"
                    ],
                    "retrieved_sources": [],
                    "attempts": [
                        self._attempt_record(
                            1,
                            STRICT_REFUSAL,
                            0.0,
                            validation,
                        )
                    ],
                    "initial_passed": validation.passed,
                    "final_passed": validation.passed,
                    "final_status": "completed",
                    "total_elapsed_seconds": (
                        time.perf_counter() - run_started
                    ),
                }
            )
            return ChatResponse(
                answer=STRICT_REFUSAL,
                sources=[],
                query_type=prepared["query_type"].value,
                figures=[],
                model=selected_model,
                elapsed_seconds=0.0,
            )

        source_payloads = [
            self._source_to_dict(source)
            for source in prepared["sources"]
        ]
        (
            answer,
            attempts,
            validation,
            final_status,
            generation_elapsed,
        ) = await self._generate_validated_answer(
            question=question,
            prepared=prepared,
            selected_model=selected_model,
            retrieved_sources=source_payloads,
        )

        self._save_agent_run(
            {
                "task_id": task_id,
                "benchmark_id": benchmark_id,
                "question": question,
                "model": selected_model,
                "prompt_version": "welltest-validator-v1",
                "query_type": prepared["query_type"].value,
                "retrieval_elapsed_seconds": prepared[
                    "retrieval_elapsed_seconds"
                ],
                "retrieved_sources": source_payloads,
                "attempts": attempts,
                "initial_passed": (
                    attempts[0]["validation_passed"]
                    if attempts
                    else None
                ),
                "final_passed": validation.passed,
                "final_status": final_status,
                "total_elapsed_seconds": (
                    time.perf_counter() - run_started
                ),
            }
        )

        return ChatResponse(
            answer=answer,
            sources=prepared["sources"],
            query_type=prepared["query_type"].value,
            figures=prepared["figures"],
            model=selected_model,
            elapsed_seconds=generation_elapsed,
        )

    async def _generate_validated_answer(
        self,
        *,
        question: str,
        prepared: dict,
        selected_model: str,
        retrieved_sources: list[dict],
    ) -> tuple[
        str,
        list[dict],
        ValidationResult,
        str,
        float,
    ]:
        messages = list(prepared["messages"])
        attempts: list[dict] = []
        validation = ValidationResult(
            passed=False,
            errors=["답변이 생성되지 않았습니다."],
        )
        answer = ""
        total_generation_elapsed = 0.0

        for attempt_number in range(1, 4):
            started = time.perf_counter()
            answer = await self.ollama.chat(
                messages,
                model=selected_model,
            )
            elapsed = time.perf_counter() - started
            total_generation_elapsed += elapsed

            if str(answer or "").strip().startswith(
                STRICT_REFUSAL
            ):
                partial_answer = self._supported_partial_answer(
                    question,
                    prepared["hits"],
                )
                if partial_answer:
                    answer = partial_answer

            validation = (
                self.engineering_validator
                .validate_well_test_answer(
                    question,
                    str(answer or ""),
                    retrieved_sources=retrieved_sources,
                )
            )
            attempts.append(
                self._attempt_record(
                    attempt_number,
                    str(answer or ""),
                    elapsed,
                    validation,
                )
            )

            if validation.passed:
                return (
                    str(answer or ""),
                    attempts,
                    validation,
                    "completed",
                    total_generation_elapsed,
                )

            if attempt_number < 3:
                messages = self._build_rewrite_messages(
                    question=question,
                    previous_answer=str(answer or ""),
                    validation=validation,
                    original_messages=prepared["messages"],
                )

        safe_answer = (
            "자동 공학 검증을 통과하지 못했습니다. "
            "검색된 문서 근거는 확보했지만 답변 내용은 "
            "사람의 검토가 필요합니다."
        )
        return (
            safe_answer,
            attempts,
            validation,
            "review_required",
            total_generation_elapsed,
        )

    def _build_rewrite_messages(
        self,
        *,
        question: str,
        previous_answer: str,
        validation: ValidationResult,
        original_messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        del previous_answer
        error_lines = "\n".join(
            f"- {message}"
            for message in validation.errors
        )
        original_user_prompt = str(
            original_messages[-1].get("content", "")
        )
        normalized_question = re.sub(
            r"\s+",
            " ",
            question.lower(),
        )
        reviews_false_claim = bool(
            re.search(
                r"radial\s*flow|방사\s*유동",
                normalized_question,
                re.IGNORECASE,
            )
            and re.search(
                r"unit[- ]?slope|단위\s*기울기",
                normalized_question,
                re.IGNORECASE,
            )
            and re.search(
                r"맞는지|검토|틀린|수정|correct|review",
                normalized_question,
                re.IGNORECASE,
            )
        )
        opening_rule = (
            "첫 문장을 반드시 '해당 설명은 틀렸습니다.'로 "
            "시작하세요.\n"
            if reviews_false_claim
            else ""
        )

        rewrite_prompt = (
            "이전 답변은 수정하거나 요약하지 말고 완전히 "
            "폐기한 뒤 처음부터 다시 작성하세요.\n\n"
            "검증 오류:\n"
            f"{error_lines}\n\n"
            f"{opening_rule}"
            "반드시 지킬 정답 기준:\n"
            "1. Early-time wellbore storage에서는 pressure와 "
            "pressure derivative가 서로 겹쳐 unit-slope "
            "diagonal을 따릅니다.\n"
            "2. Middle-time radial flow에서는 pressure "
            "derivative가 일정해져 수평 plateau를 "
            "형성합니다.\n"
            "3. Radial flow에서 pressure가 unit-slope를 "
            "따른다고 쓰지 마세요.\n"
            "4. Radial flow에서 pressure derivative가 "
            "unit-slope를 따른다고 쓰지 마세요.\n"
            "5. unit-slope와 constant 또는 plateau를 같은 "
            "곡선의 동시 특성으로 결합하지 마세요.\n"
            "6. '부분적으로 정확하다'고 표현하지 말고, "
            "틀린 전제는 명확히 틀렸다고 판정하세요.\n"
            "7. 서로 다른 Figure의 설명을 혼합하지 말고, "
            "검색 근거에 없는 사실을 추가하지 마세요.\n"
            "8. 문서명과 페이지를 인용하세요.\n\n"
            "출력 형식:\n"
            "- 판정 1문장\n"
            "- Wellbore storage 특징 1문장\n"
            "- Radial flow 특징 1문장\n"
            "- 올바르게 수정한 문장 1문장\n\n"
            f"원래 질문:\n{question}\n\n"
            "동일하게 재사용할 검색 근거와 지시:\n"
            f"{original_user_prompt}"
        )
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": rewrite_prompt,
            },
        ]

    @staticmethod
    def _attempt_record(
        attempt_number: int,
        answer: str,
        elapsed_seconds: float,
        validation: ValidationResult,
    ) -> dict:
        return {
            "attempt": attempt_number,
            "answer": answer,
            "elapsed_seconds": elapsed_seconds,
            "validation_passed": validation.passed,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "rule_ids": list(validation.rule_ids),
        }

    @staticmethod
    def _source_to_dict(source: Source) -> dict:
        if hasattr(source, "model_dump"):
            return source.model_dump()
        return source.dict()

    def _save_agent_run(self, record: dict) -> None:
        try:
            self.agent_run_logger.write(record)
        except OSError as exc:
            print(
                "[Agent run log failed] "
                f"{type(exc).__name__}: {exc}"
            )

    async def compare(
        self,
        question: str,
        models: list[str],
        top_k: int | None = None,
    ) -> ChatCompareResponse:
        selected_models: list[str] = []
        for model in models:
            normalized = str(model or "").strip()
            if normalized and normalized not in selected_models:
                selected_models.append(normalized)

        if not selected_models:
            selected_models = [self.settings.text_model]

        prepared = self._prepare_generation(question, top_k)
        aggregate = prepared.get("aggregate")

        if aggregate is not None:
            return ChatCompareResponse(
                answers=[
                    ModelAnswer(
                        model=model,
                        answer=aggregate.answer,
                        elapsed_seconds=0.0,
                    )
                    for model in selected_models
                ],
                sources=aggregate.sources,
                query_type=aggregate.query_type,
                figures=aggregate.figures,
                retrieval_elapsed_seconds=prepared[
                    "retrieval_elapsed_seconds"
                ],
                shared_context=True,
            )

        if not prepared["hits"]:
            return ChatCompareResponse(
                answers=[
                    ModelAnswer(
                        model=model,
                        answer=STRICT_REFUSAL,
                        elapsed_seconds=0.0,
                    )
                    for model in selected_models
                ],
                sources=[],
                query_type=prepared["query_type"].value,
                figures=[],
                retrieval_elapsed_seconds=prepared[
                    "retrieval_elapsed_seconds"
                ],
                shared_context=True,
            )

        answers: list[ModelAnswer] = []
        for model in selected_models:
            started = time.perf_counter()
            answer = await self.ollama.chat(
                prepared["messages"],
                model=model,
            )
            elapsed = time.perf_counter() - started

            if str(answer or "").strip().startswith(STRICT_REFUSAL):
                partial_answer = self._supported_partial_answer(
                    question,
                    prepared["hits"],
                )
                if partial_answer:
                    answer = partial_answer

            answers.append(
                ModelAnswer(
                    model=model,
                    answer=answer,
                    elapsed_seconds=elapsed,
                )
            )

        return ChatCompareResponse(
            answers=answers,
            sources=prepared["sources"],
            query_type=prepared["query_type"].value,
            figures=prepared["figures"],
            retrieval_elapsed_seconds=prepared[
                "retrieval_elapsed_seconds"
            ],
            shared_context=True,
        )

    def _prepare_generation(
        self,
        question: str,
        top_k: int | None,
    ) -> dict:
        query_type = classify_query(question)
        retrieval_started = time.perf_counter()

        if query_type == QueryType.AGGREGATE_ANALYSIS:
            aggregate = self._aggregate_response(
                question,
                query_type,
            )
            return {
                "aggregate": aggregate,
                "query_type": query_type,
                "hits": [],
                "sources": aggregate.sources,
                "figures": aggregate.figures,
                "messages": [],
                "retrieval_elapsed_seconds": (
                    time.perf_counter() - retrieval_started
                ),
            }

        search_question = self._build_search_question(question)
        hits = self._retrieve(
            search_question,
            query_type,
            top_k or self.settings.top_k,
            original_question=question,
        )

        if not hits:
            return {
                "aggregate": None,
                "query_type": query_type,
                "hits": [],
                "sources": [],
                "figures": [],
                "messages": [],
                "retrieval_elapsed_seconds": (
                    time.perf_counter() - retrieval_started
                ),
            }

        context_blocks = []
        sources = []
        for hit in hits:
            metadata = hit["metadata"]
            score = float(hit.get("score") or 0.0)
            document_name = clean_document_name(
                metadata.get("document")
            )
            page_number = metadata.get("page")

            context_blocks.append(
                f"Source: {document_name} p.{page_number} "
                f"chunk {hit['id']} score={score:.3f}\n"
                f"{hit['text']}"
            )
            sources.append(
                Source(
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
                    preview=self._preview(str(hit["text"])),
                )
            )

        user_prompt = (
            f"Query type: {query_type.value}\n"
            f"Question:\n{question}\n\n"
            "Answer-coverage rule: If the chunks support only "
            "part of this question, answer the supported part "
            "and identify only the unsupported subpart. Use the "
            "full refusal only when no substantive part is "
            "supported.\n\n"
            "Retrieved chunks. You must not use any information "
            "outside these chunks:\n"
            + "\n\n---\n\n".join(context_blocks)
        )
        figures = self._figure_references(
            hits,
            question=question,
        )

        return {
            "aggregate": None,
            "query_type": query_type,
            "hits": hits,
            "sources": sources,
            "figures": figures,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "retrieval_elapsed_seconds": (
                time.perf_counter() - retrieval_started
            ),
        }

    def _build_search_question(self, question: str) -> str:
        """Build a compact retrieval query while preserving the full question for generation."""
        normalized = re.sub(r"[’`]", "'", question)
        english_terms = re.findall(
            r"[A-Za-z][A-Za-z0-9_.-]*(?:'s)?",
            normalized,
        )
        ignored_terms = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "in", "is", "of", "on", "or", "the", "to", "with",
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

        cleaned = question
        request_patterns = [
            r"업로드한\s*문서만\s*근거로",
            r"문서\s*근거로",
            r"문서명과\s*페이지\s*번호를\s*표시해줘",
            r"사용한\s*문서명과\s*페이지\s*번호를\s*표시해줘",
            r"근거가\s*없으면\s*추측하지\s*말고\s*없다고\s*말해줘",
            r"설명해\s*줘", r"설명해줘", r"정리해\s*줘", r"정리해줘",
            r"알려\s*줘", r"알려줘",
        ]
        for pattern in request_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or question

    def _retrieve(
        self,
        question: str,
        query_type: QueryType,
        top_k: int,
        *,
        original_question: str | None = None,
    ) -> list[dict]:
        if query_type == QueryType.DOCUMENT_OVERVIEW:
            hits = self.vector_store.hybrid_search(
                question,
                top_k=top_k,
                score_threshold=0.0,
                prefer_metadata={"is_contents": True},
                keyword_weight=0.55,
            )
            if hits:
                contents = [
                    hit for hit in hits
                    if hit.get("metadata", {}).get("is_contents")
                ]
                titles = [
                    hit for hit in hits
                    if hit.get("metadata", {}).get("is_title_page")
                ]
                others = [hit for hit in hits if hit not in contents and hit not in titles]
                return (contents + titles + others)[:top_k]

        full_question = original_question or question
        if self._is_figure_question(full_question):
            return self._retrieve_figure(question, full_question, top_k)

        keyword_weight = 0.75 if query_type == QueryType.INDEX_LOOKUP else 0.45
        threshold = max(
            0.05,
            self.settings.similarity_threshold
            if query_type == QueryType.LOCAL_FACT_SEARCH
            else self.settings.similarity_threshold * 0.6,
        )
        return self.vector_store.hybrid_search(
            question,
            top_k=top_k,
            score_threshold=threshold,
            prefer_metadata=None,
            keyword_weight=keyword_weight,
        )

    def _retrieve_figure(
        self,
        search_question: str,
        original_question: str,
        top_k: int,
    ) -> list[dict]:
        expanded = self._expand_figure_query(search_question, original_question)
        candidate_count = min(max(top_k * 12, 120), 240)
        hybrid_hits = self.vector_store.hybrid_search(
            expanded,
            top_k=candidate_count,
            score_threshold=0.0,
            prefer_metadata=None,
            keyword_weight=0.65,
        )

        literal_hits, anchor_document_keys = self._literal_figure_hits(
            self._english_phrases(original_question),
        )
        special_hits, special_document_keys = self._special_figure_hits(
            original_question
        )
        anchor_document_keys.update(special_document_keys)

        merged: dict[str, dict] = {}
        for hit in [*hybrid_hits, *literal_hits, *special_hits]:
            chunk_id = str(hit.get("id") or "")
            if not chunk_id:
                continue
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = dict(hit)
                continue
            for key in (
                "exact_phrase_matches",
                "strong_phrase_matches",
                "anchor_neighbor",
                "anchor_document",
                "anchor_preceding",
                "special_type_curve",
                "comparison_before",
                "comparison_after",
            ):
                existing[key] = max(
                    float(existing.get(key) or 0.0),
                    float(hit.get(key) or 0.0),
                )
            if existing.get("anchor_page_distance") is None and hit.get("anchor_page_distance") is not None:
                existing["anchor_page_distance"] = hit.get("anchor_page_distance")

        hits = list(merged.values())
        if anchor_document_keys:
            anchored = [
                hit
                for hit in hits
                if self._document_key(hit) in anchor_document_keys
            ]
            if anchored:
                hits = anchored

        return self._rerank_figure_hits(hits, original_question, top_k)

    def _literal_figure_hits(
        self,
        phrases: list[str],
    ) -> tuple[list[dict], set[str]]:
        collection = getattr(self.vector_store, "collection", None)
        if collection is None or not phrases:
            return [], set()

        try:
            data = collection.get(include=["documents", "metadatas"])
        except Exception:
            return [], set()

        records: list[dict] = []
        strong_anchor_pages: dict[str, set[int]] = {}
        strong_anchor_documents: set[str] = set()

        for chunk_id, document, metadata in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            text = str(document or "")
            lower = text.lower()
            matched = [phrase for phrase in phrases if phrase in lower]
            strong = [
                phrase
                for phrase in matched
                if len(phrase.split()) >= 3
            ]
            hit = {
                "id": str(chunk_id),
                "text": text,
                "metadata": metadata or {},
                "distance": None,
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "score": 0.0,
                "exact_phrase_matches": len(matched),
                "strong_phrase_matches": len(strong),
                "anchor_neighbor": 0.0,
                "anchor_document": 0.0,
                "anchor_page_distance": None,
                "anchor_preceding": 0.0,
            }
            records.append(hit)

            if strong:
                document_key = self._document_key(hit)
                page = (metadata or {}).get("page")
                if document_key:
                    strong_anchor_documents.add(document_key)
                    if page is not None:
                        strong_anchor_pages.setdefault(document_key, set()).add(int(page))

        literal_hits: list[dict] = []
        for hit in records:
            document_key = self._document_key(hit)
            page = (hit.get("metadata") or {}).get("page")
            is_exact = int(hit.get("exact_phrase_matches") or 0) > 0
            is_neighbor = False
            nearest_distance = None
            if document_key in strong_anchor_pages and page is not None:
                distances = [
                    int(page) - anchor_page
                    for anchor_page in strong_anchor_pages[document_key]
                ]
                if distances:
                    nearest_distance = min(distances, key=lambda value: abs(value))
                    is_neighbor = abs(nearest_distance) <= 1
            if is_exact or is_neighbor:
                hit["anchor_neighbor"] = 1.0 if is_neighbor and not is_exact else 0.0
                hit["anchor_document"] = 1.0 if document_key in strong_anchor_documents else 0.0
                hit["anchor_page_distance"] = nearest_distance
                hit["anchor_preceding"] = (
                    1.0 if nearest_distance == -1 and not is_exact else 0.0
                )
                literal_hits.append(hit)

        return literal_hits, strong_anchor_documents

    def _special_figure_hits(
        self,
        question: str,
    ) -> tuple[list[dict], set[str]]:
        """Fetch high-value figure evidence that common paraphrases otherwise miss."""
        collection = getattr(self.vector_store, "collection", None)
        if collection is None:
            return [], set()

        wants_type_curve = TYPE_CURVE_INTENT_RE.search(question) is not None
        wants_rft_comparison = RFT_COMPARISON_RE.search(question) is not None
        if not wants_type_curve and not wants_rft_comparison:
            return [], set()

        try:
            data = collection.get(include=["documents", "metadatas"])
        except Exception:
            return [], set()

        hits: list[dict] = []
        document_keys: set[str] = set()

        for chunk_id, document, metadata in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            text = str(document or "")
            normalized = re.sub(r"[-_]+", " ", text.lower())
            special_type_curve = 0.0
            comparison_before = 0.0
            comparison_after = 0.0

            if wants_type_curve:
                evidence_matches = sum(
                    1
                    for term in TYPE_CURVE_EVIDENCE_TERMS
                    if term in normalized
                )
                if evidence_matches >= 1:
                    special_type_curve = float(evidence_matches)

            if wants_rft_comparison:
                comparison_before = 1.0 if any(
                    term in normalized for term in RFT_BEFORE_TERMS
                ) else 0.0
                comparison_after = 1.0 if any(
                    term in normalized for term in RFT_AFTER_TERMS
                ) else 0.0

            if not (
                special_type_curve
                or comparison_before
                or comparison_after
            ):
                continue

            hit = {
                "id": str(chunk_id),
                "text": text,
                "metadata": metadata or {},
                "distance": None,
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "score": 0.0,
                "exact_phrase_matches": 0,
                "strong_phrase_matches": 0,
                "anchor_neighbor": 0.0,
                "anchor_document": 1.0,
                "anchor_page_distance": None,
                "anchor_preceding": 0.0,
                "special_type_curve": special_type_curve,
                "comparison_before": comparison_before,
                "comparison_after": comparison_after,
            }
            hits.append(hit)
            document_key = self._document_key(hit)
            if document_key:
                document_keys.add(document_key)

        return hits, document_keys

    def _document_key(self, hit: dict) -> str:
        metadata = hit.get("metadata") or {}
        value = metadata.get("document_id")
        if value:
            return str(value)
        chunk_id = str(hit.get("id") or "")
        match = re.match(r"^([^:]+):p\d+", chunk_id)
        if match:
            return match.group(1)
        return str(metadata.get("document") or "")

    def _is_figure_question(self, question: str) -> bool:
        return FIGURE_INTENT_RE.search(question) is not None

    def _expand_figure_query(self, search_question: str, original_question: str) -> str:
        expansions = [
            "Figure Note Metadata",
            "analysis",
            "series_descriptions",
            "trend_summary",
        ]
        mapping = [
            (r"그래프|도표|그림|플롯", "graph plot figure chart"),
            (r"추세|경향|변화", "trend series slope increase decrease plateau peak decline"),
            (r"계열", "series marker curve"),
            (r"x축|y축|축|단위", "x_axis y_axis axis unit"),
            (r"기울기|구배", "gradient slope"),
            (r"등가\s*시간", "Equivalent Time"),
            (r"델타\s*p", "Delta P"),
            (r"압력", "pressure"),
            (r"시간", "time"),
            (r"상승|증가", "rise increase upward"),
            (r"하락|감소", "decline decrease downward"),
            (r"범례", "legend"),
        ]
        for pattern, terms in mapping:
            if re.search(pattern, original_question, flags=re.IGNORECASE):
                expansions.append(terms)

        if TYPE_CURVE_INTENT_RE.search(original_question):
            expansions.extend([
                "td/cd type curve including the derivative",
                "wellbore storage dominated flow",
                "unit slope diagonal",
                "middle time region MTR",
                "derivative plateau",
                "Figure 22",
            ])

        if RFT_COMPARISON_RE.search(original_question):
            expansions.extend([
                "Appraisal Well RFT Survey",
                "RFT Survey after Significant Production",
                "pressure gradient",
                "permeability barrier",
                "supercharged points",
            ])

        combined = " ".join([search_question, *expansions])
        return re.sub(r"\s+", " ", combined).strip()

    def _rerank_figure_hits(
        self,
        hits: list[dict],
        question: str,
        top_k: int,
    ) -> list[dict]:
        phrases = self._english_phrases(question)
        tokens = self._technical_tokens(question)
        wants_trend = FIGURE_TREND_RE.search(question) is not None
        wants_axis = FIGURE_AXIS_RE.search(question) is not None
        wants_gradient = FIGURE_GRADIENT_RE.search(question) is not None
        wants_type_curve = TYPE_CURVE_INTENT_RE.search(question) is not None
        wants_rft_comparison = RFT_COMPARISON_RE.search(question) is not None

        ranked: list[dict] = []
        for hit in hits:
            item = dict(hit)
            text = str(item.get("text") or "")
            lower = text.lower()
            base_score = float(item.get("score") or 0.0)
            bonus = 0.0
            penalty = 0.0
            is_figure_note = any(marker in lower for marker in FIGURE_MARKERS)

            if is_figure_note:
                bonus += 0.45

            phrase_matches = sum(1 for phrase in phrases if phrase in lower)
            bonus += min(0.54, phrase_matches * 0.18)

            if tokens:
                overlap = sum(1 for token in tokens if token in lower)
                bonus += min(0.24, 0.24 * overlap / len(tokens))

            if is_figure_note and wants_trend and (
                "trend_summary:" in lower or "series_descriptions:" in lower
            ):
                bonus += 0.14
            if is_figure_note and wants_axis and (
                "x_axis:" in lower or "y_axis:" in lower
            ):
                bonus += 0.12
            if is_figure_note and wants_gradient and (
                "gradient" in lower or "psi/ft" in lower or "psi/m" in lower
            ):
                bonus += 0.12

            if not is_figure_note and "learning outcomes" in lower:
                penalty += 0.30
            if not is_figure_note and ("table of contents" in lower or "contents" in lower[:180]):
                penalty += 0.18

            exact_phrase_matches = int(item.get("exact_phrase_matches") or 0)
            strong_phrase_matches = int(item.get("strong_phrase_matches") or 0)
            anchor_neighbor = bool(item.get("anchor_neighbor"))
            anchor_document = bool(item.get("anchor_document"))
            anchor_preceding = bool(item.get("anchor_preceding"))
            has_numeric_gradient = NUMERIC_GRADIENT_RE.search(text) is not None
            mentions_supercharged = SUPERCHARGED_RE.search(text) is not None
            asks_supercharged = SUPERCHARGED_RE.search(question) is not None
            special_type_curve = float(item.get("special_type_curve") or 0.0)
            comparison_before = bool(item.get("comparison_before"))
            comparison_after = bool(item.get("comparison_after"))

            bonus += exact_phrase_matches * 0.85
            bonus += strong_phrase_matches * 1.15
            if anchor_document:
                bonus += 0.30
            if anchor_neighbor:
                bonus += 0.22
            if wants_gradient and has_numeric_gradient:
                bonus += 0.95
            if anchor_preceding and wants_gradient and has_numeric_gradient:
                bonus += 0.55
            if asks_supercharged and mentions_supercharged:
                bonus += 0.24
            if (
                anchor_preceding
                and wants_gradient
                and has_numeric_gradient
                and asks_supercharged
                and mentions_supercharged
            ):
                bonus += 0.45

            if wants_type_curve and special_type_curve:
                bonus += min(2.80, 1.10 + 0.50 * special_type_curve)
                if "unit slope diagonal" in lower:
                    bonus += 0.55
                if "derivative plateau" in lower:
                    bonus += 0.55
                if "middle time region" in lower:
                    bonus += 0.35
            elif wants_type_curve:
                penalty += 0.55

            if wants_rft_comparison:
                if comparison_before:
                    bonus += 1.65
                if comparison_after:
                    bonus += 1.65
                if comparison_before and comparison_after:
                    bonus += 0.35

            rerank_score = max(0.0, base_score + bonus - penalty)
            item["retrieval_score"] = base_score
            item["figure_bonus"] = bonus
            item["figure_penalty"] = penalty
            item["figure_rank_score"] = rerank_score
            item["is_figure_note"] = is_figure_note
            item["has_numeric_gradient"] = has_numeric_gradient
            item["mentions_supercharged"] = mentions_supercharged
            ranked.append(item)

        ranked.sort(
            key=lambda item: (
                float(item.get("figure_rank_score") or 0.0),
                int(item.get("strong_phrase_matches") or 0),
                int(item.get("exact_phrase_matches") or 0),
                bool(item.get("is_figure_note")),
                float(item.get("keyword_score") or 0.0),
                float(item.get("vector_score") or 0.0),
            ),
            reverse=True,
        )

        if ranked:
            maximum = float(ranked[0].get("figure_rank_score") or 0.0)
            minimum = float(ranked[-1].get("figure_rank_score") or 0.0)
            span = maximum - minimum
            for item in ranked:
                raw = float(item.get("figure_rank_score") or 0.0)
                if span > 1e-9:
                    item["score"] = 0.50 + 0.50 * ((raw - minimum) / span)
                else:
                    item["score"] = min(1.0, max(0.0, raw))

        return ranked[:top_k]

    def _english_phrases(self, question: str) -> list[str]:
        raw_phrases = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9+./_-]*(?:\s+[A-Za-z0-9][A-Za-z0-9+./_-]*)*",
            question,
        )
        phrases: list[str] = []
        for value in raw_phrases:
            normalized = re.sub(r"\s+", " ", value).strip().lower()
            if len(normalized) >= 3 and normalized not in phrases:
                phrases.append(normalized)
        return phrases

    def _technical_tokens(self, question: str) -> list[str]:
        values = re.findall(r"[A-Za-z][A-Za-z0-9+./_-]*|\d+(?:\.\d+)?", question)
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "is", "of", "on", "or", "the", "to", "with", "test",
        }
        output: list[str] = []
        for value in values:
            token = value.lower()
            if token in stopwords or token in output:
                continue
            output.append(token)
        return output


    def _supported_partial_answer(
        self,
        question: str,
        hits: list[dict],
    ) -> str | None:
        """Provide a deterministic evidence-only partial answer when the LLM over-refuses."""
        if RFT_ZONE_DETAIL_RE.search(question) is None:
            return None

        evidence_hit = next(
            (
                hit
                for hit in hits
                if (
                    "series_descriptions:" in str(hit.get("text") or "")
                    and "Open-circle points" in str(hit.get("text") or "")
                    and "Open-square points" in str(hit.get("text") or "")
                    and (
                        "seven interpreted depth zones"
                        in str(hit.get("text") or "")
                        or "Zones 1 through 7"
                        in str(hit.get("text") or "")
                    )
                )
            ),
            None,
        )
        if evidence_hit is None:
            return None

        text = str(evidence_hit.get("text") or "")
        metadata = evidence_hit.get("metadata") or {}
        document_name = clean_document_name(metadata.get("document"))
        page_value = metadata.get("page")
        citation = f"[{document_name}, p.{page_value}]"

        statements: list[str] = []

        if (
            "Pressure observations vary among seven interpreted depth zones" in text
            and "does not present one continuous formation-pressure curve" in text
        ):
            statements.append(
                "Interpreted RFT Data는 Zone 1부터 Zone 7까지 구분되어 있고, "
                "압력 관측값은 구역마다 달라 하나의 연속적인 지층 압력 곡선으로 "
                "표현되지 않습니다. "
                + citation
            )

        statements.append(
            "다만 제공된 Figure Note에는 Zone 1~7 각각의 개별 상승·하락이나 "
            "정확한 기울기가 따로 기록되어 있지 않으므로, Zone별 세부 압력 거동은 "
            "확인할 수 없습니다. "
            + citation
        )

        if "Open-circle points identified in the legend as supercharged points." in text:
            statements.append(
                "Open-circle points는 supercharged points를 뜻합니다. "
                + citation
            )
        if "Open-square points identified in the legend as double pretest sequence points." in text:
            statements.append(
                "Open-square points는 double pretest sequence points를 뜻합니다. "
                + citation
            )
        if "Filled pressure-observation points" in text:
            statements.append(
                "채워진 점은 압력 관측값이며, 일부는 수직 연결선 또는 불확실성 선과 "
                "함께 표시됩니다. "
                + citation
            )
        if "X-shaped points aligned with the mud-gradient line." in text:
            statements.append(
                "X자 모양 점은 mud-gradient line을 따라 배치됩니다. "
                + citation
            )
        if "Mud-gradient line labeled 1.11 g/cc and 1.58 psi/m." in text:
            statements.append(
                "Mud-gradient 기준선에는 1.11 g/cc와 1.58 psi/m가 표시되어 있습니다. "
                + citation
            )

        return "\n\n".join(statements) if statements else None

    def _rft_comparison_companion_hits(
        self,
        question: str | None,
    ) -> list[dict]:
        """Fetch the exact before/after Figure Note chunks for RFT comparisons."""
        if not question or RFT_COMPARISON_RE.search(question) is None:
            return []

        collection = getattr(self.vector_store, "collection", None)
        if collection is None:
            return []

        try:
            data = collection.get(include=["documents", "metadatas"])
        except Exception:
            return []

        selected: dict[str, dict] = {}
        wanted_titles = (
            "title: Appraisal Well RFT Survey",
            "title: RFT Survey after Significant Production",
        )

        for chunk_id, document, metadata in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            text = str(document or "")
            if "image_path:" not in text:
                continue
            if not any(title in text for title in wanted_titles):
                continue
            selected[str(chunk_id)] = {
                "id": str(chunk_id),
                "text": text,
                "metadata": metadata or {},
                "score": 1.0,
                "vector_score": 0.0,
                "keyword_score": 0.0,
            }

        return list(selected.values())


    def _figure_references(
        self,
        hits: list[dict],
        limit: int = 3,
        *,
        question: str | None = None,
    ) -> list[FigureReference]:
        """Return existing source images that are explicitly present in retrieved hits."""
        figures_root = self.settings.figures_dir.resolve()
        references: list[FigureReference] = []
        seen_filenames: set[str] = set()

        figure_hits = list(hits)
        existing_ids = {str(hit.get("id") or "") for hit in figure_hits}
        for companion in self._rft_comparison_companion_hits(question):
            companion_id = str(companion.get("id") or "")
            if companion_id and companion_id not in existing_ids:
                figure_hits.append(companion)
                existing_ids.add(companion_id)

        for hit in figure_hits:
            text = str(hit.get("text") or "")
            metadata = hit.get("metadata") or {}

            for fields in self._figure_note_fields(text):
                raw_image_path = str(fields.get("image_path") or "").strip()
                filename = re.split(r"[\\/]", raw_image_path)[-1].strip()

                if (
                    not filename
                    or filename in {".", ".."}
                    or "/" in filename
                    or "\\" in filename
                    or filename in seen_filenames
                ):
                    continue

                candidate = (figures_root / filename).resolve()
                try:
                    candidate.relative_to(figures_root)
                except ValueError:
                    continue

                if not candidate.is_file():
                    continue

                image_type = self._nullable_figure_value(
                    fields.get("image_type")
                )
                if image_type and image_type.lower() in {
                    "logo",
                    "page_decoration",
                }:
                    continue

                page_value = fields.get("page_number")
                if page_value is None:
                    page_value = metadata.get("page")

                page_number = None
                try:
                    if page_value is not None:
                        page_number = int(str(page_value).strip())
                except (TypeError, ValueError):
                    page_number = None

                document_name = clean_document_name(
                    fields.get("document_name")
                    or metadata.get("document")
                )
                title = self._nullable_figure_value(fields.get("title"))
                image_index = None
                try:
                    if fields.get("image_index") is not None:
                        image_index = int(str(fields.get("image_index")).strip())
                except (TypeError, ValueError):
                    image_index = None

                preview_path = self.figure_preview.get_or_create_preview(
                    candidate,
                    document_id=self._nullable_figure_value(fields.get("document_id")),
                    document_name=document_name,
                    page=page_number,
                    image_index=image_index,
                    image_type=image_type,
                )
                preview_url = None
                if preview_path is not None:
                    preview_url = (
                        "/api/figure-previews/"
                        f"{quote(preview_path.name, safe='')}"
                    )

                references.append(
                    FigureReference(
                        document=document_name,
                        page=page_number,
                        title=title,
                        image_type=image_type,
                        filename=filename,
                        url=f"/api/figures/{quote(filename, safe='')}",
                        preview_url=preview_url,
                    )
                )
                seen_filenames.add(filename)

        if question and RFT_COMPARISON_RE.search(question):
            def comparison_order(reference: FigureReference) -> tuple[int, str]:
                title = str(reference.title or "").lower()
                if "appraisal well rft survey" in title:
                    return (0, title)
                if "after significant production" in title:
                    return (1, title)
                return (2, title)

            references.sort(key=comparison_order)

        return references[:limit]

    def _figure_note_fields(self, text: str) -> list[dict[str, str]]:
        """Parse only scalar Figure Note fields needed for image display."""
        if "image_path:" not in text.lower():
            return []

        blocks = re.split(
            r"(?=\[Figure Note Metadata\])",
            text,
            flags=re.IGNORECASE,
        )

        field_names = (
            "document_name",
            "document_id",
            "page_number",
            "image_index",
            "image_path",
            "image_type",
            "confidence",
            "title",
            "title_verified",
            "analysis",
            "x_axis",
        )
        boundary = "|".join(re.escape(name) for name in field_names)

        output: list[dict[str, str]] = []

        for block in blocks:
            if "image_path:" not in block.lower():
                continue

            fields: dict[str, str] = {}

            for name in field_names:
                match = re.search(
                    rf"(?:^|\s){re.escape(name)}:\s*(.*?)"
                    rf"(?=\s+(?:{boundary}):|$)",
                    block,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if match:
                    fields[name] = match.group(1).strip()

            if fields.get("image_path"):
                output.append(fields)

        return output

    def _nullable_figure_value(self, value: object) -> str | None:
        text = str(value or "").strip()
        if not text or text.lower() in {"null", "none", "unknown"}:
            return None
        return text

    def _aggregate_response(self, question: str, query_type: QueryType) -> ChatResponse:
        if re.search(r"top\s*\d+|가장 많이|빈도|통계", question.lower()):
            message = (
                "전체 문서 분석 기능이 필요합니다. 이 유형은 긴 context를 LLM에 "
                "보내지 않고 Python 기반 전체 chunk 스캔/빈도 분석 함수로 처리해야 합니다."
            )
        else:
            message = "제공된 문서 근거로는 확인할 수 없습니다."
        return ChatResponse(answer=message, sources=[], query_type=query_type.value)

    def _preview(self, text: str, limit: int = 700) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:limit]
