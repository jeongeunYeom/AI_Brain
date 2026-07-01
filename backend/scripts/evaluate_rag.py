from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings
from app.services.ollama import OllamaClient
from app.services.qa import QAService
from app.services.vector_store import VectorStore, expand_query_terms


DEFAULT_QUESTIONS = [
    "Archie 방정식을 제시하고, 식에 포함된 각 변수와 적용 조건을 설명해줘.",
    "업로드한 문서에서 ECD 계산식을 찾아 원래 기호 그대로 제시하고, 각 기호의 단위도 함께 적어줘.",
    "Mud weight를 psi/ft 또는 SG로 변환하는 관계식을 문서에서 찾아 설명해줘.",
    "Pore pressure와 fracture pressure 사이의 mud weight window를 설명하고, 범위를 벗어났을 때 kick 또는 lost circulation이 발생하는 이유를 설명해줘.",
    "Bottomhole pressure에 관한 내용을 찾은 모든 문서와 페이지를 표로 정리해줘.",
    "업로드한 문서의 압력 관련 그래프 하나를 찾아 x축, y축, 단위, 추세, 공학적 의미를 설명해줘.",
    "문서에 나오는 특정 유전의 2027년 생산량을 알려줘.",
]

REFUSAL_MARKERS = [
    "제공된 문서 근거로는 확인할 수 없습니다",
    "제공된 문서 근거로는 확인할 수 없어 답변할 수 없습니다",
    "제공된 문서에서 축, 단위 및 추세를 검증할 수 있는 그래프 근거를 찾지 못했습니다",
]


def extract_source_ids(answer: str) -> list[str]:
    ids: list[str] = []
    for match in re.finditer(r"\[S(\d+)\]", answer):
        source_id = f"S{match.group(1)}"
        if source_id not in ids:
            ids.append(source_id)
    return ids


def is_refusal(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer).strip()
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def source_to_dict(source: Any) -> dict[str, Any]:
    if hasattr(source, "model_dump"):
        return source.model_dump()
    return dict(source)


def check(ok: bool, reason: str = "") -> dict[str, Any]:
    return {"ok": ok, "reason": "" if ok else reason}


def has_all_terms(answer: str, terms: list[str]) -> bool:
    lower = answer.lower()
    return all(term.lower() in lower for term in terms)


def has_valid_archie_formula(answer: str) -> bool:
    compact = answer.lower()
    compact = compact.replace("\\left", "").replace("\\right", "")
    compact = re.sub(r"\s+", "", compact)
    compact = re.sub(r"\\frac\{fr_w\}\{r_t\}", "fr_w/r_t", compact)
    compact = re.sub(r"\\frac\{r_w\}\{r_t\}", "r_w/r_t", compact)
    compact = re.sub(r"[{}]", "", compact)
    wrong = bool(re.search(r"r_?w/?r_?t.*f\^?\(?1/?n\)?", compact))
    if wrong:
        return False
    raw = "s_w^n=\\fracfr_wr_t" in compact or "s_w^n=fr_w/r_t" in compact
    rearranged = (
        "s_w=(\\fracfr_wr_t)^{1/n}" in compact
        or "s_w=(fr_w/r_t)^(1/n)" in compact
        or "s_w=(fr_w/r_t)^1/n" in compact
    )
    loose = all(term in compact for term in ["s_w^n", "f", "r_w", "r_t"])
    return raw or rearranged or loose


def has_structured_graph_source(sources: list[dict[str, Any]]) -> bool:
    threshold = Settings().figure_note_min_confidence
    for source in sources:
        excerpt = str(source.get("excerpt") or "")
        lower = excerpt.lower()
        if "[figure note metadata]" not in lower:
            continue
        if not re.search(r"^image[_ ]type:\s*(graph|chart)\s*$", excerpt, re.IGNORECASE | re.MULTILINE):
            continue
        confidence_match = re.search(r"^confidence:\s*([0-9.]+)", excerpt, re.IGNORECASE | re.MULTILINE)
        if not confidence_match or float(confidence_match.group(1)) < threshold:
            continue
        path_match = re.search(r"^image[_ ]path:\s*(.+)$", excerpt, re.IGNORECASE | re.MULTILINE)
        if path_match and Path(path_match.group(1).strip()).exists():
            return True
    return False


def judge_result(
    *,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    debug: dict[str, Any],
    elapsed_seconds: float,
    error: str | None,
) -> dict[str, dict[str, Any]]:
    used_ids = extract_source_ids(answer)
    refusal = is_refusal(answer)
    lower_question = question.lower()
    lower_answer = answer.lower()
    final_ids = [f"S{index}" for index in range(1, len(sources) + 1)]

    checks: dict[str, dict[str, Any]] = {
        "citation_ids_match_sources": check(
            used_ids == final_ids[: len(used_ids)] and len(used_ids) <= len(sources),
            "answer citations are not continuous or do not match final source order",
        ),
        "refusal_has_no_sources": check((not refusal) or not sources, "refusal response returned sources"),
        "no_unknown_source_ids": check(
            all(source_id in final_ids for source_id in used_ids),
            "answer contains a source id that is not present in final sources",
        ),
        "answer_time_ok": check(elapsed_seconds < 180, "answer exceeded 180 seconds"),
        "no_error": check(error is None, str(error or "")),
    }

    if "archie" in lower_question:
        checks["archie_formula_scope"] = check(
            has_valid_archie_formula(answer) or refusal,
            "missing correct Archie equation or contains wrong F^(1/n) scope",
        )
        checks["archie_variables_present"] = check(
            refusal or has_all_terms(answer, ["S_w", "R_w", "R_t", "F", "n"]),
            "Archie variables Sw, Rw, Rt, F, n are not all present",
        )
        checks["archie_condition_present"] = check(
            refusal
            or any(
                term in lower_answer
                for term in ["clean", "nonshaly", "non-shaly", "shaly", "formation", "청결", "균일", "비셰일", "비석회"]
            ),
            "clean/nonshaly formation condition is not mentioned",
        )
        checks["no_unrelated_cyrillic"] = check(
            not re.search(r"[\u0400-\u04FF]", answer),
            "answer contains Cyrillic or unrelated mixed-language text",
        )

    if "ecd" in lower_question:
        checks["ecd_formula_terms"] = check(
            refusal or (has_all_terms(answer, ["ECD", "MW", "0.052"]) and ("d" in lower_answer or "tvd" in lower_answer)),
            "ECD answer lacks ECD, MW, 0.052, and depth/TVD terms",
        )
        checks["ecd_units_terms"] = check(
            refusal or (has_all_terms(answer, ["ppg", "psi", "ft"]) and "annular pressure" in lower_answer),
            "ECD answer lacks annular pressure loss and expected units",
        )

    if "mud weight" in lower_question and ("psi/ft" in lower_question or "sg" in lower_question):
        checks["mud_conversion_terms"] = check(
            refusal or has_all_terms(answer, ["0.052", "0.433", "psi/ft", "sg", "ppg"]),
            "mud weight conversion answer lacks required constants or units",
        )
        checks["mud_conversion_no_invented_r_symbols"] = check(
            not re.search(r"\bR_t\b|R_\\?\{?(?:\\?text\{)?(?:psi/ft|psi|ppg|sg|t)", answer, re.IGNORECASE),
            "answer invented R_t/R_psi/R_ppg style symbols",
        )

    if "mud weight window" in lower_question:
        checks["mud_window_direction"] = check(
            refusal
            or bool(
                re.search(r"(low|below|낮).*?(kick|influx)", lower_answer, re.DOTALL)
                and re.search(r"(high|above|높).*?(fracture|lost circulation)", lower_answer, re.DOTALL)
            ),
            "kick/lost circulation direction appears missing or reversed",
        )

    if any(term in lower_question for term in ["graph", "plot", "chart", "x축", "y축", "그래프", "그림"]):
        checks["graph_structured_figure_note"] = check(
            refusal or has_structured_graph_source(sources),
            "graph answer lacks a structured graph/chart Figure Note with existing image path",
        )

    if "2027" in lower_question:
        checks["future_production_not_guessed"] = check(
            refusal and not sources,
            "future production question should refuse with no sources unless exact evidence exists",
        )

    if debug.get("query_type") == "aggregate_analysis":
        checks["aggregate_not_false_refusal"] = check(
            bool(sources) or refusal,
            "aggregate answer has neither sources nor a clear refusal",
        )

    return checks


async def evaluate_questions(
    service: QAService,
    questions: list[str] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for index, question in enumerate(questions or DEFAULT_QUESTIONS, start=1):
        start = time.perf_counter()
        answer = ""
        sources: list[dict[str, Any]] = []
        debug: dict[str, Any] = {}
        error: str | None = None

        try:
            response = await service.answer(question)
            answer = response.answer
            sources = [source_to_dict(source) for source in response.sources]
            debug = dict(service.last_debug)
        except Exception as exc:  # noqa: BLE001 - evaluation must continue per question
            error = f"{type(exc).__name__}: {exc}"

        elapsed = round(time.perf_counter() - start, 3)
        used_source_ids = extract_source_ids(answer)
        result = {
            "index": index,
            "question": question,
            "query_type": debug.get("query_type"),
            "expanded_search_terms": expand_query_terms(str(debug.get("search_question") or question)),
            "search_question": debug.get("search_question"),
            "retrieved_chunk_count": debug.get("retrieved_count", 0),
            "final_source_count": len(sources),
            "sources": sources,
            "model_answer": answer,
            "raw_model_answer": debug.get("raw_model_answer", answer),
            "used_source_ids": used_source_ids,
            "is_refusal": is_refusal(answer),
            "elapsed_seconds": elapsed,
            "error": error,
            "total_scanned_chunks": debug.get("total_scanned_chunks"),
            "keyword_matched_chunks": debug.get("keyword_matched_chunks"),
            "deduplicated_chunks": debug.get("deduplicated_chunks"),
            "deduplicated_documents": debug.get("deduplicated_documents"),
            "deduplicated_pages": debug.get("deduplicated_pages"),
            "context_chunks_sent_to_llm": debug.get("context_chunks_sent_to_llm"),
            "context_characters": debug.get("context_characters"),
            "context_tokens_estimate": debug.get("context_tokens_estimate"),
            "dropped_by_document_limit": debug.get("dropped_by_document_limit"),
            "dropped_by_context_limit": debug.get("dropped_by_context_limit"),
            "exact_phrase_matches": debug.get("exact_phrase_matches"),
            "abbreviation_matches": debug.get("abbreviation_matches"),
            "weak_matches": debug.get("weak_matches"),
            "excluded_false_positive_count": debug.get("excluded_false_positive_count"),
            "excluded_false_positive_reasons": debug.get("excluded_false_positive_reasons"),
            "refusal_reason": debug.get("refusal_reason"),
            "citation_ids_before_filtering": debug.get("citation_ids_before_filtering", []),
            "citation_ids_after_filtering": debug.get("citation_ids_after_filtering", []),
        }
        result["checks"] = judge_result(
            question=question,
            answer=answer,
            sources=sources,
            debug=debug,
            elapsed_seconds=elapsed,
            error=error,
        )
        results.append(result)

    return results


def save_reports(results: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"rag_evaluation_{timestamp}.json"
    md_path = output_dir / f"rag_evaluation_{timestamp}.md"

    payload = {
        "created_at": datetime.now().isoformat(),
        "results": results,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# RAG Evaluation", ""]
    for result in results:
        lines.extend(
            [
                f"## {result['index']}. {result['question']}",
                "",
                f"- Query Type: `{result.get('query_type')}`",
                f"- Expanded Search Terms: {', '.join(result.get('expanded_search_terms') or [])}",
                f"- Retrieved Chunks: {result.get('retrieved_chunk_count')}",
                f"- Final Sources: {result.get('final_source_count')}",
                f"- Used Source IDs: {', '.join(result.get('used_source_ids') or [])}",
                f"- Refusal: {result.get('is_refusal')}",
                f"- Refusal Reason: {result.get('refusal_reason') or ''}",
                f"- Elapsed: {result.get('elapsed_seconds')}s",
                f"- Error: {result.get('error') or ''}",
                "",
                "### Debug",
                "",
            ]
        )
        for name in [
            "total_scanned_chunks",
            "keyword_matched_chunks",
            "deduplicated_chunks",
            "deduplicated_documents",
            "deduplicated_pages",
            "context_chunks_sent_to_llm",
            "context_characters",
            "context_tokens_estimate",
            "dropped_by_document_limit",
            "dropped_by_context_limit",
            "exact_phrase_matches",
            "abbreviation_matches",
            "weak_matches",
            "excluded_false_positive_count",
            "excluded_false_positive_reasons",
            "citation_ids_before_filtering",
            "citation_ids_after_filtering",
        ]:
            lines.append(f"- {name}: {result.get(name)}")
        lines.extend(["", "### Checks", ""])
        for name, detail in (result.get("checks") or {}).items():
            ok = bool(detail.get("ok")) if isinstance(detail, dict) else bool(detail)
            reason = detail.get("reason", "") if isinstance(detail, dict) else ""
            lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}` {reason}")
        lines.extend(["", "### Answer", "", result.get("model_answer") or "", "", "### Sources", ""])
        for source in result.get("sources") or []:
            lines.append(
                f"- {source.get('document')} p.{source.get('page')} "
                f"`{source.get('chunk_id')}` score={source.get('score')}"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    settings = Settings()
    output_dir = args.output_dir or settings.evaluation_dir
    service = QAService(settings, VectorStore(settings), OllamaClient(settings))
    results = await evaluate_questions(service)
    json_path, md_path = save_reports(results, output_dir)
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
