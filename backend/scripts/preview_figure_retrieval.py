from app.core.config import get_settings
from app.services.qa import QAService
from app.services.query_router import classify_query
from app.services.vector_store import VectorStore


QUESTIONS = [
    "Log-Log Plot에서 Equivalent Time과 Delta P의 두 계열은 어떤 추세를 보이는가?",
    "Appraisal Well RFT Survey에서 표시된 압력 기울기와 supercharged test의 의미를 설명해줘.",
]


def main():
    settings = get_settings()
    service = QAService(settings, VectorStore(settings), None)

    for question in QUESTIONS:
        query_type = classify_query(question)
        search_question = service._build_search_question(question)
        hits = service._retrieve(
            search_question,
            query_type,
            10,
            original_question=question,
        )
        print("=" * 80)
        print(f"question={question}")
        print(f"search_question={search_question}")
        for rank, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            text = " ".join(str(hit.get("text") or "").split())
            print(
                f"rank={rank} id={hit.get('id')} page={metadata.get('page')} "
                f"score={float(hit.get('score') or 0):.3f} "
                f"figure={hit.get('is_figure_note')} "
                f"base={float(hit.get('retrieval_score') or 0):.3f} "
                f"rank_score={float(hit.get('figure_rank_score') or 0):.3f} "
                f"exact={int(hit.get('exact_phrase_matches') or 0)} "
                f"strong={int(hit.get('strong_phrase_matches') or 0)} "
                f"anchor_doc={bool(hit.get('anchor_document'))} "
                f"neighbor={bool(hit.get('anchor_neighbor'))} "
                f"preceding={bool(hit.get('anchor_preceding'))} "
                f"numeric_gradient={bool(hit.get('has_numeric_gradient'))}"
            )
            print(text[:500])
            print()

    print("NO_FILES_MODIFIED=True")


if __name__ == "__main__":
    main()
