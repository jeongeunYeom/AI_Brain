from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.config import Settings

if TYPE_CHECKING:
    from app.services.qa import QAService


class KnowledgeTools:
    """Read-only Agent tools backed by the existing Text/Figure RAG pipeline."""

    def __init__(
        self,
        settings: Settings,
        qa_service: QAService | None = None,
    ):
        self.settings = settings
        self._qa_service = qa_service

    def search_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        evidence = self._retrieve(query, top_k)
        return {
            **evidence,
            "source_count": len(evidence["sources"]),
            "figure_count": len(evidence["figures"]),
        }

    def get_related_figures(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        evidence = self._retrieve(query, top_k)
        return {
            **evidence,
            "source_count": len(evidence["sources"]),
            "figure_count": len(evidence["figures"]),
        }

    def _retrieve(self, query: str, top_k: int) -> dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("Knowledge search requires a query.")

        try:
            normalized_top_k = int(top_k)
        except (TypeError, ValueError) as exc:
            raise ValueError("top_k must be an integer.") from exc
        if not 1 <= normalized_top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        return self._get_qa_service().retrieve_evidence(
            normalized_query,
            normalized_top_k,
        )

    def _get_qa_service(self) -> QAService:
        if self._qa_service is None:
            # Keep Agent startup lightweight. Heavy RAG dependencies are loaded
            # only when a knowledge tool is actually executed.
            from app.services.ollama import OllamaClient
            from app.services.qa import QAService
            from app.services.vector_store import VectorStore

            self._qa_service = QAService(
                self.settings,
                VectorStore(self.settings),
                OllamaClient(self.settings),
            )
        return self._qa_service
