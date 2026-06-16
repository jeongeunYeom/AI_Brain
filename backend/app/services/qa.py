from app.core.config import Settings
from app.models.schemas import ChatResponse, Source
from app.services.ollama import OllamaClient
from app.services.vector_store import VectorStore


SYSTEM_PROMPT = """You are a petroleum engineering AI agent.
Answer only from the supplied retrieved context. If the context does not contain enough evidence,
say that the knowledge base does not provide enough information. Include concise engineering reasoning,
units, assumptions, and citations in the form [document, p.page] when available. Do not invent facts.
"""


class QAService:
    def __init__(self, settings: Settings, vector_store: VectorStore, ollama: OllamaClient):
        self.settings = settings
        self.vector_store = vector_store
        self.ollama = ollama

    async def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        hits = self.vector_store.search(question, top_k or self.settings.top_k)
        if not hits:
            return ChatResponse(answer="지식베이스에서 관련 근거를 찾지 못했습니다. 문서를 먼저 업로드해 주세요.", sources=[])

        context_blocks = []
        sources = []
        for hit in hits:
            metadata = hit["metadata"]
            score = 1 - hit["distance"] if hit.get("distance") is not None else None
            context_blocks.append(
                f"Source: {metadata.get('document')} p.{metadata.get('page')} chunk {hit['id']}\n{hit['text']}"
            )
            sources.append(Source(
                document=str(metadata.get("document")),
                page=int(metadata["page"]) if metadata.get("page") is not None else None,
                chunk_id=str(hit["id"]),
                score=score,
                excerpt=str(hit["text"][:500]),
            ))

        user_prompt = f"Question:\n{question}\n\nRetrieved context:\n" + "\n\n---\n\n".join(context_blocks)
        answer = await self.ollama.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        return ChatResponse(answer=answer, sources=sources)
