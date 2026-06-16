from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import Settings


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedding_model = SentenceTransformer(settings.embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedding_model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [str(chunk["id"]) for chunk in chunks]
        documents = [str(chunk["text"]) for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = self.embed(documents)
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        query_embedding = self.embed([question])[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for index, chunk_id in enumerate(ids):
            hits.append({
                "id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index],
                "distance": distances[index] if index < len(distances) else None,
            })
        return hits
