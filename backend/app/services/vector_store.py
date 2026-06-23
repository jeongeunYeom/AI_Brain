import math
import re
from typing import Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import Settings


class VectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=str(settings.vector_db_dir)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            device=self.embedding_device,
        )

        print(
            f"[임베딩 모델] {settings.embedding_model} / "
            f"device={self.embedding_device}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.embedding_model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) >= 100,
        )
        return vectors.astype("float32").tolist()

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return

        # 큰 문서를 한꺼번에 임베딩·저장하지 않고 나누어 처리합니다.
        ingest_batch_size = 256
        total = len(chunks)

        for start in range(0, total, ingest_batch_size):
            end = min(start + ingest_batch_size, total)
            batch = chunks[start:end]

            ids = [str(chunk["id"]) for chunk in batch]
            documents = [str(chunk["text"]) for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            embeddings = self.embed(documents)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            print(f"[임베딩·ChromaDB 저장] {end}/{total}")

    def search(
        self,
        question: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embed([question])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return self._query_result_to_hits(result)

    def hybrid_search(
        self,
        question: str,
        top_k: int,
        score_threshold: float,
        prefer_metadata: dict[str, Any] | None = None,
        keyword_weight: float = 0.45,
    ) -> list[dict[str, Any]]:
        vector_hits = self.search(
            question,
            max(top_k * 3, top_k),
        )
        keyword_hits = self.keyword_search(
            question,
            max(top_k * 3, top_k),
            prefer_metadata=prefer_metadata,
        )

        merged: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(vector_hits):
            vector_score = max(
                0.0,
                1.0 - float(hit.get("distance") or 1.0),
            )
            hit["vector_score"] = vector_score
            hit["keyword_score"] = 0.0
            hit["score"] = vector_score
            hit["rank_bonus"] = 1 / (rank + 1)
            merged[hit["id"]] = hit

        for rank, hit in enumerate(keyword_hits):
            existing = merged.get(hit["id"])
            keyword_score = float(
                hit.get("keyword_score") or 0.0
            )

            if existing:
                existing["keyword_score"] = max(
                    float(existing.get("keyword_score") or 0.0),
                    keyword_score,
                )
                existing["rank_bonus"] = max(
                    float(existing.get("rank_bonus") or 0.0),
                    1 / (rank + 1),
                )
            else:
                hit["vector_score"] = 0.0
                hit["score"] = keyword_score
                hit["rank_bonus"] = 1 / (rank + 1)
                merged[hit["id"]] = hit

        for hit in merged.values():
            vector_score = float(
                hit.get("vector_score") or 0.0
            )
            keyword_score = float(
                hit.get("keyword_score") or 0.0
            )
            metadata_boost = self._metadata_boost(
                hit.get("metadata") or {},
                prefer_metadata,
            )
            hit["score"] = (
                (1 - keyword_weight) * vector_score
                + keyword_weight * keyword_score
                + metadata_boost
            )

        hits = [
            hit
            for hit in merged.values()
            if float(hit.get("score") or 0.0)
            >= score_threshold
        ]

        return sorted(
            hits,
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )[:top_k]

    def keyword_search(
        self,
        question: str,
        top_k: int,
        prefer_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        data = self.collection.get(
            include=["documents", "metadatas"]
        )
        tokens = self._keyword_tokens(question)

        if not tokens:
            return []

        hits: list[dict[str, Any]] = []

        for chunk_id, document, metadata in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            text = str(document or "")
            lower = text.lower()
            exact_matches = sum(
                1
                for token in tokens
                if token.lower() in lower
            )

            if exact_matches == 0:
                continue

            token_score = exact_matches / max(len(tokens), 1)
            density = min(
                1.0,
                exact_matches
                / max(math.log(len(text) + 10), 1),
            )
            keyword_score = min(
                1.0,
                0.75 * token_score
                + 0.25 * density
                + self._metadata_boost(
                    metadata or {},
                    prefer_metadata,
                ),
            )

            hits.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                    "distance": None,
                    "keyword_score": keyword_score,
                }
            )

        return sorted(
            hits,
            key=lambda item: float(
                item.get("keyword_score") or 0.0
            ),
            reverse=True,
        )[:top_k]

    def count(self) -> int:
        return self.collection.count()

    def _query_result_to_hits(
        self,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for index, chunk_id in enumerate(ids):
            hits.append(
                {
                    "id": chunk_id,
                    "text": documents[index],
                    "metadata": metadatas[index],
                    "distance": (
                        distances[index]
                        if index < len(distances)
                        else None
                    ),
                }
            )

        return hits

    def _keyword_tokens(self, question: str) -> list[str]:
        tokens = re.findall(
            r"[A-Za-z][A-Za-z0-9_./-]*|[가-힣]{2,}|"
            r"\d+(?:\.\d+)?",
            question,
        )
        stopwords = {
            "this",
            "that",
            "from",
            "with",
            "문서",
            "에서",
            "무엇",
            "설명",
            "정리",
            "찾아줘",
            "모두",
            "의미",
        }
        return [
            token
            for token in tokens
            if token.lower() not in stopwords
        ]

    def _metadata_boost(
        self,
        metadata: dict[str, Any],
        prefer_metadata: dict[str, Any] | None,
    ) -> float:
        if not prefer_metadata:
            return 0.0

        boost = 0.0

        for key, expected in prefer_metadata.items():
            if metadata.get(key) == expected:
                boost += 0.2

        return min(boost, 0.4)
