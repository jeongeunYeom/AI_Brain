import math
import os
import re
from typing import Any

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from app.core.config import Settings


TECHNICAL_TERM_EXPANSIONS: dict[str, list[str]] = {
    "archie": [
        "Archie's equation",
        "Archie law",
        "formation resistivity factor",
        "water saturation",
        "cementation exponent",
        "saturation exponent",
    ],
    "archie's equation": ["Archie", "Archie law", "formation resistivity factor"],
    "archie law": ["Archie", "Archie's equation", "formation resistivity factor"],
    "ecd": [
        "Equivalent Circulating Density",
        "circulating density",
        "annular pressure loss",
        "dynamic mud density",
    ],
    "equivalent circulating density": ["ECD", "circulating density", "annular pressure loss"],
    "bhp": [
        "bottomhole pressure",
        "bottom-hole pressure",
        "bottom hole pressure",
        "wellbore pressure",
    ],
    "bottomhole pressure": ["BHP", "bottom-hole pressure", "bottom hole pressure", "wellbore pressure"],
    "bottom-hole pressure": ["BHP", "bottomhole pressure", "bottom hole pressure", "wellbore pressure"],
    "bottom hole pressure": ["BHP", "bottomhole pressure", "bottom-hole pressure", "wellbore pressure"],
    "mud weight": ["ppg", "pounds per gallon", "SG", "specific gravity", "mud density"],
    "pore pressure": ["formation pressure", "kick", "mud weight window"],
    "fracture pressure": ["fracture gradient", "lost circulation", "mud weight window"],
    "kick": ["pore pressure", "formation pressure", "underbalanced"],
    "lost circulation": ["fracture pressure", "fracture gradient", "losses"],
    "sg": ["specific gravity", "mud density"],
    "specific gravity": ["SG", "mud density"],
    "ppg": ["pounds per gallon", "mud weight"],
    "pounds per gallon": ["ppg", "mud weight"],
    "psi/ft": ["pressure gradient", "mud weight", "specific gravity", "SG"],
    "formation resistivity factor": ["Archie", "Archie's equation", "resistivity"],
}

ENGLISH_STOPWORDS = {
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
    "x",
    "y",
}

KOREAN_STOPWORDS = {
    "각",
    "각 변수",
    "문서",
    "문서에서",
    "문서의",
    "문서만",
    "문서별로",
    "근거",
    "무엇",
    "설명",
    "설명해줘",
    "제시",
    "제시하고",
    "정리",
    "정리해줘",
    "찾아",
    "찾아줘",
    "모두",
    "전부",
    "의미",
    "적용 조건",
    "조건",
}

LOW_VALUE_MARKERS = (
    "nomenclature",
    "symbol list",
    "symbols",
    "index",
    "table of contents",
    "contents",
    "learning objectives",
    "references",
)

HIGH_VALUE_MARKERS = (
    "equation",
    "definition",
    "defined as",
    "is given by",
    "is expressed as",
    "archie",
    "bottomhole pressure",
    "bottom-hole pressure",
    "bottom hole pressure",
    "bhp",
    "mud weight",
    "pressure gradient",
    "formation resistivity factor",
)


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _matches_token(text: str, token: str) -> bool:
    normalized = token.lower()
    if re.fullmatch(r"[a-z0-9]+", normalized):
        return _contains_term(text, normalized)
    return normalized in text


def expand_query_terms(question: str) -> list[str]:
    normalized = re.sub(r"[’`]", "'", question).lower()
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            terms.append(value)

    for term, expansions in TECHNICAL_TERM_EXPANSIONS.items():
        if _contains_term(normalized, term):
            add(term)
            for expansion in expansions:
                add(expansion)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9_./-]*(?:'s)?|\d+(?:\.\d+)?", question):
        cleaned = re.sub(r"'s$", "", token, flags=re.IGNORECASE)
        if cleaned.lower() not in ENGLISH_STOPWORDS:
            add(cleaned)

    for token in re.findall(r"[가-힣]{2,}", question):
        if token not in KOREAN_STOPWORDS:
            add(token)

    return terms


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
        self.embedding_model: SentenceTransformer | None = None
        self.embedding_model_error: RuntimeError | None = None
        self.last_aggregate_debug: dict[str, Any] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self._get_embedding_model().encode(
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
            blended_score = (
                (1 - keyword_weight) * vector_score
                + keyword_weight * keyword_score
                + metadata_boost
            )
            hit["score"] = max(
                blended_score,
                min(1.0, keyword_score + metadata_boost),
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
            keyword_score = self._keyword_score(
                text,
                tokens,
                metadata or {},
                prefer_metadata,
            )
            if keyword_score <= 0:
                continue

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

    def aggregate_keyword_search(
        self,
        question: str,
        batch_size: int,
        max_results: int,
        max_per_document: int,
    ) -> list[dict[str, Any]]:
        tokens = self._keyword_tokens(question)
        self.last_aggregate_debug = {
            "total_scanned_chunks": 0,
            "keyword_matched_chunks": 0,
            "deduplicated_chunks": 0,
            "deduplicated_documents": 0,
            "deduplicated_pages": 0,
            "dropped_by_document_limit": 0,
        }
        if not tokens:
            return []

        merged: dict[tuple[str, int], dict[str, Any]] = {}
        offset = 0

        while True:
            batch = self.collection.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            ids = batch.get("ids", [])
            if not ids:
                break

            for chunk_id, document, metadata in zip(
                ids,
                batch.get("documents", []),
                batch.get("metadatas", []),
            ):
                self.last_aggregate_debug["total_scanned_chunks"] += 1
                text = str(document or "")
                meta = metadata or {}
                score = self._keyword_score(text, tokens, meta, None)
                if score <= 0:
                    continue
                self.last_aggregate_debug["keyword_matched_chunks"] += 1

                try:
                    page = int(meta.get("page"))
                except (TypeError, ValueError):
                    page = -1

                key = (
                    str(meta.get("document_id") or meta.get("document") or ""),
                    page,
                )
                current = merged.get(key)
                candidate = {
                    "id": str(chunk_id),
                    "text": text,
                    "metadata": meta,
                    "distance": None,
                    "vector_score": 0.0,
                    "keyword_score": score,
                    "score": score,
                }

                if current is None or score > float(current.get("score") or 0.0):
                    merged[key] = candidate
                elif current is not None and len(str(current.get("text") or "")) < 1800:
                    current["text"] = (
                        str(current.get("text") or "").rstrip()
                        + "\n\n"
                        + text[:800].strip()
                    )

            offset += batch_size

        ordered = sorted(
            merged.values(),
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )

        per_document_count: dict[str, int] = {}
        results: list[dict[str, Any]] = []
        for hit in ordered:
            metadata = hit.get("metadata") or {}
            document_key = str(metadata.get("document_id") or metadata.get("document") or "")
            count = per_document_count.get(document_key, 0)
            if count >= max_per_document:
                self.last_aggregate_debug["dropped_by_document_limit"] += 1
                continue
            per_document_count[document_key] = count + 1
            results.append(hit)
            if len(results) >= max_results:
                break

        self.last_aggregate_debug.update(
            {
                "deduplicated_chunks": len(ordered),
                "deduplicated_documents": len(
                    {
                        str((hit.get("metadata") or {}).get("document_id") or (hit.get("metadata") or {}).get("document") or "")
                        for hit in ordered
                    }
                ),
                "deduplicated_pages": len(
                    {
                        (
                            str((hit.get("metadata") or {}).get("document_id") or (hit.get("metadata") or {}).get("document") or ""),
                            (hit.get("metadata") or {}).get("page"),
                        )
                        for hit in ordered
                    }
                ),
            }
        )
        return results

    def expand_with_neighbors(
        self,
        hits: list[dict[str, Any]],
        max_total: int,
    ) -> list[dict[str, Any]]:
        if not hits or len(hits) >= max_total:
            return hits[:max_total]

        data = self.collection.get(include=["documents", "metadatas"])
        by_id: dict[str, dict[str, Any]] = {}
        by_doc_page_chunk: dict[tuple[str, int, int], str] = {}
        first_chunk_by_doc_page: dict[tuple[str, int], str] = {}

        for chunk_id, document, metadata in zip(
            data.get("ids", []),
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            meta = metadata or {}
            doc_id = str(meta.get("document_id") or meta.get("document") or "")
            try:
                page = int(meta.get("page"))
                chunk_index = int(meta.get("chunk_index", 0))
            except (TypeError, ValueError):
                continue

            by_id[str(chunk_id)] = {
                "id": str(chunk_id),
                "text": str(document or ""),
                "metadata": meta,
                "distance": None,
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "score": 0.0,
                "neighbor": True,
            }
            by_doc_page_chunk[(doc_id, page, chunk_index)] = str(chunk_id)
            first_chunk_by_doc_page.setdefault((doc_id, page), str(chunk_id))

        merged = {str(hit["id"]): hit for hit in hits}

        for hit in hits:
            metadata = hit.get("metadata") or {}
            doc_id = str(metadata.get("document_id") or metadata.get("document") or "")
            try:
                page = int(metadata.get("page"))
                chunk_index = int(metadata.get("chunk_index", 0))
            except (TypeError, ValueError):
                continue

            candidate_ids = [
                by_doc_page_chunk.get((doc_id, page, chunk_index - 1)),
                by_doc_page_chunk.get((doc_id, page, chunk_index + 1)),
                by_doc_page_chunk.get((doc_id, page - 1, chunk_index)),
                first_chunk_by_doc_page.get((doc_id, page - 1)),
                by_doc_page_chunk.get((doc_id, page + 1, chunk_index)),
                first_chunk_by_doc_page.get((doc_id, page + 1)),
            ]

            for candidate_id in candidate_ids:
                if not candidate_id or candidate_id in merged:
                    continue
                neighbor = dict(by_id[candidate_id])
                seed_score = float(hit.get("score") or 0.0)
                neighbor["score"] = min(seed_score * 0.6, 0.45)
                merged[candidate_id] = neighbor
                if len(merged) >= max_total:
                    return list(merged.values())[:max_total]

        return list(merged.values())[:max_total]

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
        return expand_query_terms(question)

    def _keyword_score(
        self,
        text: str,
        tokens: list[str],
        metadata: dict[str, Any],
        prefer_metadata: dict[str, Any] | None,
    ) -> float:
        lower = text.lower()
        exact_matches = sum(1 for token in tokens if _matches_token(lower, token))

        if exact_matches == 0:
            return 0.0

        phrase_match = any(" " in token and _matches_token(lower, token) for token in tokens)
        token_score = min(1.0, exact_matches / max(min(len(tokens), 3), 1))
        density = min(
            1.0,
            exact_matches
            / max(math.log(len(text) + 10), 1),
        )
        exact_floor = 0.8 if phrase_match else 0.65
        score = min(
            1.0,
            max(exact_floor, 0.75 * token_score + 0.25 * density)
            + self._metadata_boost(
                metadata or {},
                prefer_metadata,
            ),
        )
        if any(marker in lower for marker in HIGH_VALUE_MARKERS):
            score = min(1.0, score + 0.12)
        if any(marker in lower for marker in LOW_VALUE_MARKERS):
            score *= 0.35
        return score

    def _get_embedding_model(self) -> SentenceTransformer:
        if self.embedding_model is not None:
            return self.embedding_model
        if self.embedding_model_error is not None:
            raise self.embedding_model_error

        model_name = self.settings.embedding_model
        if self.settings.embedding_model_path:
            model_path = self.settings.embedding_model_path
            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"Embedding model path does not exist: {model_path}. "
                    "Set EMBEDDING_MODEL_PATH to a local BGE-M3 directory or unset it to use EMBEDDING_MODEL."
                )
            model_name = model_path

        try:
            self.embedding_model = SentenceTransformer(
                model_name,
                device=self.embedding_device,
            )
        except Exception as exc:
            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("TRANSFORMERS_OFFLINE") == "1"
            hint = (
                " Local offline mode is enabled; set EMBEDDING_MODEL_PATH to a downloaded model directory."
                if offline
                else " Download the model once or set EMBEDDING_MODEL_PATH to a local model directory."
            )
            self.embedding_model_error = RuntimeError(
                f"Could not load embedding model `{model_name}`.{hint}"
            )
            raise self.embedding_model_error from exc

        print(
            f"[임베딩 모델] {model_name} / "
            f"device={self.embedding_device}"
        )
        return self.embedding_model

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
