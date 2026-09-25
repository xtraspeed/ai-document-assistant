"""Hybrid dense + lexical retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _document_key(document: Document) -> tuple[str, str, str, str]:
    metadata = document.metadata
    return (
        str(metadata.get("source", "")),
        str(metadata.get("page", "")),
        str(metadata.get("chunk_index", "")),
        document.page_content[:160],
    )


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    bm25_score: float = 0.0

    @property
    def source(self) -> str:
        return str(self.document.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | str | None:
        return self.document.metadata.get("page")

    def as_dict(self, *, preview_chars: int = 1_200) -> dict[str, Any]:
        page = self.page
        location = f"page {page}" if page not in (None, "") else "section unknown"
        return {
            "source": self.source,
            "page": page,
            "location": location,
            "score": round(float(self.score), 6),
            "preview": self.document.page_content[:preview_chars],
        }


class _DenseIndex:
    """Small FAISS adapter that keeps the vector store implementation explicit."""

    def __init__(self, documents: list[Document], embeddings: Embeddings) -> None:
        self.documents = documents
        self.embeddings = embeddings
        vectors = embeddings.embed_documents([document.page_content for document in documents])
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(documents):
            raise ValueError("embedding provider returned an invalid document matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding provider returned non-finite values")
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def similarity_search_with_score(
        self, query: str, *, k: int
    ) -> list[tuple[Document, float]]:
        query_vector = np.asarray([self.embeddings.embed_query(query)], dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("embedding provider returned an invalid query vector")
        if not np.isfinite(query_vector).all():
            raise ValueError("embedding provider returned non-finite query values")
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, min(k, len(self.documents)))
        results: list[tuple[Document, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                results.append((self.documents[int(index)], float(score)))
        return results


class HybridRetriever:
    """Retrieve chunks using FAISS and BM25, fused with weighted RRF.

    If embeddings are unavailable, the retriever degrades to BM25 instead of
    making the upload flow unusable. The caller can inspect ``vector_error`` and
    explain the degraded mode to the user.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        embeddings: Embeddings | None = None,
        *,
        top_k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
        strict_vector: bool = False,
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("at least one document is required to build an index")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if vector_weight < 0 or bm25_weight < 0 or vector_weight + bm25_weight <= 0:
            raise ValueError("retrieval weights must be non-negative and not both zero")

        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.vector_error: str | None = None
        self.vector_store: _DenseIndex | None = None

        corpus = [_tokenize(document.page_content) or ["<empty>"] for document in self.documents]
        self._bm25 = BM25Okapi(corpus)

        if embeddings is not None:
            try:
                self.vector_store = _DenseIndex(self.documents, embeddings)
            except Exception as exc:
                self.vector_error = f"{type(exc).__name__}: {exc}"
                if strict_vector:
                    raise

    @property
    def chunk_count(self) -> int:
        return len(self.documents)

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        limit = min(self.top_k if top_k is None else top_k, len(self.documents))
        candidate_count = min(len(self.documents), max(limit * 4, limit))

        vector_results: list[tuple[Document, float]] = []
        if self.vector_store is not None:
            try:
                vector_results = self.vector_store.similarity_search_with_score(
                    query, k=candidate_count
                )
            except Exception as exc:
                self.vector_error = f"{type(exc).__name__}: {exc}"

        vector_ranks: dict[tuple[str, str, str, str], int] = {}
        vector_documents: dict[tuple[str, str, str, str], Document] = {}
        for rank, (document, _distance) in enumerate(vector_results, start=1):
            key = _document_key(document)
            vector_ranks.setdefault(key, rank)
            vector_documents.setdefault(key, document)

        query_tokens = list(dict.fromkeys(_tokenize(query)))
        bm25_scores = self._bm25.get_scores(query_tokens)
        ranked_bm25 = sorted(
            (
                (index, float(score))
                for index, score in enumerate(bm25_scores)
                if float(score) > 0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        bm25_ranks = {
            _document_key(self.documents[index]): rank
            for rank, (index, _score) in enumerate(ranked_bm25, start=1)
        }
        bm25_score_by_key = {
            _document_key(self.documents[index]): score for index, score in ranked_bm25
        }

        candidates: dict[tuple[str, str, str, str], RetrievedChunk] = {}
        for document in self.documents:
            key = _document_key(document)
            vector_rank = vector_ranks.get(key)
            bm25_rank = bm25_ranks.get(key)
            if vector_rank is None and bm25_rank is None:
                continue

            score = 0.0
            if vector_rank is not None:
                score += self.vector_weight / (self.rrf_k + vector_rank)
            if bm25_rank is not None:
                score += self.bm25_weight / (self.rrf_k + bm25_rank)

            selected_document = vector_documents.get(key, document)
            candidates[key] = RetrievedChunk(
                document=selected_document,
                score=score,
                vector_rank=vector_rank,
                bm25_rank=bm25_rank,
                bm25_score=bm25_score_by_key.get(key, 0.0),
            )

        ranked = sorted(
            candidates.values(),
            key=lambda item: (-item.score, item.vector_rank or 10**9, item.bm25_rank or 10**9),
        )
        return ranked[:limit]
