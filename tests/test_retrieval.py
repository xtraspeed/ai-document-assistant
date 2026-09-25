import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_assistant.retrieval import HybridRetriever


class FakeEmbeddings(Embeddings):
    """Small deterministic embedder so tests never call OpenAI."""

    dimensions = 32

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vector[hash(token) % self.dimensions] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _documents() -> list[Document]:
    return [
        Document(
            page_content="Reciprocal-rank fusion combines dense and BM25 rankings.",
            metadata={"source": "retrieval.md", "page": 1, "chunk_index": 0},
        ),
        Document(
            page_content="The assistant should cite supporting source passages.",
            metadata={"source": "answering.md", "page": 2, "chunk_index": 0},
        ),
        Document(
            page_content="A public deployment should protect its API key.",
            metadata={"source": "security.md", "page": 1, "chunk_index": 0},
        ),
    ]


def test_hybrid_search_ranks_relevant_chunk_first():
    retriever = HybridRetriever(_documents(), FakeEmbeddings(), top_k=2)

    results = retriever.search("How does reciprocal-rank fusion work?", top_k=2)

    assert results
    assert results[0].source == "retrieval.md"
    assert results[0].vector_rank is not None
    assert results[0].bm25_rank is not None


def test_keyword_fallback_works_without_embeddings():
    retriever = HybridRetriever(_documents(), embeddings=None, top_k=2)

    results = retriever.search("API key", top_k=2)

    assert results
    assert results[0].source == "security.md"
    assert retriever.vector_store is None
    assert retriever.vector_error is None


def test_empty_query_returns_no_results():
    retriever = HybridRetriever(_documents(), FakeEmbeddings())

    assert retriever.search("   ") == []
