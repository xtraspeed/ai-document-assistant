from langchain_core.documents import Document

from rag_assistant.evaluation import EvaluationCase, citation_metrics, evaluate_retrieval
from rag_assistant.retrieval import RetrievedChunk


class FakeRetriever:
    def search(self, query: str, *, top_k: int | None = None):
        limit = top_k or 1
        return [
            RetrievedChunk(
                document=Document(
                    page_content="The expected evidence is present.",
                    metadata={"source": "expected.md", "page": 1},
                ),
                score=0.1,
            ),
            RetrievedChunk(
                document=Document(
                    page_content="Unrelated material.",
                    metadata={"source": "other.md", "page": 1},
                ),
                score=0.05,
            ),
        ][:limit]


def test_retrieval_evaluation_reports_hit_and_mrr():
    cases = [
        EvaluationCase("one", "question", ("expected evidence",)),
        EvaluationCase("two", "missing", ("not present",)),
    ]

    report = evaluate_retrieval(FakeRetriever(), cases, top_k=2)

    assert report.hit_rate == 0.5
    assert report.mean_reciprocal_rank == 0.5
    assert report.cases[0].first_relevant_rank == 1
    assert report.as_dict()["case_count"] == 2


def test_citation_metrics_rejects_unknown_labels():
    sources = [{"source": "a.md"}, {"source": "b.md"}]

    metrics = citation_metrics("Answer [S1] and [S9].", sources)

    assert metrics["citation_count"] == 2
    assert metrics["valid_citation_count"] == 1
    assert metrics["citation_precision"] == 0.5
