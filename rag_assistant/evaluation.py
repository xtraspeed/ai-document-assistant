"""Small, reproducible retrieval and citation evaluation utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .retrieval import RetrievedChunk


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_terms: tuple[str, ...]


class RetrieverLike(Protocol):
    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]: ...


@dataclass
class CaseResult:
    case_id: str
    question: str
    hit: bool
    first_relevant_rank: int | None
    retrieved_sources: list[str]
    latency_ms: float


@dataclass
class EvaluationReport:
    cases: list[CaseResult]
    hit_rate: float
    mean_reciprocal_rank: float
    mean_latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_count": len(self.cases),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mean_reciprocal_rank, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "cases": [
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "hit": case.hit,
                    "first_relevant_rank": case.first_relevant_rank,
                    "retrieved_sources": case.retrieved_sources,
                    "latency_ms": round(case.latency_ms, 2),
                }
                for case in self.cases
            ],
        }

    def as_markdown(self) -> str:
        lines = [
            "# Retrieval evaluation",
            "",
            f"- Cases: **{len(self.cases)}**",
            f"- Hit rate: **{self.hit_rate:.1%}**",
            f"- Mean reciprocal rank: **{self.mean_reciprocal_rank:.3f}**",
            f"- Mean retrieval latency: **{self.mean_latency_ms:.1f} ms**",
            "",
            "| Case | Hit | First relevant rank | Sources |",
            "|---|---:|---:|---|",
        ]
        for case in self.cases:
            sources = ", ".join(case.retrieved_sources[:3]) or "—"
            rank = case.first_relevant_rank or "—"
            lines.append(f"| {case.case_id} | {'yes' if case.hit else 'no'} | {rank} | {sources} |")
        return "\n".join(lines)


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    if not isinstance(raw, list):
        raise ValueError("evaluation file must contain a JSON list of cases")

    cases: list[EvaluationCase] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        terms = item.get("expected_terms", item.get("expected_sources", []))
        if not question or not terms:
            continue
        cases.append(
            EvaluationCase(
                case_id=str(item.get("id", f"case-{index:02d}")),
                question=question,
                expected_terms=tuple(str(term) for term in terms),
            )
        )
    return cases


def _is_relevant(case: EvaluationCase, chunks: Iterable[RetrievedChunk]) -> bool:
    haystack = "\n".join(
        f"{chunk.source}\n{chunk.document.page_content}" for chunk in chunks
    ).lower()
    return any(term.lower() in haystack for term in case.expected_terms)


def evaluate_retrieval(
    retriever: RetrieverLike, cases: Iterable[EvaluationCase], *, top_k: int = 5
) -> EvaluationReport:
    import time

    results: list[CaseResult] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []

    for case in cases:
        started = time.perf_counter()
        chunks = retriever.search(case.question, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1_000
        first_relevant_rank: int | None = None
        for rank, chunk in enumerate(chunks, start=1):
            candidate = EvaluationCase(case.case_id, case.question, case.expected_terms)
            if _is_relevant(candidate, [chunk]):
                first_relevant_rank = rank
                break
        hit = first_relevant_rank is not None
        if first_relevant_rank is not None:
            reciprocal_ranks.append(1 / first_relevant_rank)
        latencies.append(latency_ms)
        results.append(
            CaseResult(
                case_id=case.case_id,
                question=case.question,
                hit=hit,
                first_relevant_rank=first_relevant_rank,
                retrieved_sources=[chunk.source for chunk in chunks],
                latency_ms=latency_ms,
            )
        )

    count = len(results)
    return EvaluationReport(
        cases=results,
        hit_rate=sum(case.hit for case in results) / count if count else 0.0,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count if count else 0.0,
        mean_latency_ms=sum(latencies) / count if count else 0.0,
    )


def citation_metrics(answer: str, sources: list[dict[str, Any]]) -> dict[str, float | int]:
    """Measure whether generated citations refer to retrieved source labels."""

    labels = re.findall(r"\[S(\d+)\]", answer)
    valid_labels = {str(index) for index in range(1, len(sources) + 1)}
    valid = [label for label in labels if label in valid_labels]
    unique_valid = set(valid)
    return {
        "citation_count": len(labels),
        "valid_citation_count": len(valid),
        "citation_precision": len(valid) / len(labels) if labels else 0.0,
        "source_coverage": len(unique_valid) / len(sources) if sources else 0.0,
    }
