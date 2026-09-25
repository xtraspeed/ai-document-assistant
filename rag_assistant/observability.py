"""Lightweight privacy-conscious request metrics for the Streamlit UI."""

from __future__ import annotations

from typing import Any, Iterable


def record_from_result(result: Any) -> dict[str, Any]:
    """Convert an AnswerResult into a JSON-friendly metrics record."""

    return {
        "latency_ms": round(float(result.latency_ms), 2),
        "retrieved_chunks": len(result.sources),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "standalone_query": result.standalone_query,
    }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    if not values:
        return {
            "requests": 0,
            "average_latency_ms": 0.0,
            "average_retrieved_chunks": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def average(key: str) -> float:
        numeric = [float(item[key]) for item in values if item.get(key) is not None]
        return sum(numeric) / len(numeric) if numeric else 0.0

    def total(key: str) -> int:
        return sum(int(item.get(key) or 0) for item in values)

    return {
        "requests": len(values),
        "average_latency_ms": round(average("latency_ms"), 2),
        "average_retrieved_chunks": round(average("retrieved_chunks"), 2),
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
    }
