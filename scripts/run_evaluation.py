"""Run the bundled retrieval benchmark against the sample knowledge base.

Usage:
    python scripts/run_evaluation.py

The script requires OPENAI_API_KEY because the production retriever uses OpenAI
embeddings. It prints a Markdown report that can be copied into the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repository importable when the script is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_openai import OpenAIEmbeddings  # noqa: E402

from rag_assistant.config import Settings  # noqa: E402
from rag_assistant.evaluation import evaluate_retrieval, load_evaluation_cases  # noqa: E402
from rag_assistant.ingestion import chunk_documents, load_file_payloads_from_paths  # noqa: E402
from rag_assistant.retrieval import HybridRetriever  # noqa: E402


def main() -> None:
    settings = Settings.from_environment()
    if not settings.has_api_key:
        raise SystemExit("Set OPENAI_API_KEY before running the evaluation.")

    ingestion = load_file_payloads_from_paths((ROOT / "data" / "samples").glob("*.md"))
    if ingestion.errors:
        print("Ingestion warnings:")
        for error in ingestion.errors:
            print(f"- {error}")
    chunks = chunk_documents(
        ingestion.documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_chunks=settings.max_chunks,
    )
    retriever = HybridRetriever(
        chunks,
        OpenAIEmbeddings(
            model=settings.embeddings_model,
            api_key=settings.openai_api_key,
        ),
        top_k=settings.top_k,
    )
    if retriever.vector_error:
        print(f"Warning: {retriever.vector_error}")

    cases = load_evaluation_cases(ROOT / "evaluation" / "cases.json")
    report = evaluate_retrieval(retriever, cases, top_k=settings.top_k)
    print(report.as_markdown())


if __name__ == "__main__":
    main()
