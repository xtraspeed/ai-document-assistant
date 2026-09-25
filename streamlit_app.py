"""Streamlit entrypoint for the AI Document Assistant."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from rag_assistant.assistant import AssistantError, RAGAssistant
from rag_assistant.config import Settings
from rag_assistant.ingestion import (
    FilePayload,
    chunk_documents,
    load_file_payloads,
    payload_from_upload,
)
from rag_assistant.observability import record_from_result, summarize
from rag_assistant.retrieval import HybridRetriever

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "data" / "samples"


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "metrics": [],
        "retriever": None,
        "ingestion_summary": None,
        "index_warning": None,
        "index_label": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["metrics"] = []


def _reset_index() -> None:
    st.session_state["retriever"] = None
    st.session_state["ingestion_summary"] = None
    st.session_state["index_warning"] = None
    st.session_state["index_label"] = None
    _reset_conversation()


def _friendly_vector_error(raw_error: str | None) -> str | None:
    if not raw_error:
        return None
    lowered = raw_error.lower()
    if "insufficient_quota" in lowered or "credit_balance_exhausted" in lowered:
        return (
            "OpenAI embeddings are unavailable because the account has no remaining credits. "
            "Add credits and rebuild; keyword retrieval remains available."
        )
    if "rate limit" in lowered or "429" in lowered:
        return (
            "OpenAI embeddings are temporarily rate-limited. Wait and rebuild; "
            "keyword retrieval remains available."
        )
    return (
        "Semantic retrieval is unavailable. Verify OpenAI access and rebuild; "
        "keyword retrieval remains available."
    )


def _build_index(payloads: list[FilePayload], settings: Settings, label: str) -> None:
    with st.spinner("Parsing, chunking, and indexing documents…"):
        ingestion = load_file_payloads(
            payloads,
            max_file_size_mb=settings.max_file_size_mb,
            max_files=settings.max_files,
        )
        if not ingestion.documents:
            message = "No readable documents were found."
            st.session_state["ingestion_summary"] = {
                "label": label,
                "files_processed": ingestion.files_processed,
                "pages_processed": ingestion.pages_processed,
                "characters": ingestion.characters,
                "errors": [message, *ingestion.errors],
            }
            st.session_state["retriever"] = None
            return

        chunks = chunk_documents(
            ingestion.documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_chunks=settings.max_chunks,
        )
        embeddings = None
        index_warning: str | None = None
        if settings.has_api_key:
            try:
                from langchain_openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings(
                    model=settings.embeddings_model,
                    api_key=settings.openai_api_key,
                )
            except Exception as exc:
                index_warning = f"Embeddings could not be initialized: {type(exc).__name__}."
        else:
            index_warning = (
                "No OpenAI key is configured; the index is keyword-only. "
                "Add a key and rebuild for semantic retrieval and answers."
            )

        retriever = HybridRetriever(chunks, embeddings, top_k=settings.top_k)
        if retriever.vector_error:
            index_warning = _friendly_vector_error(retriever.vector_error)

        st.session_state["retriever"] = retriever
        st.session_state["ingestion_summary"] = {
            "label": label,
            "files_processed": ingestion.files_processed,
            "pages_processed": ingestion.pages_processed,
            "characters": ingestion.characters,
            "chunk_count": len(chunks),
            "errors": ingestion.errors,
        }
        st.session_state["index_warning"] = index_warning
        st.session_state["index_label"] = label
        _reset_conversation()


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("Sources used for this answer"):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**S{index} — {source.get('source', 'unknown document')}** "
                f"({source.get('location', 'location unknown')})"
            )
            st.caption(f"Hybrid retrieval score: {source.get('score', 0):.4f}")
            st.text(source.get("preview", ""))


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message.get("role", "user")):
        st.markdown(message.get("content", ""))
        if message.get("role") == "assistant":
            _render_sources(message.get("sources", []))
            metrics = message.get("metrics", {})
            if metrics:
                st.caption(
                    f"{metrics.get('retrieved_chunks', 0)} passages · "
                    f"{metrics.get('latency_ms', 0):.0f} ms retrieval/generation"
                )


def _friendly_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return "OpenAI could not authenticate the request. Check the API key and try again."
    if "rate limit" in lowered or "quota" in lowered:
        return "The OpenAI request was rate-limited or exceeded the account quota. Try again later."
    if "could not initialize" in lowered:
        return "The OpenAI client could not be initialized. Check the model configuration."
    return "The assistant could not complete that request. Check the document index and try again."


def _handle_question(question: str, settings: Settings) -> None:
    retriever: HybridRetriever | None = st.session_state["retriever"]
    if retriever is None:
        st.warning("Build a knowledge base before asking a question.")
        return
    if not settings.has_api_key:
        st.error("Add an OpenAI API key in the sidebar before asking questions.")
        return

    history = list(st.session_state["messages"])
    try:
        with st.spinner("Searching the knowledge base and generating an answer…"):
            assistant = RAGAssistant(retriever, settings)
            result = assistant.answer(question, history=history)
    except AssistantError as exc:
        st.error(_friendly_error(exc))
        return
    except Exception as exc:
        logger.error("Unexpected assistant failure: %s", type(exc).__name__)
        st.error(_friendly_error(exc))
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["messages"].append(result.as_message())
    st.session_state["metrics"].append(record_from_result(result))
    st.rerun()


def _render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.markdown("### Configuration")
        configured_settings = Settings.from_environment()
        configured_key = configured_settings.openai_api_key
        typed_key = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="Enter locally or configure a Streamlit secret",
            key="api_key_input",
        )
        effective_key = typed_key.strip() or configured_key
        settings = Settings.from_environment(api_key=effective_key)
        if effective_key:
            st.success("OpenAI key is available for this session.")
        else:
            st.info("Add an OpenAI key to enable semantic retrieval and answers.")

        if configured_settings.app_access_code:
            entered_code = st.text_input(
                "Shared access code",
                type="password",
                key="access_code_input",
            )
            if entered_code != configured_settings.app_access_code:
                st.info("Enter the shared access code to open the assistant.")
                st.stop()
            st.success("Access code accepted.")

        st.caption(
            f"Chat model: `{settings.chat_model}`  ·  "
            f"Embeddings: `{settings.embeddings_model}`"
        )

        st.markdown("#### Knowledge base")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help=(
                f"Up to {settings.max_files} files, {settings.max_file_size_mb} MB each. "
                "Scanned PDFs need OCR before upload."
            ),
        )
        col_upload, col_sample = st.columns(2)
        with col_upload:
            build_clicked = st.button(
                "Build index",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_files,
            )
        with col_sample:
            sample_clicked = st.button(
                "Load samples",
                use_container_width=True,
            )

        if build_clicked and uploaded_files:
            _build_index(
                [payload_from_upload(uploaded_file) for uploaded_file in uploaded_files],
                settings,
                label="Uploaded documents",
            )
            st.rerun()

        if sample_clicked:
            sample_paths = list(SAMPLE_DIR.glob("*.md"))
            if not sample_paths:
                st.error("Sample documents are missing from the deployment.")
            else:
                _build_index(
                    [
                        FilePayload(name=path.name, data=path.read_bytes())
                        for path in sample_paths
                    ],
                    settings,
                    label="Bundled sample knowledge base",
                )
                st.rerun()

        retriever: HybridRetriever | None = st.session_state["retriever"]
        if retriever is not None:
            st.success(f"Indexed {retriever.chunk_count} chunks")
            summary = st.session_state.get("ingestion_summary") or {}
            if summary.get("files_processed"):
                st.caption(
                    f"{summary['files_processed']} files · "
                    f"{summary.get('pages_processed', 0)} pages · "
                    f"{summary.get('characters', 0):,} characters"
                )
            if st.session_state.get("index_warning"):
                st.warning(st.session_state["index_warning"])

        st.markdown("#### Session")
        if st.button("Clear conversation", use_container_width=True):
            _reset_conversation()
            st.rerun()
        if st.button("Clear knowledge base", use_container_width=True):
            _reset_index()
            st.rerun()

        metrics = summarize(st.session_state.get("metrics", []))
        with st.expander("Session metrics"):
            st.json(metrics)

    return settings


def _render_feedback() -> None:
    messages = st.session_state.get("messages", [])
    if not messages or messages[-1].get("role") != "assistant":
        return

    st.markdown("#### Was this answer useful?")
    helpful, not_helpful = st.columns(2)
    if helpful.button("👍 Helpful", use_container_width=True, key="feedback_helpful"):
        messages[-1]["feedback"] = "helpful"
        if st.session_state["metrics"]:
            st.session_state["metrics"][-1]["feedback"] = "helpful"
        st.rerun()
    if not_helpful.button(
        "👎 Needs improvement",
        use_container_width=True,
        key="feedback_not_helpful",
    ):
        messages[-1]["feedback"] = "needs_improvement"
        if st.session_state["metrics"]:
            st.session_state["metrics"][-1]["feedback"] = "needs_improvement"
        st.rerun()
    if messages[-1].get("feedback"):
        st.caption("Thanks — your feedback was recorded for this session.")


def _render_welcome() -> None:
    st.markdown(
        """
        Upload a few documents or load the bundled samples, build the index, and ask
        questions in natural language. Answers are constrained to retrieved passages
        and include source labels so you can inspect the evidence.
        """
    )
    st.markdown("#### Suggested questions")
    st.markdown(
        "- What are the main principles in the documents?\n"
        "- Compare the recommended retrieval strategies.\n"
        "- Give me a concise summary and cite the supporting pages."
    )


def main() -> None:
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1100px; padding-top: 2rem;}
        [data-testid="stMetric"] {
            background: rgba(37, 99, 235, 0.06);
            padding: 1rem;
            border-radius: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _initialize_state()
    initial_settings = Settings.from_environment()
    settings = _render_sidebar(initial_settings)

    st.title("📚 AI Document Assistant")
    st.caption("A grounded, citation-first RAG assistant powered by LangChain and OpenAI.")

    if st.session_state["retriever"] is None:
        _render_welcome()
        return

    for message in st.session_state["messages"]:
        _render_message(message)

    transcript = "\n\n".join(
        f"{message.get('role', 'user').title()}: {message.get('content', '')}"
        for message in st.session_state["messages"]
    )
    if transcript:
        st.download_button(
            "Download conversation",
            data=transcript,
            file_name="document-assistant-transcript.txt",
            mime="text/plain",
        )
        _render_feedback()

    question = st.chat_input("Ask a question about the indexed documents…")
    if question:
        _handle_question(question, settings)


if __name__ == "__main__":
    main()
