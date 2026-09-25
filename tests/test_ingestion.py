from pathlib import Path

from langchain_core.documents import Document

from rag_assistant.ingestion import (
    FilePayload,
    chunk_documents,
    load_file_payloads,
    load_file_payloads_from_paths,
)


def test_loads_text_and_preserves_source_metadata():
    result = load_file_payloads(
        [FilePayload(name="notes.md", data=b"# Notes\n\nRetrieval uses citations.")]
    )

    assert result.files_processed == 1
    assert result.pages_processed == 1
    assert not result.errors
    assert result.documents[0].metadata["source"] == "notes.md"
    assert result.documents[0].metadata["file_type"] == "md"
    assert "citations" in result.documents[0].page_content


def test_rejects_unsupported_and_duplicate_files():
    result = load_file_payloads(
        [
            FilePayload(name="image.png", data=b"not-an-image"),
            FilePayload(name="notes.txt", data=b"same content"),
            FilePayload(name="copy.txt", data=b"same content"),
        ]
    )

    assert result.files_processed == 1
    assert len(result.errors) == 2
    assert any("unsupported" in error for error in result.errors)
    assert any("duplicate" in error for error in result.errors)


def test_chunking_adds_stable_metadata():
    source = Document(
        page_content="A" * 900,
        metadata={"source": "long.txt", "page": 2},
    )

    chunks = chunk_documents([source], chunk_size=300, chunk_overlap=50, max_chunks=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "long.txt" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_loads_files_from_paths(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("A small knowledge base.", encoding="utf-8")

    result = load_file_payloads_from_paths([path])

    assert result.files_processed == 1
    assert result.documents[0].page_content == "A small knowledge base."
