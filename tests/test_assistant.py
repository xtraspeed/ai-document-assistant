from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from rag_assistant.assistant import RAGAssistant
from rag_assistant.config import Settings
from rag_assistant.retrieval import HybridRetriever


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="standalone question")
        return AIMessage(content="The answer is supported by [S1].")


def test_assistant_rewrites_follow_up_and_cites_context():
    retriever = HybridRetriever(
        [
            Document(
                page_content="The deployment uses Streamlit Community Cloud.",
                metadata={"source": "deployment.md", "page": 1, "chunk_index": 0},
            )
        ],
        FakeEmbeddings(),
        top_k=1,
    )
    settings = Settings(
        openai_api_key="test-key",
        chat_model="test-chat",
        embeddings_model="test-embedding",
        chunk_size=1_000,
        chunk_overlap=150,
        top_k=1,
        max_file_size_mb=10,
        max_files=10,
        max_chunks=100,
        temperature=0.0,
        app_access_code=None,
    )
    llm = FakeLLM()
    assistant = RAGAssistant(retriever, settings, llm=llm)

    result = assistant.answer(
        "Where is it deployed?",
        history=[{"role": "user", "content": "Tell me about the deployment."}],
    )

    assert result.standalone_query == "standalone question"
    assert "[S1]" in result.answer
    assert result.sources[0].source == "deployment.md"
    assert llm.calls == 2


def test_assistant_uses_labelled_extractive_fallback_without_openai():
    retriever = HybridRetriever(
        [
            Document(
                page_content="The deployment uses Streamlit Community Cloud. It is public.",
                metadata={"source": "deployment.md", "page": 1, "chunk_index": 0},
            )
        ],
        FakeEmbeddings(),
        top_k=1,
    )
    settings = Settings(
        openai_api_key=None,
        chat_model="test-chat",
        embeddings_model="test-embedding",
        chunk_size=1_000,
        chunk_overlap=150,
        top_k=1,
        max_file_size_mb=10,
        max_files=10,
        max_chunks=100,
        temperature=0.0,
        app_access_code=None,
    )

    result = RAGAssistant(retriever, settings).answer("Where is it deployed?")

    assert result.generation_mode == "extractive_fallback"
    assert "OpenAI" in result.answer
    assert "[S1]" in result.answer
