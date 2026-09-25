# AI Document Assistant

**Live demo:** [ai-document-assistant-glearpxyg6v9m36arajubn.streamlit.app](https://ai-document-assistant-glearpxyg6v9m36arajubn.streamlit.app/)  
**Source:** [github.com/xtraspeed/ai-document-assistant](https://github.com/xtraspeed/ai-document-assistant)

A production-minded Retrieval-Augmented Generation (RAG) document assistant built with **Streamlit**, **LangChain**, and **OpenAI**.

The application lets a user upload a private set of PDF, TXT, Markdown, or DOCX files, builds a per-session knowledge base, retrieves relevant passages with hybrid search, and generates grounded answers with inline source citations.

## Why this is more than a basic chatbot

A tutorial RAG app usually uploads one document and calls a vector store. This project adds the engineering pieces that make a portfolio project credible:

- Dense OpenAI embeddings plus BM25 lexical retrieval
- Weighted reciprocal-rank fusion (RRF) for ranking and reranking
- Page-aware metadata and source previews
- Conversation-aware query rewriting for follow-up questions
- Grounded generation that refuses unsupported claims and cites passages
- File validation, duplicate detection, size/count limits, and per-session isolation
- Safe secret handling through environment variables or Streamlit secrets
- Optional shared access code for a public deployment
- Request metrics for latency, retrieval count, and token usage
- Reproducible retrieval evaluation with hit rate and MRR
- Sample documents, tests, CI, Docker support, and deployment documentation

## Architecture

```text
Upload / sample documents
          ↓
   Parse + normalize
          ↓
  Page-aware LangChain Documents
          ↓
 Recursive chunking + metadata
          ↓
 ┌────────────────────────────┐
 │ OpenAI embeddings + FAISS  │  dense retrieval
 │ BM25Okapi                  │  lexical retrieval
 └──────────────┬─────────────┘
                ↓
       Weighted reciprocal-rank fusion
                ↓
       Top-k evidence passages
                ↓
   OpenAI grounded answer + [S#] citations
```

## Features

### Document ingestion

- PDF pages are extracted with `pypdf` and retain page numbers.
- TXT and Markdown files are UTF-8 normalized.
- DOCX paragraphs and tables are extracted with `python-docx`.
- Empty, unsupported, oversized, and duplicate files are reported without stopping valid files.
- Chunks retain source filename, page, chunk index, content hash, and a stable chunk ID.

### Retrieval

- OpenAI `text-embedding-3-small` embeddings by default.
- FAISS dense retrieval.
- BM25 lexical retrieval for exact terms and rare identifiers.
- Weighted RRF combines the two ranked lists without requiring score calibration.
- A keyword-only fallback keeps the upload/index flow inspectable if embeddings are unavailable.

### Answering

- OpenAI chat model, configurable with `OPENAI_CHAT_MODEL`.
- Follow-up questions are rewritten into standalone search queries when conversation history exists.
- The answer prompt treats retrieved text as untrusted reference data.
- Answers must cite source labels such as `[S1]` and acknowledge insufficient evidence.

### Evaluation

The bundled benchmark checks whether expected evidence appears in the top five retrieved chunks. It reports:

- Hit rate
- Mean reciprocal rank (MRR)
- Mean retrieval latency
- Per-case source results

Run it with:

```powershell
$env:OPENAI_API_KEY = "your-key-here"
python scripts/run_evaluation.py
```

The exact numbers should be copied into the README only after running the benchmark in your environment.

## Local setup

Python 3.10+ is required.

```powershell
cd C:\Users\Ashish\Documents\ai_document_assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Add your key to `.env`:

```dotenv
OPENAI_API_KEY=your-key-here
```

Start the app:

```powershell
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). `app.py` is also provided as a compatibility entrypoint:

```powershell
streamlit run app.py
```

Do not commit `.env` or `.streamlit/secrets.toml`.

## Tests and quality checks

```powershell
pytest -q
ruff check .
```

The test suite uses fake embeddings for retrieval tests and does not call OpenAI.

## Streamlit Community Cloud deployment

1. Create a new GitHub repository and push this project.
2. In Streamlit Community Cloud, choose **New app** and connect the repository.
3. Use `streamlit_app.py` as the entrypoint.
4. Add `OPENAI_API_KEY` under **Settings → Secrets**.
5. Deploy and wait for the health check to pass.
6. Share the generated `*.streamlit.app` URL.

Optional secrets/settings:

| Name | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI authentication | required for answers |
| `OPENAI_CHAT_MODEL` | Chat model | `gpt-4o-mini` |
| `OPENAI_EMBEDDINGS_MODEL` | Embedding model | `text-embedding-3-small` |
| `APP_ACCESS_CODE` | Optional shared gate | disabled |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `150` |
| `TOP_K` | Retrieved passages per question | `5` |
| `MAX_FILE_SIZE_MB` | Per-file upload limit | `10` |
| `MAX_FILES` | Files per session | `10` |

For a public portfolio demo, set a shared `APP_ACCESS_CODE` if the OpenAI budget is limited. The link remains external, but visitors need the code.

## Troubleshooting

### OpenAI reports no remaining credits

OpenAI embeddings and chat generation require an active billing balance or API credits. If the sidebar reports that semantic retrieval is unavailable, add credits in the OpenAI billing settings and rebuild the index. The app intentionally falls back to keyword-only retrieval so the upload flow remains inspectable, but chat answers still require a usable OpenAI chat balance.

## Docker

```powershell
docker build -t ai-document-assistant .
docker run --rm -p 8501:8501 --env-file .env ai-document-assistant
```

## Security and production notes

- Never put API keys in Git, screenshots, logs, or issue reports.
- The default index lives in the current Streamlit session and is not written to disk.
- Public deployments should use an access code or authentication and should monitor API usage.
- Scanned/image-only PDFs require OCR before they can be searched; this is an intentional scope boundary.
- The app records aggregate session metrics but does not intentionally log raw questions or document contents.
- Streamlit Community Cloud's free tier may sleep and should not be treated as a durable production data store.

## Suggested résumé description

> Built and deployed a citation-first RAG document assistant with Streamlit, LangChain, OpenAI embeddings, FAISS, BM25 hybrid retrieval, reciprocal-rank fusion, conversational query rewriting, page-level citations, retrieval evaluation, automated tests, CI, and secure cloud configuration.

Replace the general description with measured results from your own evaluation run before submitting it to a résumé.
