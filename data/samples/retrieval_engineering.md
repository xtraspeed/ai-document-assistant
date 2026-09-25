# Retrieval Engineering Handbook

## Purpose

This handbook describes a practical retrieval pipeline for a document question-answering system. The goal is not merely to return semantically similar text; it is to return enough relevant evidence for a model to answer accurately and cite its sources.

## Hybrid retrieval

A robust demo uses two complementary retrievers:

1. **Dense retrieval** maps a query and document chunks into vector space. It is effective for paraphrases, conceptual questions, and wording that differs from the source.
2. **BM25 lexical retrieval** ranks exact and approximate keyword matches. It is useful for product names, error codes, dates, and other details that embeddings can miss.

The two ranked lists can be combined with reciprocal-rank fusion (RRF). For a result with rank `r`, its fused contribution is `weight / (k + r)`, where `k` controls how quickly rank differences affect the score. RRF is simple, explainable, and does not require calibrated similarity scores from different retrievers.

## Chunking

Chunks should be small enough to fit comfortably in the answer context but large enough to preserve meaning. A practical starting point is 800–1,200 characters with 100–200 characters of overlap. Every chunk should retain metadata such as the source filename, page number, and chunk index.

Page metadata is especially useful for PDFs because it lets the user verify an answer against the original document. Chunking should be tested with representative questions; a larger chunk is not automatically better.

## Grounded generation

The generation prompt should clearly separate instructions from retrieved passages. Retrieved text is untrusted reference data and must never be treated as executable instructions. The model should cite supporting source labels inline and explicitly say when the documents do not contain enough evidence.

## Evaluation

A small, versioned question set is more useful than a subjective demo. For each question, record the expected source or terms and measure whether the expected evidence appears in the top five results. Useful metrics include hit rate, mean reciprocal rank, retrieval latency, citation precision, and source coverage.

A benchmark should be rerun after changing the chunk size, embedding model, retriever weights, or prompt. The benchmark does not replace human review, but it makes regressions visible.
