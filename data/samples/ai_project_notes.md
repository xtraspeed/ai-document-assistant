# AI Project Notes

## Product goals

The document assistant should make a knowledge base easy to explore without pretending that every answer is certain. The interface should show the passages used to construct an answer and make it simple to start a new session.

## Recommended user flow

1. Upload one or more documents.
2. Review the number of files, pages, and indexed chunks.
3. Ask a focused question.
4. Inspect the cited source passages.
5. Ask a follow-up question that uses conversational context.
6. Export the transcript when the result needs to be shared.

## Responsible design

Uploaded files are held in the current Streamlit session and are not written to the application repository. API keys are read from environment variables or Streamlit secrets rather than source code. File size and count limits protect the host from accidental oversized uploads.

A public deployment should use a shared access code or another access-control mechanism if the OpenAI account has a limited budget. The app should also avoid logging raw document contents or user questions by default.
