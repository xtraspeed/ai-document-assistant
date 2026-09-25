"""Application configuration.

Configuration is read from environment variables first and Streamlit secrets second.
The explicit arguments are used by the UI when a user supplies an API key at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _read_setting(name: str, default: Any = None) -> Any:
    """Read an environment variable, falling back to Streamlit secrets."""

    value = os.getenv(name)
    if value not in (None, ""):
        return value

    try:
        import streamlit as st  # type: ignore[import-not-found]

        secret_value = st.secrets.get(name)
        if secret_value not in (None, ""):
            return secret_value
    except Exception:
        # The core package must remain usable in tests and command-line scripts
        # where Streamlit or a secrets file is not available.
        pass

    return default


def _as_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = _read_setting(name, default)
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _as_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = _read_setting(name, default)
    try:
        return min(maximum, max(minimum, float(value)))
    except (TypeError, ValueError):
        return default


def _as_bool(name: str, default: bool) -> bool:
    value = _read_setting(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for ingestion, retrieval, and generation."""

    openai_api_key: str | None
    chat_model: str
    embeddings_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_file_size_mb: int
    max_files: int
    max_chunks: int
    temperature: float
    app_access_code: str | None
    allow_extractive_fallback: bool = True

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key)

    @classmethod
    def from_environment(
        cls,
        *,
        api_key: str | None = None,
        access_code: str | None = None,
    ) -> "Settings":
        resolved_key = api_key or _read_setting("OPENAI_API_KEY")
        resolved_access_code = access_code or _read_setting("APP_ACCESS_CODE")

        chunk_size = _as_int("CHUNK_SIZE", 1_000, minimum=256)
        chunk_overlap = _as_int("CHUNK_OVERLAP", 150, minimum=0)
        chunk_overlap = min(chunk_overlap, chunk_size - 1)

        return cls(
            openai_api_key=str(resolved_key) if resolved_key else None,
            chat_model=str(_read_setting("OPENAI_CHAT_MODEL", "gpt-4o-mini")),
            embeddings_model=str(
                _read_setting("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
            ),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_as_int("TOP_K", 5, minimum=1),
            max_file_size_mb=_as_int("MAX_FILE_SIZE_MB", 10, minimum=1),
            max_files=_as_int("MAX_FILES", 10, minimum=1),
            max_chunks=_as_int("MAX_CHUNKS", 6_000, minimum=1),
            temperature=_as_float("TEMPERATURE", 0.0, minimum=0.0, maximum=1.0),
            app_access_code=str(resolved_access_code) if resolved_access_code else None,
            allow_extractive_fallback=_as_bool("ALLOW_EXTRACTIVE_FALLBACK", True),
        )
