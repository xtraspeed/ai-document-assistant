from rag_assistant.config import Settings


def test_explicit_api_key_overrides_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    settings = Settings.from_environment(api_key="runtime-key")

    assert settings.openai_api_key == "runtime-key"
    assert settings.has_api_key is True


def test_chunk_settings_are_bounded(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")

    settings = Settings.from_environment(api_key="key")

    assert settings.chunk_size == 256
    assert settings.chunk_overlap == 255
