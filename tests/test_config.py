"""Unit tests for configuration (no API calls)."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from app.config import Settings


def test_settings_from_env_requires_api_key() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Settings.from_env()


def test_settings_from_env_loads_values() -> None:
    # Settings.from_env() has no fallback for most fields (by design --
    # fail fast on missing config), so the full set must be mocked here,
    # not just the 3 fields this test happens to assert on.
    env = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_TEMPERATURE": "0.5",
        "OPENAI_MAX_RETRIES": "2",
        "EMBEDDING_MODEL": "text-embedding-3-small",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "50",
        "CHROMA_DIR": "/tmp/chroma",
        "COLLECTION_NAME": "docs",
        "TOP_K": "4",
        "LOG_LEVEL": "INFO",
        "JWT_SECRET_KEY": "test-secret-for-unit-tests-1234567890",
        "JWT_ALGORITHM": "HS256",
        "ACCESS_TOKEN_EXPIRE_MINUTES": "30",
        "DATABASE_URL": "sqlite:///:memory:",
        "CONVERSATION_HISTORY_TURNS": "6",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = Settings.from_env()

    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.openai_temperature == 0.5
    assert settings.openai_max_retries == 2
    assert settings.chunk_size == 500
    assert settings.top_k == 4
    assert settings.conversation_history_turns == 6
    assert settings.chat.temperature == 0.5
    assert settings.chat.top_k == 4
    assert settings.evaluation.temperature == 0.0
    assert settings.evaluation.top_k == 4
