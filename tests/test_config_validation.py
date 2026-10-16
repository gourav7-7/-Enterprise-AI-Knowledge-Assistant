"""Environment parsing and validation (app/config.py)."""

from __future__ import annotations

import pytest

from app.config import Settings

GOOD_SECRET = "a" * 8 + "b" * 8 + "c" * 8 + "d" * 8  # 32 chars, not a placeholder

OPTIONAL_VARS = [
    "OPENAI_MODEL", "OPENAI_TEMPERATURE", "OPENAI_MAX_RETRIES", "EMBEDDING_MODEL",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "CHROMA_DIR", "COLLECTION_NAME", "TOP_K",
    "JWT_ALGORITHM", "ACCESS_TOKEN_EXPIRE_MINUTES", "DATABASE_URL",
    "CONVERSATION_HISTORY_TURNS", "CHAT_TEMPERATURE", "CHAT_TOP_K",
    "EVAL_TEMPERATURE", "EVAL_TOP_K",
]


@pytest.fixture
def env(monkeypatch):
    """Only the two required secrets set; every optional var removed."""
    for name in OPTIONAL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("JWT_SECRET_KEY", GOOD_SECRET)
    return monkeypatch


def test_defaults_apply_when_optional_vars_are_missing(env) -> None:
    s = Settings.from_env()

    assert s.openai_model == "gpt-4o-mini"
    assert s.openai_temperature == 0.3
    assert s.openai_max_retries == 3
    assert s.chunk_size == 700
    assert s.chunk_overlap == 120
    assert s.top_k == 6
    assert s.jwt_algorithm == "HS256"
    assert s.access_token_expire_minutes == 30
    assert s.database_url == "sqlite:///data/app.db"
    assert s.conversation_history_turns == 4
    assert s.chat.top_k == 6 and s.chat.temperature == 0.3
    assert s.evaluation.top_k == 4 and s.evaluation.temperature == 0.0


def test_values_from_env_override_defaults(env) -> None:
    env.setenv("TOP_K", "9")
    env.setenv("CHAT_TOP_K", "2")
    env.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")

    s = Settings.from_env()

    assert s.top_k == 9
    assert s.chat.top_k == 2
    assert s.database_url.startswith("postgresql+psycopg://")


def test_blank_value_is_treated_as_unset(env) -> None:
    env.setenv("TOP_K", "   ")

    assert Settings.from_env().top_k == 6


def test_missing_api_key_is_rejected(env) -> None:
    env.delenv("OPENAI_API_KEY")

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        Settings.from_env()


def test_placeholder_api_key_is_rejected(env) -> None:
    env.setenv("OPENAI_API_KEY", "sk-...")

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        Settings.from_env()


def test_missing_jwt_secret_is_rejected(env) -> None:
    env.delenv("JWT_SECRET_KEY")

    with pytest.raises(ValueError, match="JWT_SECRET_KEY is not set"):
        Settings.from_env()


@pytest.mark.parametrize("secret", ["change-me-32-byte-hex", "change-me", "secret", "short"])
def test_weak_jwt_secret_is_rejected(env, secret) -> None:
    env.setenv("JWT_SECRET_KEY", secret)

    with pytest.raises(ValueError, match="placeholder or shorter than 32"):
        Settings.from_env()


def test_non_numeric_int_names_the_variable(env) -> None:
    env.setenv("TOP_K", "abc")

    with pytest.raises(ValueError, match="TOP_K must be an integer"):
        Settings.from_env()


def test_int_below_minimum_is_rejected(env) -> None:
    env.setenv("TOP_K", "0")

    with pytest.raises(ValueError, match="TOP_K must be >= 1"):
        Settings.from_env()


def test_chunk_overlap_must_be_smaller_than_chunk_size(env) -> None:
    env.setenv("CHUNK_SIZE", "300")
    env.setenv("CHUNK_OVERLAP", "300")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings.from_env()


def test_temperature_out_of_range_is_rejected(env) -> None:
    env.setenv("OPENAI_TEMPERATURE", "3")

    with pytest.raises(ValueError, match="OPENAI_TEMPERATURE must be between"):
        Settings.from_env()


@pytest.mark.parametrize("algorithm", ["none", "RS256", "garbage"])
def test_unsupported_jwt_algorithm_is_rejected(env, algorithm) -> None:
    env.setenv("JWT_ALGORITHM", algorithm)

    with pytest.raises(ValueError, match="JWT_ALGORITHM"):
        Settings.from_env()


def test_jwt_algorithm_is_case_insensitive(env) -> None:
    env.setenv("JWT_ALGORITHM", "hs512")

    assert Settings.from_env().jwt_algorithm == "HS512"
