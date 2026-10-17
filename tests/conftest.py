from __future__ import annotations

import os

# Must run before any `app.*` import: app.db.database reads settings at import time.
# setdefault never overrides real env vars, but load_dotenv() (called when app.config
# is imported) doesn't override them either, so these also shield tests from a real .env.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-at-least-32-characters-long")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

import pytest

from app.config import ChatProfile, EvaluationProfile, Settings


@pytest.fixture
def fake_settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_model="openai_model",
        openai_temperature=0.7,
        openai_max_retries=1,
        embedding_model="embedding_model",
        chunk_size=100,
        chunk_overlap=20,
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        top_k=3,
        jwt_secret_key="test-secret",
        jwt_algorithm="HS256",
        access_token_expire_minutes=30,
        database_url="sqlite:///:memory:",
        conversation_history_turns=4,
        chat=ChatProfile(temperature=0.7, top_k=3),
        evaluation=EvaluationProfile(temperature=0.0, top_k=3),
    )
