from __future__ import annotations

import pytest

from app.config import Settings


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
    )
