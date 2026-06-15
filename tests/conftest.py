from __future__ import annotations
 
import pytest
 
from app.config import Settings
 
 
@pytest.fixture
def fake_settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        chunk_size=100,
        chunk_overlap=20,
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        openai_model = 'openai_model',  
        openai_temperature = 0.7, 
        openai_max_retries = 1, 
        embedding_model = 'embedding_model', 
        top_k = 3,
        log_level = 'INFO'
    )