"""Load settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_temperature: float
    openai_max_retries: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    chroma_dir: str
    collection_name: str
    top_k: int
    # --- auth / db ---
    jwt_secret_key: str
    jwt_algorithm: str
    access_token_expire_minutes: int
    database_url: str

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE")),
            openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES")),
            embedding_model=os.getenv("EMBEDDING_MODEL"),
            chunk_size=int(os.getenv("CHUNK_SIZE")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
            chroma_dir=os.getenv("CHROMA_DIR"),
            collection_name=os.getenv("COLLECTION_NAME"),
            top_k=int(os.getenv("TOP_K")),
            jwt_secret_key=os.getenv("JWT_SECRET_KEY"),
            jwt_algorithm=os.getenv("JWT_ALGORITHM"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")),
            database_url=os.getenv("DATABASE_URL")
        )


def get_settings() -> Settings:
    return Settings.from_env()