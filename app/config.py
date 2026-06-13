"""Load settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root (parent of app/)
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
    top_k: int
    log_level: str
    collection_name: str

    @classmethod
    def from_env(cls) -> Settings:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            openai_max_retries= int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            embedding_model=os.getenv("EMBEDDING_MODEL"),
            chunk_size = int(os.getenv("CHUNK_SIZE")),
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP")),
            chroma_dir = os.getenv("CHROMA_DIR"),
            top_k = int(os.getenv("TOP_K")),
            log_level = os.getenv("LOG_LEVEL"),
            collection_name = os.getenv("COLLECTION_NAME")
        )


def get_settings() -> Settings:
    return Settings.from_env()
