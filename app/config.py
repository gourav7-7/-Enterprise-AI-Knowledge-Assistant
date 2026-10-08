"""Load settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

_MIN_JWT_SECRET_LEN = 32
_ALLOWED_JWT_ALGORITHMS = {"HS256", "HS384", "HS512"}
_PLACEHOLDER_SECRETS = {"change-me", "changeme", "secret", "your-secret-key"}
_PLACEHOLDER_API_KEYS = {"sk-...", "your-api-key", "your-key-here"}
_SECRET_HINT = 'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'


def _env(name: str) -> str | None:
    """Return the stripped value, or None if unset/blank."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _env_str(name: str, default: str) -> str:
    return _env(name) or default


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = _env(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer, got {raw!r}.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = _env(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{name} must be a number, got {raw!r}.") from None
    if not (minimum <= value <= maximum):
        raise ValueError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _require_api_key() -> str:
    key = _env("OPENAI_API_KEY")
    if key is None or key.lower() in _PLACEHOLDER_API_KEYS:
        raise ValueError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def _require_jwt_secret() -> str:
    secret = _env("JWT_SECRET_KEY")
    if secret is None:
        raise ValueError(f"JWT_SECRET_KEY is not set. {_SECRET_HINT}")
    if (
        secret.lower() in _PLACEHOLDER_SECRETS
        or secret.lower().startswith("change-me")
        or len(secret) < _MIN_JWT_SECRET_LEN
    ):
        raise ValueError(
            f"JWT_SECRET_KEY is a placeholder or shorter than "
            f"{_MIN_JWT_SECRET_LEN} characters. {_SECRET_HINT}"
        )
    return secret


@dataclass(frozen=True)
class ChatProfile:
    temperature: float
    top_k: int


@dataclass(frozen=True)
class EvaluationProfile:
    temperature: float
    top_k: int


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
    conversation_history_turns: int
    # --- Evaluation ---
    chat: ChatProfile
    evaluation: EvaluationProfile

    @classmethod
    def from_env(cls) -> Settings:
        api_key = _require_api_key()
        jwt_secret = _require_jwt_secret()

        jwt_algorithm = _env_str("JWT_ALGORITHM", "HS256").upper()
        if jwt_algorithm not in _ALLOWED_JWT_ALGORITHMS:
            raise ValueError(
                f"JWT_ALGORITHM must be one of {sorted(_ALLOWED_JWT_ALGORITHMS)}, "
                f"got {jwt_algorithm!r}."
            )

        chunk_size = _env_int("CHUNK_SIZE", 700, minimum=50)
        chunk_overlap = _env_int("CHUNK_OVERLAP", 120, minimum=0)
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({chunk_size})."
            )

        openai_temperature = _env_float("OPENAI_TEMPERATURE", 0.3, 0.0, 2.0)
        top_k = _env_int("TOP_K", 6, minimum=1)

        return cls(
            openai_api_key=api_key,
            openai_model=_env_str("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=openai_temperature,
            openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3, minimum=0),
            embedding_model=_env_str("EMBEDDING_MODEL", "text-embedding-3-small"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chroma_dir=_env_str("CHROMA_DIR", "data/chroma"),
            collection_name=_env_str("COLLECTION_NAME", "knowledge_base"),
            top_k=top_k,
            jwt_secret_key=jwt_secret,
            jwt_algorithm=jwt_algorithm,
            access_token_expire_minutes=_env_int("ACCESS_TOKEN_EXPIRE_MINUTES", 30, minimum=1),
            database_url=_env_str("DATABASE_URL", "sqlite:///data/app.db"),
            conversation_history_turns=_env_int("CONVERSATION_HISTORY_TURNS", 4, minimum=0),
            chat=ChatProfile(
                temperature=_env_float("CHAT_TEMPERATURE", openai_temperature, 0.0, 2.0),
                top_k=_env_int("CHAT_TOP_K", top_k, minimum=1),
            ),
            evaluation=EvaluationProfile(
                temperature=_env_float("EVAL_TEMPERATURE", 0.0, 0.0, 2.0),
                top_k=_env_int("EVAL_TOP_K", 4, minimum=1),
            ),
        )


def get_settings() -> Settings:
    return Settings.from_env()