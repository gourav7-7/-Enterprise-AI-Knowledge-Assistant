from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_settings = get_settings()
DATABASE_URL = _settings.database_url
_IS_SQLITE = DATABASE_URL.startswith("sqlite")

if DATABASE_URL.startswith("sqlite:///"):
    db_path = DATABASE_URL.replace("sqlite:///", "", 1)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


class Base(DeclarativeBase):
    """Declarative base all ORM models inherit from."""


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # check_same_thread is a SQLite-only option (FastAPI uses threads).
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(retries: int = 10, delay: float = 2.0) -> None:
    """Create tables. Retries because in Docker the app can start before Postgres."""
    from app.db import models  # noqa: F401  (import registers models on Base.metadata)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database initialised (%s)", "sqlite" if _IS_SQLITE else "postgres")
            return
        except OperationalError as e:
            last_error = e
            logger.warning("DB not ready (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(delay)

    raise RuntimeError(
        f"Could not connect to the database after {retries} attempts"
    ) from last_error


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()