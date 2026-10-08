"""POST /upload : upload a PDF and ingest it into the vector store.

Plain `def` endpoint (not async): ingestion is blocking (PDF parse + embeddings),
and FastAPI runs sync endpoints in a threadpool, so the event loop is not stalled.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.exception import IngestionError
from app.core.logger import get_logger
from app.db.models import User
from app.dependencies import get_current_user, get_ingestor
from app.rag.ingestion import DocIngestor
from app.schemas.document import UploadResponse

logger = get_logger(__name__)

router = APIRouter(tags=["documents"])

UPLOAD_DIR = Path("data/uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_READ_CHUNK = 1024 * 1024
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")


def _safe_filename(raw: str | None) -> str:
    """Strip directory parts and unsafe characters; require a .pdf name."""
    name = Path((raw or "").replace("\\", "/")).name
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    if len(name) <= 4 or not name.lower().endswith(".pdf"):
        raise IngestionError("Only .pdf files are supported.")
    return name[-150:]


@router.post("/upload", response_model=UploadResponse)
def upload(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    ingestor: DocIngestor = Depends(get_ingestor),
) -> UploadResponse:
    name = _safe_filename(file.filename)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / name
    tmp = UPLOAD_DIR / f".{uuid.uuid4().hex}.part"
    existed = dest.exists()
    size = 0

    # Stream to a temp file so an oversized or bogus upload never touches `dest`.
    try:
        with tmp.open("wb") as out:
            while chunk := file.file.read(_READ_CHUNK):
                if size == 0 and b"%PDF-" not in chunk[:1024]:
                    raise IngestionError("File is not a valid PDF.")
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise IngestionError(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
                out.write(chunk)
        if size == 0:
            raise IngestionError("Uploaded file is empty.")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)

    # Incremental ingest (no reset) — adds to the corpus.
    try:
        summary = ingestor.ingest(dest)
    except Exception:
        if not existed:
            dest.unlink(missing_ok=True)
        raise

    logger.info("User %s uploaded %s (%d bytes)", current_user.id, name, size)
    return UploadResponse(
        filename=summary["file"],
        pages=summary["pages"],
        chunks=summary["chunks"],
    )