"""POST /upload : upload a PDF and ingest it into the vector store.
 
Plain `def` endpoint (not async): ingestion is blocking (PDF parse + embeddings),
and FastAPI runs sync endpoints in a threadpool, so the event loop is not stalled.
"""

from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile
from app.core.exception import IngestionError
from app.dependencies import get_ingestor
from app.rag.ingestion import DocIngestor
from app.schemas.document import UploadResponse

router = APIRouter(tags=["documents"])

UPLOAD_DIR = Path("data/uploads")

@router.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...),ingestor: DocIngestor = Depends(get_ingestor)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise IngestionError("Only .pdf files are supported.")
 
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(file.file.read())
 
    # Incremental ingest (no reset) — adds to the corpus.
    summary = ingestor.ingest(dest)
    return UploadResponse(
        filename=summary["file"],
        pages=summary["pages"],
        chunks=summary["chunks"],
    )