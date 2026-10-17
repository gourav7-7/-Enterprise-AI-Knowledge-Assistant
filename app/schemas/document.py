# Response model for the /upload endpoint.

from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    status: str = "ingested"

class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
