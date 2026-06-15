# Request/response models for the /query endpoint.

from __future__ import annotations
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")

class Source(BaseModel):
    source: str = Field(..., description="Source document filename")
    page: int | None = Field(None, description="0-indexed page number, if known")
    snippet: str = Field(..., description="Short excerpt of the cited chunk")

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)

