from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field

class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    sources: list[dict] = Field(default_factory=list)
    created_at: datetime