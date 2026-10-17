from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chat_id: int = Field(..., strict=True, ge=1, description="ID of an owned chat turn")
    rating: int = Field(..., strict=True, ge=1, le=5, description="1 (poor) to 5 (great)")
    comment: str | None = Field(None, max_length=2000, description="Optional feedback")


class FeedbackResponse(BaseModel):
    status: str
    id: int
