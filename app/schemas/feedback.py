# Request/response models for the /feedback endpoint.
from __future__ import annotations
from pydantic import BaseModel, Field
 
 
class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5, description="1 (poor) to 5 (great)")
    comment: str | None = Field(None, description="Optional free-text feedback")
 
 
class FeedbackResponse(BaseModel):
    status: str
    id: str