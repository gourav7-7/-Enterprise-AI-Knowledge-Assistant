"""POST /feedback : record a rating/comment on an answer.
 
Plain `def` endpoint: it does a small blocking file write, threadpooled by FastAPI.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends
from app.db.feedback_store import FeedbackStore
from app.dependencies import get_feedback_store
from app.schemas.feedback import FeedbackResponse, FeedbackRequest

router = APIRouter(tags=["feedback"])

@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback(payload: FeedbackRequest, store: FeedbackStore = Depends(get_feedback_store)) -> FeedbackResponse:
    feedback_id = store.add(
        question=payload.question,
        answer=payload.answer,
        rating=payload.rating,
        comment=payload.comment,
    )
    return FeedbackResponse(status="recorded", id=feedback_id)