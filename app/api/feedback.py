"""POST /feedback: record feedback on a chat turn owned by the caller."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_current_user
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback(
    payload: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    record = crud.save_feedback(
        db,
        user_id=current_user.id,
        chat_id=payload.chat_id,
        rating=payload.rating,
        comment=payload.comment,
    )
    return FeedbackResponse(status="recorded", id=record.id)
