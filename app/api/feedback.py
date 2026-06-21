"""POST /feedback : record a rating/comment on an answer.
 
Plain `def` endpoint: it does a small blocking file write, threadpooled by FastAPI.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_current_user
from app.schemas.feedback import FeedbackResponse, FeedbackRequest

router = APIRouter(tags=["feedback"])

@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback(payload: FeedbackRequest, current_user: User = Depends(get_current_user),db: Session = Depends(get_db)) -> FeedbackResponse:
    record = crud.save_feedback(
        db,
        user_id=current_user.id,
        question=payload.question,
        answer=payload.answer,
        rating=payload.rating,
        comment=payload.comment,
    )
    return FeedbackResponse(status="recorded", id=record.id)
 