from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.exception import NotFoundError
from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_current_user
from app.schemas.history import HistoryItem

router = APIRouter(tags=["history"])


@router.get("/history", response_model=list[HistoryItem])
def history(
    session_id: int | None = Query(
        None, ge=1, description="Only return turns from this conversation."
    ),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[HistoryItem]:
    if session_id is not None and (
        crud.get_chat_session(db, session_id, current_user.id) is None
    ):
        raise NotFoundError("Conversation not found.")

    records = crud.get_chat_history(
        db, current_user.id, limit=limit, session_id=session_id
    )
    return [
        HistoryItem(
            id=r.id,
            session_id=r.session_id,
            question=r.question,
            answer=r.answer,
            sources=json.loads(r.sources) if r.sources else [],
            created_at=r.created_at,
        )
        for r in records
    ]