from __future__ import annotations
 
import json
 
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
 
from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_current_user
from app.schemas.history import HistoryItem

router = APIRouter(tags=["history"])

@router.get("/history", response_model=list[HistoryItem])
def history(current_user: User = Depends(get_current_user),db: Session = Depends(get_db)) -> list[HistoryItem]:
    records = crud.get_chat_history(db, current_user.id)
    return [
        HistoryItem(
            id=r.id,
            question=r.question,
            answer=r.answer,
            sources=json.loads(r.sources) if r.sources else [],
            created_at=r.created_at,
        )
        for r in records
    ]