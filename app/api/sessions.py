"""Conversation sessions: POST /sessions starts a new chat, GET /sessions lists them."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_current_user

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionResponse(BaseModel):
    id: int
    created_at: datetime


@router.post("", response_model=SessionResponse, status_code=201)
def create_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionResponse:
    session = crud.create_chat_session(db, current_user.id)
    return SessionResponse(id=session.id, created_at=session.created_at)


@router.get("", response_model=list[SessionResponse])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SessionResponse]:
    sessions = crud.list_chat_sessions(db, current_user.id)
    return [SessionResponse(id=s.id, created_at=s.created_at) for s in sessions]