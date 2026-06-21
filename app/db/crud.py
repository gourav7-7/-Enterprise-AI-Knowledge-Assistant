from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatHistory, Feedback, User

def get_user_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))

def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.get(User, user_id)

def create_user(db: Session, username: str, hashed_password: str) -> User:
    user = User(username=username, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def save_chat(db: Session, user_id: int, question: str, answer: str, sources: list) -> ChatHistory:
    record = ChatHistory(
        user_id=user_id,
        question=question,
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_chat_history(db: Session, user_id: int, limit: int = 50) -> list[ChatHistory]:
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.user_id == user_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())

def save_feedback(db: Session, user_id: int, question: str, answer: str, rating: int, comment: str | None) -> Feedback:
    record = Feedback(
        user_id=user_id,
        question=question,
        answer=answer,
        rating=rating,
        comment=comment,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record