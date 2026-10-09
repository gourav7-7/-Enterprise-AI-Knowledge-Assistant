from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatHistory, ChatSession, Feedback, User


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


# ---------------------------------------------------------------- sessions

def create_chat_session(db: Session, user_id: int) -> ChatSession:
    session = ChatSession(user_id=user_id)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_chat_session(db: Session, session_id: int, user_id: int) -> ChatSession | None:
    """Return the session only if it belongs to user_id (ownership check)."""
    stmt = select(ChatSession).where(
        ChatSession.id == session_id, ChatSession.user_id == user_id
    )
    return db.scalar(stmt)


def list_chat_sessions(db: Session, user_id: int, limit: int = 50) -> list[ChatSession]:
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())


# ------------------------------------------------------------ chat history

def save_chat(
    db: Session,
    user_id: int,
    question: str,
    answer: str,
    sources: list,
    session_id: int | None = None,
) -> ChatHistory:
    record = ChatHistory(
        user_id=user_id,
        session_id=session_id,
        question=question,
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_chat_history(
    db: Session, user_id: int, limit: int = 50, session_id: int | None = None
) -> list[ChatHistory]:
    """Newest first. session_id=None means ALL of the user's turns (no filter)."""
    stmt = select(ChatHistory).where(ChatHistory.user_id == user_id)
    if session_id is not None:
        stmt = stmt.where(ChatHistory.session_id == session_id)
    stmt = stmt.order_by(ChatHistory.created_at.desc()).limit(limit)
    return list(db.scalars(stmt).all())


def get_recent_chat_history(
    db: Session, user_id: int, limit: int, session_id: int | None = None
) -> list[ChatHistory]:
    """Oldest -> newest, scoped to ONE conversation.

    session_id=None selects the legacy thread (rows with no session), so old
    behaviour is unchanged for clients that don't send a session id.
    """
    stmt = select(ChatHistory).where(ChatHistory.user_id == user_id)
    if session_id is None:
        stmt = stmt.where(ChatHistory.session_id.is_(None))
    else:
        stmt = stmt.where(ChatHistory.session_id == session_id)
    stmt = stmt.order_by(ChatHistory.created_at.desc()).limit(limit)
    records = list(db.scalars(stmt).all())
    records.reverse()
    return records


# ---------------------------------------------------------------- feedback

def save_feedback(
    db: Session,
    user_id: int,
    question: str,
    answer: str,
    rating: int,
    comment: str | None,
) -> Feedback:
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