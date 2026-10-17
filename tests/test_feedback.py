"""Feedback API and ownership tests using a temporary in-memory database."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.feedback import router
from app.core.exception import register_exception_handlers
from app.db.database import Base, get_db
from app.db.models import ChatHistory, Feedback, User
from app.dependencies import get_current_user


@pytest.fixture
def feedback_app():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        db.add_all(
            [
                User(id=1, username="first", hashed_password="unused"),
                User(id=2, username="second", hashed_password="unused"),
            ]
        )
        db.commit()
        db.add_all(
            [
                ChatHistory(id=11, user_id=1, question="Stored question", answer="Stored answer"),
                ChatHistory(id=22, user_id=2, question="Private question", answer="Private answer"),
            ]
        )
        db.commit()

    def override_db():
        with sessions() as db:
            yield db

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1)
    try:
        with TestClient(app) as client:
            yield client, sessions, app
    finally:
        engine.dispose()


def test_feedback_links_owned_turn_and_uses_stored_text(feedback_app):
    client, sessions, _ = feedback_app
    response = client.post(
        "/feedback",
        json={"chat_id": 11, "rating": 5, "comment": "Helpful"},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "recorded"
    with sessions() as db:
        record = db.get(Feedback, response.json()["id"])
        assert record is not None
        assert record.user_id == 1
        assert record.chat_id == 11
        assert record.question == "Stored question"
        assert record.answer == "Stored answer"
        assert record.rating == 5
        assert record.comment == "Helpful"


@pytest.mark.parametrize("chat_id", [22, 999])
def test_feedback_rejects_foreign_or_missing_turn(feedback_app, chat_id):
    client, sessions, _ = feedback_app
    response = client.post("/feedback", json={"chat_id": chat_id, "rating": 5})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["error"]["message"] == "Chat turn not found."
    with sessions() as db:
        assert list(db.scalars(select(Feedback))) == []


def test_feedback_requires_authentication(feedback_app):
    client, sessions, app = feedback_app
    del app.dependency_overrides[get_current_user]
    response = client.post("/feedback", json={"chat_id": 11, "rating": 5})

    assert response.status_code == 401
    with sessions() as db:
        assert list(db.scalars(select(Feedback))) == []


@pytest.mark.parametrize(
    "payload",
    [
        {"rating": 5},
        {"chat_id": 11, "rating": 5, "question": "Forged", "answer": "Forged"},
        {"chat_id": 0, "rating": 5},
        {"chat_id": True, "rating": 5},
        {"chat_id": 11, "rating": 6},
    ],
)
def test_feedback_rejects_invalid_or_forged_payload(feedback_app, payload):
    client, sessions, _ = feedback_app
    response = client.post("/feedback", json=payload)

    assert response.status_code == 422
    with sessions() as db:
        assert list(db.scalars(select(Feedback))) == []
