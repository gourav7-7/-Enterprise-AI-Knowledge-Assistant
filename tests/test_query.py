"""Tests for POST /query (app/api/query.py).

The router is mounted on a throwaway FastAPI() app so these don't need
the real app.main entrypoint. Auth, DB session, and the RAG chain are
all overridden/mocked -- no real DB, no real OpenAI calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import query as query_module
from app.api.query import router
from app.core.exception import register_exception_handlers
from app.db.database import get_db
from app.dependencies import get_conversational_rag_chain, get_current_user

OWNED_SESSION_ID = 7


@pytest.fixture
def fake_user() -> SimpleNamespace:
    return SimpleNamespace(id=1)


@pytest.fixture
def fake_chain() -> MagicMock:
    chain = MagicMock()
    chain.aanswer = AsyncMock(
        return_value={
            "answer": "The sky is blue.",
            "sources": [{"source": "sky.pdf", "page": 1, "snippet": "..."}],
            "error": False,
        }
    )
    return chain


@pytest.fixture
def saved() -> list[dict]:
    """Every call to crud.save_chat, in order."""
    return []


@pytest.fixture
def history_calls() -> list[dict]:
    """Every call to crud.get_recent_chat_history, in order."""
    return []


@pytest.fixture
def client(
    monkeypatch,
    fake_user: SimpleNamespace,
    fake_chain: MagicMock,
    saved: list[dict],
    history_calls: list[dict],
) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)  # so AppError subclasses become real responses
    app.include_router(router)

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_conversational_rag_chain] = lambda: fake_chain
    app.dependency_overrides[get_db] = lambda: MagicMock()

    monkeypatch.setattr(
        query_module,
        "get_settings",
        lambda: SimpleNamespace(conversation_history_turns=5),
    )

    def _fake_recent(db, user_id, turns, session_id=None):
        history_calls.append({"user_id": user_id, "session_id": session_id})
        return []

    def _fake_get_session(db, session_id, user_id):
        # Only session 7 exists, and it belongs to fake_user.
        if session_id == OWNED_SESSION_ID and user_id == fake_user.id:
            return SimpleNamespace(id=session_id)
        return None

    def _fake_save_chat(db, user_id, question, answer, sources, session_id=None):
        saved.append(
            {
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
            }
        )

    monkeypatch.setattr(query_module.crud, "get_recent_chat_history", _fake_recent)
    monkeypatch.setattr(query_module.crud, "get_chat_session", _fake_get_session)
    monkeypatch.setattr(query_module.crud, "save_chat", _fake_save_chat)

    return TestClient(app)


def test_query_calls_aanswer_and_returns_response(
    client: TestClient, fake_chain: MagicMock
) -> None:
    response = client.post("/query", json={"question": "What color is the sky?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The sky is blue."
    assert body["sources"] == [{"source": "sky.pdf", "page": 1, "snippet": "..."}]
    assert body["session_id"] is None

    fake_chain.aanswer.assert_awaited_once_with(
        "What color is the sky?", chat_history=[]
    )


def test_query_saves_chat_after_answering(
    client: TestClient, saved: list[dict], fake_user: SimpleNamespace
) -> None:
    client.post("/query", json={"question": "What color is the sky?"})

    assert len(saved) == 1
    assert saved[0]["user_id"] == fake_user.id
    assert saved[0]["question"] == "What color is the sky?"
    assert saved[0]["answer"] == "The sky is blue."
    assert saved[0]["session_id"] is None


def test_query_converts_chat_history_records_to_messages(
    client: TestClient, monkeypatch, fake_chain: MagicMock
) -> None:
    records = [SimpleNamespace(question="hi", answer="hello")]
    monkeypatch.setattr(
        query_module.crud,
        "get_recent_chat_history",
        lambda db, user_id, turns, session_id=None: records,
    )

    client.post("/query", json={"question": "follow up"})

    _, kwargs = fake_chain.aanswer.call_args
    history = kwargs["chat_history"]
    assert len(history) == 2
    assert history[0].content == "hi"
    assert history[1].content == "hello"


def test_query_missing_question_returns_422(client: TestClient) -> None:
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_blank_question_returns_422(client: TestClient) -> None:
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422


def test_query_too_long_question_returns_422(client: TestClient) -> None:
    response = client.post("/query", json={"question": "a" * 2001})

    assert response.status_code == 422


def test_query_failed_generation_returns_502_and_is_not_saved(
    client: TestClient, fake_chain: MagicMock, saved: list[dict]
) -> None:
    fake_chain.aanswer.return_value = {
        "answer": "Sorry, something went wrong.",
        "sources": [],
        "error": True,
    }

    response = client.post("/query", json={"question": "What color is the sky?"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "generation_error"
    # The apology must never reach chat history, or it would be replayed to the LLM.
    assert saved == []


def test_query_with_session_scopes_history_and_saves_into_it(
    client: TestClient, saved: list[dict], history_calls: list[dict]
) -> None:
    response = client.post(
        "/query",
        json={"question": "What color is the sky?", "session_id": OWNED_SESSION_ID},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == OWNED_SESSION_ID
    assert history_calls == [{"user_id": 1, "session_id": OWNED_SESSION_ID}]
    assert saved[0]["session_id"] == OWNED_SESSION_ID


def test_query_without_session_uses_legacy_thread(
    client: TestClient, history_calls: list[dict]
) -> None:
    client.post("/query", json={"question": "What color is the sky?"})

    assert history_calls == [{"user_id": 1, "session_id": None}]


def test_query_unknown_or_foreign_session_returns_404(
    client: TestClient, fake_chain: MagicMock, saved: list[dict]
) -> None:
    response = client.post("/query", json={"question": "hi", "session_id": 99999})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    fake_chain.aanswer.assert_not_awaited()
    assert saved == []