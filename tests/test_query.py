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
from app.db.database import get_db
from app.dependencies import get_conversational_rag_chain, get_current_user


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
def client(monkeypatch, fake_user: SimpleNamespace, fake_chain: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_conversational_rag_chain] = lambda: fake_chain
    app.dependency_overrides[get_db] = lambda: MagicMock()

    monkeypatch.setattr(
        query_module,
        "get_settings",
        lambda: SimpleNamespace(conversation_history_turns=5),
    )
    monkeypatch.setattr(
        query_module.crud,
        "get_recent_chat_history",
        lambda db, user_id, turns: [],
    )
    monkeypatch.setattr(
        query_module.crud,
        "save_chat",
        lambda db, user_id, question, answer, sources: None,
    )

    return TestClient(app)


def test_query_calls_aanswer_and_returns_response(
    client: TestClient, fake_chain: MagicMock
) -> None:
    response = client.post("/query", json={"question": "What color is the sky?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The sky is blue."
    assert body["sources"] == [{"source": "sky.pdf", "page": 1, "snippet": "..."}]

    fake_chain.aanswer.assert_awaited_once_with(
        "What color is the sky?", chat_history=[]
    )


def test_query_saves_chat_after_answering(
    client: TestClient, monkeypatch, fake_user: SimpleNamespace
) -> None:
    saved: dict = {}

    def _fake_save_chat(db, user_id, question, answer, sources) -> None:
        saved["user_id"] = user_id
        saved["question"] = question
        saved["answer"] = answer
        saved["sources"] = sources

    monkeypatch.setattr(query_module.crud, "save_chat", _fake_save_chat)

    client.post("/query", json={"question": "What color is the sky?"})

    assert saved["user_id"] == fake_user.id
    assert saved["question"] == "What color is the sky?"
    assert saved["answer"] == "The sky is blue."


def test_query_converts_chat_history_records_to_messages(
    client: TestClient, monkeypatch, fake_chain: MagicMock
) -> None:
    records = [SimpleNamespace(question="hi", answer="hello")]
    monkeypatch.setattr(
        query_module.crud,
        "get_recent_chat_history",
        lambda db, user_id, turns: records,
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
