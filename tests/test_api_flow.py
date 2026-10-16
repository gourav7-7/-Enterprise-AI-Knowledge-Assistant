"""End-to-end API tests: auth, rate limiting, sessions, history, ownership.

Real routers, real crud and a real (in-memory SQLite) database. Only the RAG
chain is faked, so nothing here touches OpenAI or Chroma.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import auth, history, query, sessions
from app.core.exception import register_exception_handlers
from app.core.rate_limit import login_failure_limiter, register_limiter
from app.db import models  # noqa: F401  (registers tables on Base)
from app.db.database import Base, get_db
from app.dependencies import get_conversational_rag_chain

PASSWORD = "Zx9-kV2m-Qp7"
OTHER_PASSWORD = "Rt5-hN8w-Lc3"


@pytest.fixture(autouse=True)
def _clean_limiters():
    for limiter in (login_failure_limiter, register_limiter):
        limiter._events.clear()
    yield
    for limiter in (login_failure_limiter, register_limiter):
        limiter._events.clear()


@pytest.fixture
def chain() -> MagicMock:
    mock = MagicMock()
    mock.aanswer = AsyncMock(
        return_value={"answer": "42", "sources": [], "error": False}
    )
    return mock


@pytest.fixture
def client(chain) -> TestClient:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one shared in-memory DB across connections
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def _get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    register_exception_handlers(app)
    for module in (auth, sessions, history, query):
        app.include_router(module.router)
    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_conversational_rag_chain] = lambda: chain
    return TestClient(app)


def _register(client: TestClient, username: str, password: str = PASSWORD):
    return client.post("/auth/register", json={"username": username, "password": password})


def _login(client: TestClient, username: str, password: str = PASSWORD):
    return client.post("/auth/login", data={"username": username, "password": password})


def _headers(client: TestClient, username: str, password: str = PASSWORD) -> dict:
    assert _register(client, username, password).status_code == 201
    token = _login(client, username, password).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------ registration

def test_register_creates_user_without_leaking_password(client) -> None:
    response = _register(client, "alice")

    assert response.status_code == 201
    assert set(response.json()) == {"id", "username"}


def test_register_trims_username_whitespace(client) -> None:
    response = _register(client, "  alice_01  ")

    assert response.status_code == 201
    assert response.json()["username"] == "alice_01"


def test_register_duplicate_username_returns_409(client) -> None:
    _register(client, "alice")

    assert _register(client, "alice").status_code == 409


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("alice", "short"),                  # too short
        ("alice", "password123"),            # common password
        ("alice", "aaaaaaaaaaaa"),           # repetitive
        ("alice", "my-alice-secret-1"),      # contains the username
        ("bad name!", PASSWORD),             # username characters
        ("al", PASSWORD),                    # username too short
    ],
)
def test_register_rejects_weak_input(client, username, password) -> None:
    assert _register(client, username, password).status_code == 422


# ------------------------------------------------------------------- login

def test_login_returns_token_that_authenticates_me(client) -> None:
    headers = _headers(client, "alice")

    response = client.get("/auth/me", headers=headers)

    assert response.status_code == 200
    assert response.json()["username"] == "alice"


def test_login_failure_does_not_reveal_whether_user_exists(client) -> None:
    _register(client, "alice")

    wrong_password = _login(client, "alice", "not-the-password")
    unknown_user = _login(client, "nobody", "not-the-password")

    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()


def test_protected_routes_require_a_token(client) -> None:
    assert client.get("/auth/me").status_code == 401
    assert client.post("/sessions").status_code == 401
    assert client.get("/history").status_code == 401
    assert client.post("/query", json={"question": "hi"}).status_code == 401


# ----------------------------------------------------------- rate limiting

def test_login_is_locked_after_repeated_failures(client, monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    _register(client, "alice")

    statuses = [_login(client, "alice", "wrong-password").status_code for _ in range(5)]
    locked = _login(client, "alice", "wrong-password")

    assert statuses == [401] * 5
    assert locked.status_code == 429
    assert locked.json()["error"]["code"] == "rate_limited"
    assert locked.json()["error"]["details"]["retry_after_seconds"] >= 1
    # even the correct password is refused while locked
    assert _login(client, "alice", PASSWORD).status_code == 429


def test_lockout_is_per_username(client, monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    _register(client, "alice")
    _register(client, "bobby", OTHER_PASSWORD)
    for _ in range(5):
        _login(client, "alice", "wrong-password")

    assert _login(client, "bobby", OTHER_PASSWORD).status_code == 200


def test_successful_login_resets_the_failure_counter(client, monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    _register(client, "alice")
    for _ in range(4):
        assert _login(client, "alice", "wrong-password").status_code == 401

    assert _login(client, "alice", PASSWORD).status_code == 200

    for _ in range(4):  # counter restarted, so these are still plain 401s
        assert _login(client, "alice", "wrong-password").status_code == 401


def test_registration_is_rate_limited_per_client(client, monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    assert _register(client, "alice").status_code == 201
    for _ in range(9):  # attempts count even when they fail
        assert _register(client, "alice").status_code == 409

    blocked = _register(client, "someone_else")

    assert blocked.status_code == 429


# ---------------------------------------------------------------- sessions

def test_create_and_list_sessions_are_private_to_each_user(client) -> None:
    alice = _headers(client, "alice")
    bobby = _headers(client, "bobby", OTHER_PASSWORD)

    created = client.post("/sessions", headers=alice)
    alice_list = client.get("/sessions", headers=alice).json()
    bobby_list = client.get("/sessions", headers=bobby).json()

    assert created.status_code == 201
    assert [s["id"] for s in alice_list] == [created.json()["id"]]
    assert bobby_list == []


def test_query_in_a_session_only_sees_that_sessions_history(client, chain) -> None:
    headers = _headers(client, "alice")
    s1 = client.post("/sessions", headers=headers).json()["id"]
    s2 = client.post("/sessions", headers=headers).json()["id"]

    client.post("/query", headers=headers, json={"question": "q1", "session_id": s1})
    client.post("/query", headers=headers, json={"question": "q2", "session_id": s1})
    history_in_s1 = chain.aanswer.call_args.kwargs["chat_history"]
    client.post("/query", headers=headers, json={"question": "q3", "session_id": s2})
    history_in_s2 = chain.aanswer.call_args.kwargs["chat_history"]

    assert [m.content for m in history_in_s1] == ["q1", "42"]
    assert history_in_s2 == []


def test_legacy_thread_is_separate_from_sessions(client, chain) -> None:
    headers = _headers(client, "alice")
    sid = client.post("/sessions", headers=headers).json()["id"]
    client.post("/query", headers=headers, json={"question": "in session", "session_id": sid})

    client.post("/query", headers=headers, json={"question": "legacy one"})

    assert chain.aanswer.call_args.kwargs["chat_history"] == []


def test_user_cannot_query_another_users_session(client, chain) -> None:
    alice = _headers(client, "alice")
    bobby = _headers(client, "bobby", OTHER_PASSWORD)
    alice_session = client.post("/sessions", headers=alice).json()["id"]

    response = client.post(
        "/query", headers=bobby, json={"question": "hi", "session_id": alice_session}
    )

    assert response.status_code == 404
    chain.aanswer.assert_not_awaited()


def test_user_cannot_read_another_users_session_history(client) -> None:
    alice = _headers(client, "alice")
    bobby = _headers(client, "bobby", OTHER_PASSWORD)
    alice_session = client.post("/sessions", headers=alice).json()["id"]
    client.post("/query", headers=alice, json={"question": "secret", "session_id": alice_session})

    response = client.get(f"/history?session_id={alice_session}", headers=bobby)

    assert response.status_code == 404
    assert client.get("/history", headers=bobby).json() == []


# ----------------------------------------------------------------- history

def test_history_filters_by_session_and_orders_newest_first(client) -> None:
    headers = _headers(client, "alice")
    s1 = client.post("/sessions", headers=headers).json()["id"]
    s2 = client.post("/sessions", headers=headers).json()["id"]
    for question, sid in [("a1", s1), ("a2", s1), ("b1", s2)]:
        client.post("/query", headers=headers, json={"question": question, "session_id": sid})
    client.post("/query", headers=headers, json={"question": "legacy"})

    in_s1 = client.get(f"/history?session_id={s1}", headers=headers).json()
    everything = client.get("/history", headers=headers).json()

    assert [h["question"] for h in in_s1] == ["a2", "a1"]
    assert all(h["session_id"] == s1 for h in in_s1)
    assert len(everything) == 4
    assert len(client.get("/history?limit=2", headers=headers).json()) == 2


def test_failed_generation_returns_502_and_is_not_saved(client, chain) -> None:
    headers = _headers(client, "alice")
    chain.aanswer.return_value = {"answer": "sorry", "sources": [], "error": True}

    response = client.post("/query", headers=headers, json={"question": "hi"})

    assert response.status_code == 502
    assert client.get("/history", headers=headers).json() == []
