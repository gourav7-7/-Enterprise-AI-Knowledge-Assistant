"""Unit tests for app/dependencies.py::get_current_user."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import app.dependencies as deps
from app.core.exception import AuthError
from app.db.models import User
from app.dependencies import get_current_user


def _fake_user(user_id: int) -> User:
    return User(id=user_id, username="gourav", hashed_password="hashed")


def test_get_current_user_returns_user_for_valid_token(monkeypatch) -> None:
    monkeypatch.setattr(deps, "decode_access_token", lambda token: "1")
    monkeypatch.setattr(deps.crud, "get_user_by_id", lambda db, user_id: _fake_user(1))

    user = get_current_user(token="valid-token", db=MagicMock())

    assert user.id == 1


def test_get_current_user_casts_decoded_id_to_int(monkeypatch) -> None:
    monkeypatch.setattr(deps, "decode_access_token", lambda token: "42")
    captured: dict = {}

    def _fake_get_user_by_id(db, user_id):
        captured["user_id"] = user_id
        captured["type"] = type(user_id)
        return _fake_user(42)

    monkeypatch.setattr(deps.crud, "get_user_by_id", _fake_get_user_by_id)

    get_current_user(token="valid-token", db=MagicMock())

    assert captured["user_id"] == 42
    assert captured["type"] is int  # decode_access_token returns str, must be cast


def test_get_current_user_raises_autherror_when_user_not_found(monkeypatch) -> None:
    monkeypatch.setattr(deps, "decode_access_token", lambda token: "999")
    monkeypatch.setattr(deps.crud, "get_user_by_id", lambda db, user_id: None)

    with pytest.raises(AuthError):
        get_current_user(token="valid-token", db=MagicMock())