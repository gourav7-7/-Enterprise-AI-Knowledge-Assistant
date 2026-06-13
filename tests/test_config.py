"""Unit tests for configuration (no API calls)."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from app.config import Settings


def test_settings_from_env_requires_api_key() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            Settings.from_env()


def test_settings_from_env_loads_values() -> None:
    env = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_TEMPERATURE": "0.5",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = Settings.from_env()
    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.openai_temperature == 0.5
