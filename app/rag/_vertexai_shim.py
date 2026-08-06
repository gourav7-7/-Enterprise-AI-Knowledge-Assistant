"""Workaround for a ragas bug, not an app issue.

ragas/llms/base.py still does:
    from langchain_community.chat_models.vertexai import ChatVertexAI
That submodule was removed from langchain-community (moved to the
langchain-google-vertexai package). We don't use VertexAI anywhere in
this app, so instead of downgrading langchain-community (which breaks
langgraph/langchain-openai/langchain-classic/langchain-chroma, all of
which need langchain-core >=1.x), we pre-register a fake module that
satisfies ragas's import and nothing else. Import this before `ragas`.
"""

from __future__ import annotations

import sys
import types

_MODULE_NAME = "langchain_community.chat_models.vertexai"

if _MODULE_NAME not in sys.modules:
    _shim = types.ModuleType(_MODULE_NAME)

    class ChatVertexAI:  # pragma: no cover - never actually used
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "ChatVertexAI is a stub (this app doesn't use VertexAI). "
                "If you actually need it, install langchain-google-vertexai "
                "and import ChatVertexAI from there instead."
            )

    _shim.ChatVertexAI = ChatVertexAI
    sys.modules[_MODULE_NAME] = _shim
