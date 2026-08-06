"""Unit tests for app/rag/chain.py (RAGChain, ConversationalRAGChain).

No network/API calls. Heavy __init__ dependencies (LLM, retriever) are
bypassed via object.__new__, and self._chain is swapped for a fake stub
exposing sync .invoke() and async .ainvoke() -- so these tests exercise
the real answer()/aanswer() logic without needing an OpenAI key or a
live Chroma store.
"""

from __future__ import annotations

import asyncio

from app.rag.chain import ConversationalRAGChain, RAGChain, _format_sources


class _FakeDoc:
    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}


class _FakeChain:
    """Stub for the LCEL runnable built by create_retrieval_chain."""

    def __init__(self, response: dict, raise_on_call: bool = False) -> None:
        self._response = response
        self._raise_on_call = raise_on_call
        self.invoke_calls: list[dict] = []
        self.ainvoke_calls: list[dict] = []

    def invoke(self, payload: dict) -> dict:
        self.invoke_calls.append(payload)
        if self._raise_on_call:
            raise RuntimeError("boom")
        return self._response

    async def ainvoke(self, payload: dict) -> dict:
        self.ainvoke_calls.append(payload)
        if self._raise_on_call:
            raise RuntimeError("boom")
        return self._response


def _make_conversational_chain(response: dict, raise_on_call: bool = False) -> ConversationalRAGChain:
    chain = object.__new__(ConversationalRAGChain)
    chain._settings = None
    chain._chain = _FakeChain(response, raise_on_call=raise_on_call)
    return chain


def _make_rag_chain(response: dict) -> RAGChain:
    chain = object.__new__(RAGChain)
    chain._settings = None
    chain._chain = _FakeChain(response)
    return chain


# ---- _format_sources ----------------------------------------------------

def test_format_sources_extracts_source_page_snippet() -> None:
    doc = _FakeDoc("x" * 250, {"source": "handbook.pdf", "page": 3})

    sources = _format_sources([doc])

    assert sources == [{"source": "handbook.pdf", "page": 3, "snippet": "x" * 200}]


def test_format_sources_defaults_unknown_source() -> None:
    doc = _FakeDoc("hello")

    sources = _format_sources([doc])

    assert sources[0]["source"] == "unknown"
    assert sources[0]["page"] is None


# ---- RAGChain -------------------------------------------------------------

def test_rag_chain_answer_is_sync_and_uses_invoke() -> None:
    chain = _make_rag_chain({"answer": "42", "context": [_FakeDoc("ctx", {"source": "a.pdf"})]})

    result = chain.answer("what is the answer?")

    assert result["answer"] == "42"
    assert result["sources"][0]["source"] == "a.pdf"
    assert chain._chain.invoke_calls == [{"input": "what is the answer?"}]


def test_rag_chain_aanswer_is_async_and_uses_ainvoke() -> None:
    chain = _make_rag_chain({"answer": "42", "context": []})

    result = asyncio.run(chain.aanswer("q"))

    assert result["answer"] == "42"
    assert chain._chain.ainvoke_calls == [{"input": "q"}]
    assert chain._chain.invoke_calls == []  # never touches the sync path


# ---- ConversationalRAGChain -----------------------------------------------
# NOTE: ConversationalRAGChain is async-only now (no sync answer() --
# confirmed dropped from app/rag/chain.py). Only aanswer() is tested.
# If a sync answer() gets re-added later, restore tests for it here,
# mirroring the RAGChain.answer() tests above.

def test_conversational_aanswer_is_async_and_uses_ainvoke() -> None:
    """Regression test for the fixed bug.

    aanswer() must `await self._chain.ainvoke(...)`, not call the sync
    `.invoke()`. Awaiting the sync call's plain dict return was the
    exact cause of:
        TypeError: object dict can't be used in 'await' expression
    """
    chain = _make_conversational_chain({"answer": "hi", "context": []})

    result = asyncio.run(chain.aanswer("hello", chat_history=[]))

    assert result == {"answer": "hi", "sources": [], "error": False}
    assert chain._chain.ainvoke_calls == [{"input": "hello", "chat_history": []}]
    assert chain._chain.invoke_calls == []  # never touches the sync path


def test_conversational_aanswer_returns_error_payload_on_exception() -> None:
    chain = _make_conversational_chain({}, raise_on_call=True)

    result = asyncio.run(chain.aanswer("hello"))

    assert result["error"] is True
    assert result["sources"] == []
