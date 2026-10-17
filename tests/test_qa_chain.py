"""Offline chain tests with mocked LLM/retrieval dependencies."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.rag import chain as chain_module
from app.rag.chain import ConversationalRAGChain, RAGChain, _format_sources


class _FakeChain:
    def __init__(self, response: dict, *, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.invoke_calls = []
        self.ainvoke_calls = []

    def invoke(self, payload: dict) -> dict:
        self.invoke_calls.append(payload)
        if self.fail:
            raise RuntimeError("boom")
        return self.response

    async def ainvoke(self, payload: dict) -> dict:
        self.ainvoke_calls.append(payload)
        await asyncio.sleep(0)
        if self.fail:
            raise RuntimeError("boom")
        return self.response


def _make_chain(cls, response, *, fail=False):
    chain = object.__new__(cls)
    runnable = _FakeChain(response, fail=fail)
    chain._chain_for_owner = MagicMock(return_value=runnable)
    return chain, runnable


def test_format_sources_extracts_source_page_snippet() -> None:
    doc = SimpleNamespace(
        page_content="x" * 250, metadata={"source": "handbook.pdf", "page": 3}
    )
    assert _format_sources([doc]) == [
        {"source": "handbook.pdf", "page": 3, "snippet": "x" * 200}
    ]


def test_format_sources_defaults_unknown_source() -> None:
    doc = SimpleNamespace(page_content="hello", metadata={})
    assert _format_sources([doc]) == [
        {"source": "unknown", "page": None, "snippet": "hello"}
    ]


def test_rag_chain_answer_is_sync_and_uses_invoke() -> None:
    chain, runnable = _make_chain(RAGChain, {"answer": "42", "context": []})
    assert chain.answer("q", owner_id=1) == {"answer": "42", "sources": []}
    chain._chain_for_owner.assert_called_once_with(1)
    assert runnable.invoke_calls == [{"input": "q"}]
    assert runnable.ainvoke_calls == []


def test_rag_chain_aanswer_is_async_and_uses_ainvoke() -> None:
    chain, runnable = _make_chain(RAGChain, {"answer": "42", "context": []})
    assert asyncio.run(chain.aanswer("q", owner_id=2))["answer"] == "42"
    chain._chain_for_owner.assert_called_once_with(2)
    assert runnable.ainvoke_calls == [{"input": "q"}]
    assert runnable.invoke_calls == []


def test_conversational_aanswer_is_async_and_uses_ainvoke() -> None:
    chain, runnable = _make_chain(
        ConversationalRAGChain, {"answer": "hi", "context": []}
    )
    history = [SimpleNamespace(content="previous question")]
    result = asyncio.run(chain.aanswer("hello", chat_history=history, owner_id=2))
    assert result == {"answer": "hi", "sources": [], "error": False}
    chain._chain_for_owner.assert_called_once_with(2)
    assert runnable.ainvoke_calls == [{"input": "hello", "chat_history": history}]
    assert runnable.invoke_calls == []


def test_conversational_aanswer_returns_error_payload_on_exception() -> None:
    chain, _ = _make_chain(ConversationalRAGChain, {}, fail=True)
    result = asyncio.run(chain.aanswer("hello", owner_id=1))
    assert result["error"] is True
    assert result["sources"] == []


@pytest.mark.parametrize("cls", [RAGChain, ConversationalRAGChain])
def test_concurrent_requests_keep_owner_retrievers_separate(monkeypatch, cls) -> None:
    chain = object.__new__(cls)
    chain._retriever = SimpleNamespace(
        as_lc_retriever=MagicMock(side_effect=lambda *, owner_id: owner_id)
    )
    chain._doc_chain = object()
    chain._llm = object()
    chain._contextualize_prompt = object()

    history_factory = MagicMock(side_effect=lambda llm, retriever, prompt: retriever)
    monkeypatch.setattr(chain_module, "create_history_aware_retriever", history_factory)

    def build(retriever, doc_chain):
        doc = SimpleNamespace(
            page_content=f"Owner {retriever} content",
            metadata={"source": f"owner-{retriever}.pdf"},
        )
        return _FakeChain({"answer": str(retriever), "context": [doc]})

    monkeypatch.setattr(chain_module, "create_retrieval_chain", build)

    async def run_both():
        return await asyncio.gather(
            chain.aanswer("q", owner_id=1),
            chain.aanswer("q", owner_id=2),
        )

    first, second = asyncio.run(run_both())
    assert first["answer"] == "1"
    assert second["answer"] == "2"
    assert first["sources"][0]["source"] == "owner-1.pdf"
    assert second["sources"][0]["source"] == "owner-2.pdf"
    assert chain._retriever.as_lc_retriever.call_count == 2
    chain._retriever.as_lc_retriever.assert_any_call(owner_id=1)
    chain._retriever.as_lc_retriever.assert_any_call(owner_id=2)
    assert history_factory.call_count == (2 if cls is ConversationalRAGChain else 0)
