"""Unit tests for LangChain chain construction (no API calls)."""

from __future__ import annotations

from langchain.chains import LLMChain

from app.chains.qa_chain import QA_PROMPT, build_qa_chain
from app.config import Settings


def test_prompt_template_variables() -> None:
    assert QA_PROMPT.input_variables == ["context", "question"]


def test_build_qa_chain_returns_llm_chain() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        openai_temperature=0.0,
    )
    chain = build_qa_chain(settings)
    assert isinstance(chain, LLMChain)
    assert chain.prompt is QA_PROMPT
