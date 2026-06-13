"""
LangChain QA chain — Task 1 deliverable.

Uses:
  - ChatOpenAI (ChatModels)
  - PromptTemplate
  - LLMChain
"""

from __future__ import annotations

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
 
from app.config import Settings, get_settings


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an enterprise knowledge assistant. Answer the user's question clearly and concisely.

Context (may be empty for general questions):
{context}

Question: {question}

Answer:""",
)


def build_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """Create the LangChain ChatOpenAI model (ChatModels API)."""
    cfg = settings or get_settings()
    return ChatOpenAI(
        model=cfg.openai_model,
        temperature=cfg.openai_temperature,
        api_key=cfg.openai_api_key,
    )


def build_qa_chain(settings: Settings | None = None) -> LLMChain:
    """Build an LLMChain wired to PromptTemplate + ChatOpenAI."""
    llm = build_chat_model(settings)
    return LLMChain(llm=llm, prompt=QA_PROMPT, verbose=False)


def run_qa_chain(question: str, context: str = "", settings: Settings | None = None) -> str:
    """Invoke the QA chain and return the model's text response."""
    chain = build_qa_chain(settings)
    result = chain.invoke({"context": context or "No additional context.", "question": question})
    return result["text"].strip()
