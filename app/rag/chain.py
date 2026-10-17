from __future__ import annotations

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.core.logger import get_logger
from app.rag.retriever import Retriever

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.

Answer ONLY using the retrieved context.

Guidelines:
- Use only the retrieved context.
- If multiple retrieved chunks contain relevant information,
  combine them into a concise answer.
- If the retrieved context does not contain enough information to answer confidently, reply:
"I don't know based on the provided documents."
- Never invent facts.
- Prefer concise factual answers.
- If possible, preserve terminology from the documents.

Context:
{context}"""

CONTEXTUALIZE_SYSTEM_PROMPT = """Given the chat history and the latest user question — which might reference
context in the chat history — formulate a standalone question that can be
understood without the chat history. Do NOT answer the question; just
reformulate it if needed, otherwise return it unchanged."""


def _build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.chat.temperature,
        api_key=settings.openai_api_key,
        max_retries=settings.openai_max_retries,
    )


def _format_sources(documents) -> list[dict]:
    return [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:200].strip(),
        }
        for doc in documents
    ]


class RAGChain:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        llm = _build_llm(self._settings)
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{input}")]
        )
        self._retriever = Retriever(self._settings, top_k=self._settings.chat.top_k)
        self._doc_chain = create_stuff_documents_chain(llm, prompt)

    def _chain_for_owner(self, owner_id: int):
        retriever = self._retriever.as_lc_retriever(owner_id=owner_id)
        return create_retrieval_chain(retriever, self._doc_chain)

    def answer(self, question: str, *, owner_id: int) -> dict:
        chain = self._chain_for_owner(owner_id)
        res = chain.invoke({"input": question})
        sources = _format_sources(res.get("context", []))
        logger.info("Answered query using %d source chunk(s)", len(sources))
        return {"answer": res["answer"], "sources": sources}

    async def aanswer(self, question: str, *, owner_id: int) -> dict:
        chain = self._chain_for_owner(owner_id)
        res = await chain.ainvoke({"input": question})
        sources = _format_sources(res.get("context", []))
        logger.info("Answered query using %d source chunk(s)", len(sources))
        return {"answer": res["answer"], "sources": sources}


class ConversationalRAGChain:
    def __init__(
        self, settings: Settings | None = None, *, top_k: int | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = _build_llm(self._settings)
        self._retriever = Retriever(self._settings, top_k=top_k)
        self._contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self._doc_chain = create_stuff_documents_chain(self._llm, qa_prompt)

    def _chain_for_owner(self, owner_id: int):
        base_retriever = self._retriever.as_lc_retriever(owner_id=owner_id)
        history_aware_retriever = create_history_aware_retriever(
            self._llm, base_retriever, self._contextualize_prompt
        )
        return create_retrieval_chain(history_aware_retriever, self._doc_chain)

    async def aanswer(
        self,
        question: str,
        chat_history: list | None = None,
        *,
        owner_id: int,
    ) -> dict:
        chain = self._chain_for_owner(owner_id)
        try:
            res = await chain.ainvoke(
                {"input": question, "chat_history": chat_history or []}
            )
        except Exception:
            logger.error("Conversational RAG generation failed.")
            return {
                "answer": "Sorry - I ran into an error answering that. Please try again.",
                "sources": [],
                "error": True,
            }
        sources = _format_sources(res.get("context", []))
        logger.info("Answered query using %d source chunk(s)", len(sources))
        return {"answer": res["answer"], "sources": sources, "error": False}
