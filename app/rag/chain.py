from __future__ import annotations

from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.core.logger import get_logger
from app.rag.retriever import Retriever

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the answer is not contained in the context, say you don't know — "
    "do not invent information. Be concise and accurate.\n\n"
    "Context:\n{context}"
)

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given the chat history and the latest user question — which might reference "
    "context in the chat history — formulate a standalone question that can be "
    "understood without the chat history. Do NOT answer the question; just "
    "reformulate it if needed, otherwise return it unchanged."
)

def _build_llm(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model = settings.openai_model,
        temperature= settings.openai_temperature,
        api_key= settings.openai_api_key,
        max_retries= settings.openai_max_retries
    )

def _format_sources(documents) -> list[dict]:
    return[
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:200].strip()
        }
        for doc in documents
    ]

class RAGChain:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        llm = _build_llm(self._settings)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        retriever = Retriever(self._settings).as_lc_retriever()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        self._chain = create_retrieval_chain(retriever, doc_chain)

    def answer(self, question: str) -> dict:
        res = self._chain.invoke({"input": question})
        sources = _format_sources(res.get("context", []))
        logger.info("Answered query using %d source chunk(s)", len(sources))
        return {"answer": res["answer"], "sources": sources} 

    async def aanswer(self, question: str) -> dict:
        res = await self._chain.ainvoke({"input": question})
        sources = _format_sources(res.get("context", []))
        logger.info("Answer query (async) using %d source chunk(s)", len(sources))
        return {"answer": res["answer"], "sources": sources}
    
class ConversationalRAGChain:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        llm = _build_llm(self._settings)
        base_retriever = Retriever(self._settings).as_lc_retriever()

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, base_retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        doc_chain = create_stuff_documents_chain(llm, qa_prompt)
        self._chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    def answer(self, question:str, chat_history: list |None = None) -> dict:
        chat_history = chat_history or []
        try:
            res = self._chain.invoke(
                {"input": question, "chat_history": chat_history}
            )
        except Exception:
            logger.exception("Conversational RAG failed for question: %s,question")
            return{
                "answer": "Sorry - I ran into an error answering that. Please try again.",
                "sources": [],
                "error": True

            }
        sources = _format_sources(res.get("context", []))
        logger.info("Answered quesry using &d source chunk(s)", len(sources))
        return {"answer": res["answer"], "sources": sources, "error": False}