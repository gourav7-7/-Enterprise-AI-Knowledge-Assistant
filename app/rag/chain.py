from __future__ import annotations

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.core.logger import get_logger
from app.rag.retriever import Retriver

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. "
    "Answer the user's question using ONLY the context provided below. "
    "If the answer is not contained in the context, say you don't know — "
    "do not invent information. Be concise and accurate.\n\n"
    "Context:\n{context}"
)


class RAGChain:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        llm = ChatOpenAI(
            model=self._settings.openai_model,
            temperature=self._settings.openai_temperature,
            api_key=self._settings.openai_api_key,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )

        retriever = Retriver(self._settings).as_lc_retriever()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        self._chain = create_retrieval_chain(retriever, doc_chain)

    def answer(self, question: str) -> dict:
        res = self._chain.invoke({"input": question})

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:200].strip(),
            }
            for doc in res.get("context", [])
        ]

        logger.info("Answered query using %d source chunk(s)", len(sources))
        return {"answer": res["answer"], "sources": sources} 