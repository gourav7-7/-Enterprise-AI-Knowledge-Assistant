from __future__ import annotations
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from app.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self, settings: Settings | None = None,*,top_k:int | None = None) -> None:
        self._settings = settings or get_settings()
        self._top_k = top_k or self._settings.chat.top_k

        self._embeddings = OpenAIEmbeddings(
            model= self._settings.embedding_model,
            api_key= self._settings.openai_api_key
        )

        self._store = Chroma(
            collection_name= self._settings.collection_name,
            embedding_function= self._embeddings,
            persist_directory= self._settings.chroma_dir
        )

    def as_lc_retriever(self) -> VectorStoreRetriever:
        return self._store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self._top_k,
                "fetch_k": max(20, self._top_k * 4),
                "lambda_mult": 0.4,
            },
        )

    def retrieve(self, question: str) -> list[Document]:
        results = self._store.similarity_search_with_score(
            question,
            k=self._top_k,
        )

        docs: list[Document] = []

        for doc, score in results:
            doc.metadata["score"] = score
            docs.append(doc)

        logger.info("Retrieved %d chunks for query", len(docs))

        return docs
