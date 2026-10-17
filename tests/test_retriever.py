"""Owner isolation against a real temporary Chroma store; no API calls."""

from __future__ import annotations

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.retriever import Retriever


class _LocalEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


@pytest.fixture
def retriever(tmp_path):
    store = Chroma(
        collection_name="owner_isolation_tests",
        embedding_function=_LocalEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    store.add_documents(
        [
            Document(page_content="User one private text", metadata={"owner_id": 1}),
            Document(page_content="User two private text", metadata={"owner_id": 2}),
            Document(page_content="Legacy unowned text", metadata={"source": "old.pdf"}),
        ],
        ids=["owner-one", "owner-two", "legacy"],
    )
    instance = object.__new__(Retriever)
    instance._store = store
    instance._top_k = 10
    try:
        yield instance
    finally:
        store.delete_collection()


@pytest.mark.parametrize("owner_id", [1, 2, 3])
@pytest.mark.parametrize("method", ["dense", "mmr"])
def test_retrieval_only_returns_owned_chunks(retriever, owner_id, method):
    if method == "dense":
        docs = retriever.retrieve("private text", owner_id=owner_id)
    else:
        docs = retriever.as_lc_retriever(owner_id=owner_id).invoke("private text")

    if owner_id == 3:
        assert docs == []
    else:
        assert len(docs) == 1
        assert docs[0].metadata["owner_id"] == owner_id
        expected = "User one private text" if owner_id == 1 else "User two private text"
        assert docs[0].page_content == expected


@pytest.mark.parametrize("owner_id", [None, True, 0, -1, "1"])
def test_invalid_owner_is_rejected_before_accessing_store(owner_id):
    retriever = object.__new__(Retriever)

    class ForbiddenStore:
        def as_retriever(self, **kwargs):
            pytest.fail("Store must not be called for an invalid owner")

        def similarity_search_with_score(self, *args, **kwargs):
            pytest.fail("Store must not be called for an invalid owner")

    retriever._store = ForbiddenStore()
    retriever._top_k = 3
    with pytest.raises(ValueError, match="owner_id"):
        retriever.retrieve("q", owner_id=owner_id)
    with pytest.raises(ValueError, match="owner_id"):
        retriever.as_lc_retriever(owner_id=owner_id)


def test_owner_is_required():
    retriever = object.__new__(Retriever)
    with pytest.raises(TypeError):
        retriever.retrieve("q")
    with pytest.raises(TypeError):
        retriever.as_lc_retriever()
