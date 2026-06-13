from __future__ import annotations
from langchain_core.documents import Document
from app.config import Settings

def _fake_settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        chunk_size=100,
        chunk_overlap=20,
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )

def test_splitter_produces_multiple_chunks(tmp_path) -> None:
    from app.rag.ingestion import DocIngestor

    ingestor = DocIngestor(_fake_settings(tmp_path))
    long_txt = "word" * 500
    pages = [Document(page_content=long_txt, metadata = {"source":"doc.pdf", "page":0})]

    chunks = ingestor._splitter.split_documents(pages)

    assert len(chunks) > 1

def test_splitter_preserves_metadata(tmp_path) -> None:
    from app.rag.ingestion import DocIngestor
    ingestor = DocIngestor(_fake_settings(tmp_path))
    pages = [Document(page_content="x " * 300, metadata={"source":"doc.pdf","page":3})]
    chunks = ingestor._splitter.split_documents(pages)

    assert all(c.metadata.get("page") == 3 for c in chunks)
    assert all(c.metadata.get("source") == "doc.pdf" for c in chunks)

    