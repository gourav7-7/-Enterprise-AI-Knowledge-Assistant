from __future__ import annotations
from langchain_core.documents import Document
import pytest

from app.core.exception import DocumentNotFoundError, IngestionError
from app.rag import ingestion
from app.rag.ingestion import DocIngestor


def test_splitter_produces_multiple_chunks(fake_settings) -> None:

    ingestor = DocIngestor(fake_settings)
    pages = [Document(page_content="word "*500, metadata = {"source":"doc.pdf", "page":0})]

    chunks = ingestor._splitter.split_documents(pages)

    assert len(chunks) > 1

def test_splitter_preserves_metadata(fake_settings) -> None:
    ingestor = DocIngestor(fake_settings)
    pages = [Document(page_content="x " * 300, metadata={"source":"doc.pdf","page":3})]
    chunks = ingestor._splitter.split_documents(pages)

    assert all(c.metadata.get("page") == 3 for c in chunks)
    assert all(c.metadata.get("source") == "doc.pdf" for c in chunks)

def test_missing_file_raises(fake_settings, tmp_path) -> None:
    ingestor = DocIngestor(fake_settings)
    with pytest.raises(DocumentNotFoundError):
        ingestor.ingest(tmp_path / "does_not_exist.pdf")

def test_non_odf_raises(fake_settings, tmp_path) -> None:
    not_pdf = tmp_path / "notes.txt"
    not_pdf.write_text("hello")
    ingestor = DocIngestor(fake_settings)
    with pytest.raises(IngestionError):
        ingestor.ingest(not_pdf)

def test_ingest_stores_chunks(fake_settings, tmp_path, monkeypatch) -> None:
    class _FakeLoader:
        def __init__(self, path):
            self._path = path
        def load(self):
            return [
                Document(
                    page_content="word "* 400,
                    metadata={"source": str(self._path), "page": 0}
                )
            ]

    monkeypatch.setattr(ingestion, "PyPDFLoader", _FakeLoader)

    ingestor = DocIngestor(fake_settings)

    captured: dict = {}

    class _FakeStore:
            def add_documents(self, docs):
                captured["docs"] = docs
    
    ingestor._store = _FakeStore()
    
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")  # passes the .exists() / suffix checks
    
    result = ingestor.ingest(pdf)
    
    assert result["chunks"] > 0
    assert result["pages"] == 1
    assert "docs" in captured and len(captured["docs"]) == result["chunks"]
    # source normalised to the bare filename for clean citations
    assert all(d.metadata["source"] == "doc.pdf" for d in captured["docs"])