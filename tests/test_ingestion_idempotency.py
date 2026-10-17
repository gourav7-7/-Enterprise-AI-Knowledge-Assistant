"""Idempotent ingestion (app/rag/ingestion.py).

Chroma, embeddings and the PDF loader are faked, so these run fully offline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from app.core.exception import DocumentNotFoundError, IngestionError
from app.rag import ingestion as ingestion_module
from app.rag.ingestion import DocIngestor


class FakeStore:
    """Minimal in-memory stand-in for the parts of Chroma that DocIngestor uses."""

    def __init__(self) -> None:
        self.docs: dict[str, tuple[str, dict]] = {}
        self.add_calls = 0

    def _match(self, where: dict) -> list[str]:
        return [
            doc_id
            for doc_id, (_, meta) in self.docs.items()
            if all(meta.get(k) == v for k, v in where.items())
        ]

    def get(self, where=None, include=None) -> dict:
        ids = self._match(where or {})
        return {"ids": ids, "metadatas": [self.docs[i][1] for i in ids]}

    def delete(self, ids=None, where=None) -> None:
        for doc_id in ids or []:
            self.docs.pop(doc_id, None)
        if where is not None:
            for doc_id in self._match(where):
                self.docs.pop(doc_id, None)

    def add_documents(self, documents, ids=None) -> None:
        self.add_calls += 1
        for doc, doc_id in zip(documents, ids, strict=False):
            self.docs[doc_id] = (doc.page_content, dict(doc.metadata))

    def delete_collection(self) -> None:
        self.docs.clear()


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    fake = FakeStore()
    builds = []

    def _chroma(**kwargs):
        builds.append(kwargs)
        return fake

    monkeypatch.setattr(ingestion_module, "Chroma", _chroma)
    monkeypatch.setattr(ingestion_module, "OpenAIEmbeddings", lambda **kw: object())

    def _loader(path: str):
        # Pages are the file's text split on form-feed; metadata mimics PyPDFLoader.
        text = open(path, encoding="utf-8").read()
        pages = [
            Document(page_content=t, metadata={"source": path, "page": i})
            for i, t in enumerate(text.split("\f"))
        ]
        return SimpleNamespace(load=lambda: pages)

    monkeypatch.setattr(ingestion_module, "PyPDFLoader", _loader)
    fake.builds = builds
    return fake


@pytest.fixture
def ingestor(store, fake_settings) -> DocIngestor:
    return DocIngestor(fake_settings)


def _write_pdf(path, pages: list[str]):
    path.write_text("\f".join(pages), encoding="utf-8")
    return path


LONG_PAGE = "gradient descent minimises a loss function step by step. " * 12


def test_first_ingest_stores_chunks_with_stable_ids_and_metadata(
    ingestor, store, tmp_path
) -> None:
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE, LONG_PAGE])

    result = ingestor.ingest(pdf)

    assert result["status"] == "ingested"
    assert result["file"] == "notes.pdf"
    assert result["pages"] == 2
    assert result["chunks"] == len(store.docs) > 2
    assert store.add_calls == 1
    for doc_id, (_, meta) in store.docs.items():
        assert meta["source"] == "notes.pdf"
        assert meta["n_chunks"] == result["chunks"]
        assert doc_id.startswith(meta["file_hash"][:16])
        assert len(meta["file_hash"]) == 64


def test_reingesting_identical_file_is_skipped(ingestor, store, tmp_path) -> None:
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE, LONG_PAGE])
    first = ingestor.ingest(pdf)

    second = ingestor.ingest(pdf)

    assert second["status"] == "already_ingested"
    assert second["chunks"] == first["chunks"]
    assert second["pages"] == first["pages"]
    assert store.add_calls == 1  # nothing was embedded the second time
    assert len(store.docs) == first["chunks"]  # and nothing was duplicated


def test_identical_content_under_another_name_is_skipped(
    ingestor, store, tmp_path
) -> None:
    ingestor.ingest(_write_pdf(tmp_path / "a.pdf", [LONG_PAGE]))

    result = ingestor.ingest(_write_pdf(tmp_path / "copy_of_a.pdf", [LONG_PAGE]))

    assert result["status"] == "already_ingested"
    assert store.add_calls == 1


def test_partial_earlier_ingest_is_cleaned_up_and_redone(
    ingestor, store, tmp_path
) -> None:
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE, LONG_PAGE])
    first = ingestor.ingest(pdf)
    store.docs.pop(next(iter(store.docs)))  # simulate a crash that lost one chunk

    result = ingestor.ingest(pdf)

    assert result["status"] == "ingested"
    assert len(store.docs) == first["chunks"]
    assert store.add_calls == 2


def test_changed_file_with_same_name_replaces_old_chunks(
    ingestor, store, tmp_path
) -> None:
    pdf = tmp_path / "notes.pdf"
    _write_pdf(pdf, ["old content about regression. " * 20])
    ingestor.ingest(pdf)
    old_ids = set(store.docs)

    _write_pdf(pdf, ["brand new content about clustering. " * 20])
    result = ingestor.ingest(pdf)

    assert result["status"] == "ingested"
    assert old_ids.isdisjoint(store.docs)  # old version fully removed
    assert len(store.docs) == result["chunks"]


def test_other_files_are_untouched_when_one_is_replaced(
    ingestor, store, tmp_path
) -> None:
    ingestor.ingest(_write_pdf(tmp_path / "keep.pdf", ["keep me. " * 30]))
    keep_ids = set(store.docs)
    pdf = tmp_path / "notes.pdf"
    _write_pdf(pdf, ["v1 text. " * 30])
    ingestor.ingest(pdf)

    _write_pdf(pdf, ["v2 text. " * 30])
    ingestor.ingest(pdf)

    assert keep_ids <= set(store.docs)


def test_unsupported_extension_is_rejected(ingestor, tmp_path) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")

    with pytest.raises(IngestionError):
        ingestor.ingest(txt)


def test_missing_file_is_reported(ingestor, tmp_path) -> None:
    with pytest.raises(DocumentNotFoundError):
        ingestor.ingest(tmp_path / "nope.pdf")


def test_reset_drops_and_rebuilds_the_collection(ingestor, store, tmp_path) -> None:
    ingestor.ingest(_write_pdf(tmp_path / "notes.pdf", [LONG_PAGE]))
    builds_before = len(store.builds)

    ingestor.reset()

    assert store.docs == {}
    assert len(store.builds) == builds_before + 1
