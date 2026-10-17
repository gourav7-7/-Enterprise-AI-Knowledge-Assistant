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
            if all(
                meta.get(key) == value
                for clause in where.get("$and", [where])
                for key, value in clause.items()
            )
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

    result = ingestor.ingest(pdf, owner_id=1)

    assert result["status"] == "ingested"
    assert result["file"] == "notes.pdf"
    assert result["pages"] == 2
    assert result["chunks"] == len(store.docs) > 2
    assert store.add_calls == 1
    for doc_id, (_, meta) in store.docs.items():
        assert meta["source"] == "notes.pdf"
        assert meta["n_chunks"] == result["chunks"]
        assert meta["owner_id"] == 1
        assert doc_id.startswith(f"1-{meta['file_hash']}-")
        assert len(meta["file_hash"]) == 64


def test_reingesting_identical_file_is_skipped(ingestor, store, tmp_path) -> None:
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE, LONG_PAGE])
    first = ingestor.ingest(pdf, owner_id=1)

    second = ingestor.ingest(pdf, owner_id=1)

    assert second["status"] == "already_ingested"
    assert second["chunks"] == first["chunks"]
    assert second["pages"] == first["pages"]
    assert store.add_calls == 1  # nothing was embedded the second time
    assert len(store.docs) == first["chunks"]  # and nothing was duplicated


def test_identical_content_under_another_name_is_skipped(
    ingestor, store, tmp_path
) -> None:
    ingestor.ingest(_write_pdf(tmp_path / "a.pdf", [LONG_PAGE]),owner_id=1)

    result = ingestor.ingest(_write_pdf(tmp_path / "copy_of_a.pdf", [LONG_PAGE]),owner_id=1)

    assert result["status"] == "already_ingested"
    assert store.add_calls == 1


def test_partial_earlier_ingest_is_cleaned_up_and_redone(
    ingestor, store, tmp_path
) -> None:
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE, LONG_PAGE])
    first = ingestor.ingest(pdf, owner_id=1)
    store.docs.pop(next(iter(store.docs)))  # simulate a crash that lost one chunk

    result = ingestor.ingest(pdf, owner_id=1)

    assert result["status"] == "ingested"
    assert len(store.docs) == first["chunks"]
    assert store.add_calls == 2


def test_changed_file_with_same_name_replaces_old_chunks(
    ingestor, store, tmp_path
) -> None:
    pdf = tmp_path / "notes.pdf"
    _write_pdf(pdf, ["old content about regression. " * 20])
    ingestor.ingest(pdf, owner_id=1)
    old_ids = set(store.docs)

    _write_pdf(pdf, ["brand new content about clustering. " * 20])
    result = ingestor.ingest(pdf,owner_id=1)

    assert result["status"] == "ingested"
    assert old_ids.isdisjoint(store.docs)  # old version fully removed
    assert len(store.docs) == result["chunks"]


def test_other_files_are_untouched_when_one_is_replaced(
    ingestor, store, tmp_path
) -> None:
    ingestor.ingest(_write_pdf(tmp_path / "keep.pdf", ["keep me. " * 30]), owner_id=1)
    keep_ids = set(store.docs)
    pdf = tmp_path / "notes.pdf"
    _write_pdf(pdf, ["v1 text. " * 30])
    ingestor.ingest(pdf, owner_id=1)

    _write_pdf(pdf, ["v2 text. " * 30])
    ingestor.ingest(pdf, owner_id=1)

    assert keep_ids <= set(store.docs)


def test_unsupported_extension_is_rejected(ingestor, tmp_path) -> None:
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")

    with pytest.raises(IngestionError):
        ingestor.ingest(txt, owner_id=1)


def test_missing_file_is_reported(ingestor, tmp_path) -> None:
    with pytest.raises(DocumentNotFoundError):
        ingestor.ingest(tmp_path / "nope.pdf", owner_id=1)


def test_reset_drops_and_rebuilds_the_collection(ingestor, store, tmp_path) -> None:
    ingestor.ingest(_write_pdf(tmp_path / "notes.pdf", [LONG_PAGE]), owner_id=1)
    builds_before = len(store.builds)

    ingestor.reset()

    assert store.docs == {}
    assert len(store.builds) == builds_before + 1

def test_identical_content_is_ingested_separately_per_owner(
    ingestor, store, tmp_path
) -> None:
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE])

    first = ingestor.ingest(pdf, owner_id=1)
    first_ids = set(store.docs)
    second = ingestor.ingest(pdf, owner_id=2)
    second_ids = set(store.docs) - first_ids

    assert first["status"] == second["status"] == "ingested"
    assert len(first_ids) == first["chunks"]
    assert len(second_ids) == second["chunks"]
    assert all(store.docs[i][1]["owner_id"] == 1 for i in first_ids)
    assert all(store.docs[i][1]["owner_id"] == 2 for i in second_ids)
    assert store.add_calls == 2

    assert ingestor.ingest(pdf, owner_id=1)["status"] == "already_ingested"
    assert ingestor.ingest(pdf, owner_id=2)["status"] == "already_ingested"
    assert store.add_calls == 2


def test_replacement_preserves_other_owners_chunks(
    ingestor, store, tmp_path
) -> None:
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE])
    ingestor.ingest(pdf, owner_id=1)
    ingestor.ingest(pdf, owner_id=2)

    owner_one_ids = {
        doc_id for doc_id, (_, meta) in store.docs.items()
        if meta["owner_id"] == 1
    }
    owner_two_docs = {
        doc_id: (text, dict(meta))
        for doc_id, (text, meta) in store.docs.items()
        if meta["owner_id"] == 2
    }

    _write_pdf(pdf, ["replacement content about clustering. " * 30])
    result = ingestor.ingest(pdf, owner_id=1)

    assert result["status"] == "ingested"
    assert owner_one_ids.isdisjoint(store.docs)
    assert {
        doc_id: value
        for doc_id, value in store.docs.items()
        if value[1]["owner_id"] == 2
    } == owner_two_docs
    assert sum(
        meta["owner_id"] == 1 for _, meta in store.docs.values()
    ) == result["chunks"]

def test_list_documents_is_scoped_to_owner(ingestor, store, tmp_path):
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE, LONG_PAGE])
    first = ingestor.ingest(pdf, owner_id=1)
    ingestor.ingest(pdf, owner_id=2)
    file_hash = next(iter(store.docs.values()))[1]["file_hash"]
    store.docs["legacy"] = (
        "unowned content",
        {"source": "legacy.pdf", "file_hash": "f" * 64, "page": 0},
    )

    assert ingestor.list_documents(owner_id=1) == [
        {
            "document_id": file_hash,
            "filename": "notes.pdf",
            "pages": 2,
            "chunks": first["chunks"],
        }
    ]
    assert ingestor.list_documents(owner_id=3) == []


def test_delete_document_preserves_identical_content_for_other_owner(
    ingestor, store, tmp_path
):
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE])
    first = ingestor.ingest(pdf, owner_id=1)
    ingestor.ingest(pdf, owner_id=2)
    file_hash = next(iter(store.docs.values()))[1]["file_hash"]
    other_docs = {
        key: (text, dict(meta))
        for key, (text, meta) in store.docs.items()
        if meta["owner_id"] == 2
    }

    result = ingestor.delete_document(file_hash, owner_id=1)

    assert result == {
        "document_id": file_hash,
        "chunks": first["chunks"],
        "filenames": ["notes.pdf"],
    }
    assert store.docs == other_docs
    assert ingestor.list_documents(owner_id=1) == []
    assert len(ingestor.list_documents(owner_id=2)) == 1


def test_delete_foreign_document_is_not_found_and_changes_nothing(
    ingestor, store, tmp_path
):
    pdf = _write_pdf(tmp_path / "notes.pdf", [LONG_PAGE])
    ingestor.ingest(pdf, owner_id=2)
    file_hash = next(iter(store.docs.values()))[1]["file_hash"]
    before = {
        key: (text, dict(meta)) for key, (text, meta) in store.docs.items()
    }

    with pytest.raises(DocumentNotFoundError, match="Document not found"):
        ingestor.delete_document(file_hash, owner_id=1)

    assert store.docs == before
