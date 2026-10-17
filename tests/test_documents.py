"""HTTP tests for owner-scoped document listing."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import documents as documents_module
from app.api.documents import router
from app.core.exception import DocumentNotFoundError, register_exception_handlers
from app.dependencies import get_current_user, get_ingestor


@pytest.fixture
def ingestor():
    mock = MagicMock()
    mock.list_documents.return_value = []
    return mock


def _client(ingestor, *, owner_id=None):
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_ingestor] = lambda: ingestor
    if owner_id is not None:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=owner_id)
    return TestClient(app)


def test_document_listing_requires_authentication(ingestor):
    with _client(ingestor) as client:
        response = client.get("/documents")
    assert response.status_code == 401
    ingestor.list_documents.assert_not_called()


def test_document_listing_returns_empty_list(ingestor):
    with _client(ingestor, owner_id=1) as client:
        response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []
    ingestor.list_documents.assert_called_once_with(owner_id=1)


@pytest.mark.parametrize("owner_id", [1, 2])
def test_document_listing_uses_authenticated_owner(ingestor, owner_id):
    documents = [
        {
            "document_id": "a" * 64,
            "filename": "notes.pdf",
            "pages": 2,
            "chunks": 5,
        }
    ]
    ingestor.list_documents.return_value = documents
    with _client(ingestor, owner_id=owner_id) as client:
        response = client.get("/documents", params={"owner_id": 999})
    assert response.status_code == 200
    assert response.json() == documents
    ingestor.list_documents.assert_called_once_with(owner_id=owner_id)

def test_document_deletion_requires_authentication(ingestor):
    with _client(ingestor) as client:
        response = client.delete(f"/documents/{'a' * 64}")
    assert response.status_code == 401
    ingestor.delete_document.assert_not_called()


def test_document_deletion_removes_only_matching_owner_files(
    ingestor, tmp_path, monkeypatch
):
    monkeypatch.setattr(documents_module, "UPLOAD_DIR", tmp_path)
    owner_dir = tmp_path / "1"
    other_dir = tmp_path / "2"
    owner_dir.mkdir()
    other_dir.mkdir()
    content = b"%PDF-1.4 same content"
    document_id = hashlib.sha256(content).hexdigest()
    for name in ("notes.pdf", "copy.PDF"):
        (owner_dir / name).write_bytes(content)
    (owner_dir / "keep.pdf").write_bytes(b"different content")
    (other_dir / "notes.pdf").write_bytes(content)
    ingestor.delete_document.return_value = {"document_id": document_id, "chunks": 3}

    with _client(ingestor, owner_id=1) as client:
        response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "chunks_deleted": 3,
        "files_deleted": ["copy.PDF", "notes.pdf"],
    }
    ingestor.delete_document.assert_called_once_with(document_id, owner_id=1)
    assert not (owner_dir / "notes.pdf").exists()
    assert not (owner_dir / "copy.PDF").exists()
    assert (owner_dir / "keep.pdf").read_bytes() == b"different content"
    assert (other_dir / "notes.pdf").read_bytes() == content


def test_document_deletion_not_found_preserves_files(ingestor, tmp_path, monkeypatch):
    monkeypatch.setattr(documents_module, "UPLOAD_DIR", tmp_path)
    owner_dir = tmp_path / "1"
    owner_dir.mkdir()
    content = b"%PDF-1.4 content"
    path = owner_dir / "notes.pdf"
    path.write_bytes(content)
    document_id = hashlib.sha256(content).hexdigest()
    ingestor.delete_document.side_effect = DocumentNotFoundError("Document not found.")

    with _client(ingestor, owner_id=1) as client:
        response = client.delete(f"/documents/{document_id}", params={"owner_id": 2})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"
    ingestor.delete_document.assert_called_once_with(document_id, owner_id=1)
    assert path.read_bytes() == content


def test_document_deletion_handles_missing_upload_folder(ingestor, tmp_path, monkeypatch):
    monkeypatch.setattr(documents_module, "UPLOAD_DIR", tmp_path)
    document_id = "a" * 64
    ingestor.delete_document.return_value = {"document_id": document_id, "chunks": 2}

    with _client(ingestor, owner_id=1) as client:
        response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["chunks_deleted"] == 2
    assert response.json()["files_deleted"] == []


def test_document_deletion_preserves_changed_file(ingestor, tmp_path, monkeypatch):
    monkeypatch.setattr(documents_module, "UPLOAD_DIR", tmp_path)
    owner_dir = tmp_path / "1"
    owner_dir.mkdir()
    path = owner_dir / "notes.pdf"
    path.write_bytes(b"old content")
    document_id = hashlib.sha256(b"old content").hexdigest()

    def delete_chunks(requested_id, *, owner_id):
        assert requested_id == document_id
        assert owner_id == 1
        path.write_bytes(b"new content")
        return {"document_id": document_id, "chunks": 1}

    ingestor.delete_document.side_effect = delete_chunks
    with _client(ingestor, owner_id=1) as client:
        response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["files_deleted"] == []
    assert path.read_bytes() == b"new content"
