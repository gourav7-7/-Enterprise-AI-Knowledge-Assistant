"""Tests for POST /upload (app/api/documents.py).

The ingestor is mocked (no embeddings, no Chroma) and uploads go to a temp dir.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import documents as documents_module
from app.api.documents import router
from app.core.exception import IngestionError, register_exception_handlers
from app.dependencies import get_current_user, get_ingestor

PDF_BYTES = b"%PDF-1.4\n% minimal, correctly-headed fake pdf\n"


@pytest.fixture
def ingestor() -> MagicMock:
    mock = MagicMock()
    mock.ingest.return_value = {
        "file": "notes.pdf",
        "pages": 2,
        "chunks": 5,
        "status": "ingested",
    }
    return mock


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    path = tmp_path / "uploads"
    monkeypatch.setattr(documents_module, "UPLOAD_DIR", path)
    return path / "1"


def _client(
    ingestor: MagicMock, authenticated: bool = True, *, user_id: int = 1
) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_ingestor] = lambda: ingestor
    if authenticated:
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=user_id)
    return TestClient(app)


def _post(client: TestClient, filename: str, content: bytes):
    return client.post(
        "/upload", files={"file": (filename, content, "application/pdf")}
    )


def test_upload_requires_authentication(ingestor, upload_dir) -> None:
    client = _client(ingestor, authenticated=False)

    response = _post(client, "notes.pdf", PDF_BYTES)

    assert response.status_code == 401
    ingestor.ingest.assert_not_called()


def test_upload_valid_pdf_is_saved_and_ingested(ingestor, upload_dir) -> None:
    response = _post(_client(ingestor), "notes.pdf", PDF_BYTES)

    assert response.status_code == 200
    assert response.json() == {
        "filename": "notes.pdf",
        "pages": 2,
        "chunks": 5,
        "status": "ingested",
    }
    assert (upload_dir / "notes.pdf").read_bytes() == PDF_BYTES
    ingestor.ingest.assert_called_once_with(upload_dir / "notes.pdf", owner_id=1)


def test_upload_reports_already_ingested_status(ingestor, upload_dir) -> None:
    ingestor.ingest.return_value = {
        "file": "notes.pdf",
        "pages": 2,
        "chunks": 5,
        "status": "already_ingested",
    }

    response = _post(_client(ingestor), "notes.pdf", PDF_BYTES)

    assert response.status_code == 200
    assert response.json()["status"] == "already_ingested"


def test_upload_rejects_non_pdf_extension(ingestor, upload_dir) -> None:
    response = _post(_client(ingestor), "notes.txt", PDF_BYTES)

    assert response.status_code == 400
    ingestor.ingest.assert_not_called()


def test_upload_rejects_file_without_pdf_header(ingestor, upload_dir) -> None:
    response = _post(_client(ingestor), "fake.pdf", b"hello, I am not a pdf")

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["error"]["message"]
    ingestor.ingest.assert_not_called()
    assert list(upload_dir.iterdir()) == []  # nothing left behind


def test_upload_rejects_empty_file(ingestor, upload_dir) -> None:
    response = _post(_client(ingestor), "empty.pdf", b"")

    assert response.status_code == 400
    ingestor.ingest.assert_not_called()
    assert list(upload_dir.iterdir()) == []


def test_upload_rejects_oversized_file(ingestor, upload_dir, monkeypatch) -> None:
    monkeypatch.setattr(documents_module, "MAX_UPLOAD_BYTES", 100)

    response = _post(_client(ingestor), "big.pdf", PDF_BYTES + b"x" * 500)

    assert response.status_code == 400
    assert "limit" in response.json()["error"]["message"]
    ingestor.ingest.assert_not_called()
    assert list(upload_dir.iterdir()) == []


@pytest.mark.parametrize("evil_name", ["../../evil.pdf", "..\\..\\evil.pdf", "/etc/evil.pdf"])
def test_upload_strips_directory_parts_from_filename(
    ingestor, upload_dir, tmp_path, evil_name
) -> None:
    response = _post(_client(ingestor), evil_name, PDF_BYTES)

    assert response.status_code == 200
    assert (upload_dir / "evil.pdf").exists()
    assert not (tmp_path / "evil.pdf").exists()  # did not escape the upload dir
    ingestor.ingest.assert_called_once_with(upload_dir / "evil.pdf", owner_id=1)


def test_upload_removes_new_file_when_ingestion_fails(ingestor, upload_dir) -> None:
    ingestor.ingest.side_effect = IngestionError("No extractable text found.")

    response = _post(_client(ingestor), "scan.pdf", PDF_BYTES)

    assert response.status_code == 400
    assert not (upload_dir / "scan.pdf").exists()


def test_upload_keeps_existing_file_when_reingestion_fails(ingestor, upload_dir) -> None:
    upload_dir.mkdir(parents=True)
    (upload_dir / "notes.pdf").write_bytes(b"%PDF-old")
    ingestor.ingest.side_effect = IngestionError("boom")

    response = _post(_client(ingestor), "notes.pdf", PDF_BYTES)

    assert response.status_code == 400
    assert (upload_dir / "notes.pdf").exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("notes.pdf", "notes.pdf"),
        ("../../evil.pdf", "evil.pdf"),
        ("..\\..\\evil.pdf", "evil.pdf"),
        ("/etc/passwd.pdf", "passwd.pdf"),
        ("my report (final).pdf", "my report _final_.pdf"),
        ("UPPER.PDF", "UPPER.PDF"),
    ],
)
def test_safe_filename_sanitizes(raw: str, expected: str) -> None:
    assert documents_module._safe_filename(raw) == expected


@pytest.mark.parametrize("raw", [None, "", ".pdf", "..", "notes.txt", "notes.pdf.exe"])
def test_safe_filename_rejects_invalid_names(raw) -> None:
    with pytest.raises(IngestionError):
        documents_module._safe_filename(raw)

def test_same_filename_is_isolated_between_users(ingestor, upload_dir) -> None:
    first_bytes = PDF_BYTES + b"user-one"
    second_bytes = PDF_BYTES + b"user-two"

    first = _post(_client(ingestor, user_id=1), "notes.pdf", first_bytes)
    second = _post(_client(ingestor, user_id=2), "notes.pdf", second_bytes)

    assert first.status_code == 200
    assert second.status_code == 200

    first_path = upload_dir / "notes.pdf"
    second_path = upload_dir.parent / "2" / "notes.pdf"

    assert first_path.read_bytes() == first_bytes
    assert second_path.read_bytes() == second_bytes
    assert ingestor.ingest.call_count == 2
    ingestor.ingest.assert_any_call(first_path, owner_id=1)
    ingestor.ingest.assert_any_call(second_path, owner_id=2)
