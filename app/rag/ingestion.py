from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.core.exception import DocumentNotFoundError, IngestionError
from app.core.logger import get_logger

logger = get_logger(__name__)

_HASH_BLOCK = 1024 * 1024


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def _page_count(metas: list[dict]) -> int:
    first = metas[0] or {}
    if first.get("total_pages"):
        return int(first["total_pages"])
    return max(int((m or {}).get("page", 0)) for m in metas) + 1


class DocIngestor:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        self._embeddings = OpenAIEmbeddings(
            model=self._settings.embedding_model,
            api_key=self._settings.openai_api_key,
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            add_start_index=True,
        )

        self._store = self._build_store()

    def _build_store(self) -> Chroma:
        return Chroma(
            collection_name=self._settings.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._settings.chroma_dir,
        )

    def _find_existing(
        self, file_hash: str, owner_id: int
    ) -> tuple[list[str], list[dict]]:
        got = self._store.get(
            where={"$and": [{"owner_id": owner_id}, {"file_hash": file_hash}]},
            include=["metadatas"],
        )
        return got["ids"], got["metadatas"]

    def ingest(self, file_path: str | Path, *, owner_id: int) -> dict:
        if type(owner_id) is not int or owner_id <= 0:
            raise IngestionError("owner_id must be a positive integer.")

        path = Path(file_path)
        if not path.exists():
            raise DocumentNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise IngestionError(
                f"Unsupported file type '{path.suffix}'. Only .pdf is supported."
            )

        file_hash = _file_sha256(path)

        # Idempotency: identical content that is fully stored is not re-embedded.
        ids, metas = self._find_existing(file_hash, owner_id)
        if ids and len(ids) == (metas[0] or {}).get("n_chunks"):
            logger.info("%s already ingested (%d chunks); skipping.", path.name, len(ids))
            return {
                "file": path.name,
                "pages": _page_count(metas),
                "chunks": len(ids),
                "status": "already_ingested",
            }
        if ids:
            logger.warning(
                "Found %d partial chunks for %s; removing before re-ingest.",
                len(ids),
                path.name,
            )
            self._store.delete(ids=ids)

        logger.info("Loading PDF : %s", path.name)
        pages = PyPDFLoader(str(path)).load()
        if not pages:
            raise IngestionError(f"No extractable text found in {path.name}.")

        chunks = self._splitter.split_documents(pages)
        if not chunks:
            raise IngestionError(f"Splitting produced no chunks for {path.name}.")

        chunk_ids = [
            f"{owner_id}-{file_hash}-{i:05d}" for i in range(len(chunks))
        ]
        for chunk in chunks:
            chunk.metadata["source"] = path.name
            chunk.metadata["owner_id"] = owner_id
            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["n_chunks"] = len(chunks)

        # A re-upload under the same filename replaces the older version.
        self._store.delete(
            where={"$and": [{"owner_id": owner_id}, {"source": path.name}]}
        )

        logger.info("Split into %d chunks; embedding and storing...", len(chunks))
        self._store.add_documents(chunks, ids=chunk_ids)
        logger.info(
            "Stored %d chunks in collection '%s'",
            len(chunks),
            self._settings.collection_name,
        )

        return {
            "file": path.name,
            "pages": len(pages),
            "chunks": len(chunks),
            "status": "ingested",
        }

    def ingestDirectory(self, dir_path: str | Path | None = None, *, owner_id: int) -> list[dict]:
        """Ingest PDFs for one owner; default to data/uploads/<owner_id>."""
        if type(owner_id) is not int or owner_id <= 0:
            raise IngestionError("owner_id must be a positive integer.")

        directory = (
            Path("data/uploads") / str(owner_id)
            if dir_path is None
            else Path(dir_path)
        )

        if not directory.exists():
            raise DocumentNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise DocumentNotFoundError(f"Not a directory: {directory}")

        pdfs = sorted(directory.glob("*.pdf"))
        if not pdfs:
            raise DocumentNotFoundError(f"No PDF files found in {directory}")

        logger.info("Ingesting %d PDF(s) from %s", len(pdfs), directory)
        return [self.ingest(pdf, owner_id=owner_id) for pdf in pdfs]

    def list_documents(self, *, owner_id: int) -> list[dict]:
        """List indexed documents belonging to one owner."""
        if type(owner_id) is not int or owner_id <= 0:
            raise IngestionError("owner_id must be a positive integer.")

        result = self._store.get(
            where={"owner_id": owner_id},
            include=["metadatas"],
        )
        grouped: dict[str, list[dict]] = {}
        for metadata in result["metadatas"] or []:
            if not metadata or metadata.get("owner_id") != owner_id:
                continue
            file_hash = metadata.get("file_hash")
            if not isinstance(file_hash, str) or not file_hash:
                continue
            grouped.setdefault(file_hash, []).append(metadata)

        documents = []
        for file_hash, metas in grouped.items():
            documents.append(
                {
                    "document_id": file_hash,
                    "filename": metas[0].get("source", "unknown"),
                    "pages": _page_count(metas),
                    "chunks": len(metas),
                }
            )
        return sorted(
            documents,
            key=lambda item: (item["filename"], item["document_id"]),
        )

    def delete_document(self, document_id: str, *, owner_id: int) -> dict:
        """Delete one owner's indexed document and return its stored filenames."""
        if type(owner_id) is not int or owner_id <= 0:
            raise IngestionError("owner_id must be a positive integer.")
        if (
            not isinstance(document_id, str)
            or len(document_id) != 64
            or any(char not in "0123456789abcdef" for char in document_id)
        ):
            raise DocumentNotFoundError("Document not found.")

        where = {"$and": [{"owner_id": owner_id}, {"file_hash": document_id}]}
        result = self._store.get(where=where, include=["metadatas"])
        if not result["ids"]:
            raise DocumentNotFoundError("Document not found.")

        filenames = sorted(
            {
                metadata["source"]
                for metadata in result["metadatas"] or []
                if metadata and isinstance(metadata.get("source"), str)
            }
        )
        self._store.delete(where=where)
        return {
            "document_id": document_id,
            "chunks": len(result["ids"]),
            "filenames": filenames,
        }

    def reset(self) -> None:
        logger.warning("Resetting collection '%s'", self._settings.collection_name)
        try:
            self._store.delete_collection()
        except Exception as e:
            logger.warning("delete_collection failed: %s", e)
        self._store = self._build_store()


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a PDF file or a directory of PDFs into the vector store."
    )
    parser.add_argument("path", help="Path to a .pdf file or a directory of PDFs")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the entire collection for ALL users before ingesting",
    )
    parser.add_argument(
        "--owner-id",
        type=int,
        required=True,
        help="Document owner user ID",
    )
    args = parser.parse_args()
    if args.owner_id <= 0:
        parser.error("--owner-id must be a positive integer")

    ingestor = DocIngestor()
    if args.reset:
        ingestor.reset()

    target = Path(args.path)
    if target.is_dir():
        res = ingestor.ingestDirectory(target, owner_id=args.owner_id)
        total = sum(r["chunks"] for r in res)
        print(f"Ingested {len(res)} file(s), {total} chunks total: ")
        for r in res:
            print(f" -{r['file']}: {r['pages']} pages -> {r['chunks']} chunks ({r['status']})")
    else:
        r = ingestor.ingest(target, owner_id=args.owner_id)
        print(f"{r['status']}: {r['file']}: {r['pages']} pages -> {r['chunks']} chunks")


if __name__ == "__main__":
    _cli()
