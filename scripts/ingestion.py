"""Ingest one user's PDFs: python -m scripts.ingestion --owner-id USER_ID."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.core.logger import get_logger
from app.rag.ingestion import DocIngestor

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs for one document owner.")
    parser.add_argument("--owner-id", type=int, required=True, help="Document owner user ID")
    parser.add_argument(
        "--directory",
        type=Path,
        help="PDF directory; defaults to data/uploads/<owner_id>",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Clear the entire collection for ALL users before ingesting",
    )
    args = parser.parse_args()
    if args.owner_id <= 0:
        parser.error("--owner-id must be a positive integer")

    directory = args.directory or Path("data/uploads") / str(args.owner_id)
    if not directory.is_dir():
        parser.error(f"Not a directory: {directory}")
    if not any(directory.glob("*.pdf")):
        parser.error(f"No PDF files found in {directory}")

    settings = get_settings()
    print(f"Owner      : {args.owner_id}")
    print(f"Directory  : {directory.resolve()}")
    print(f"Collection : {settings.collection_name}")

    try:
        ingestor = DocIngestor(settings)
        if args.reset_all:
            ingestor.reset()
        results = ingestor.ingestDirectory(directory, owner_id=args.owner_id)
    except Exception:
        logger.exception("Document ingestion failed.")
        raise

    for item in results:
        print(
            f"{item['file']}: {item['pages']} pages, "
            f"{item['chunks']} chunks ({item['status']})"
        )
    print(f"Processed {len(results)} file(s).")


if __name__ == "__main__":
    main()
