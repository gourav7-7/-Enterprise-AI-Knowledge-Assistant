"""
Rebuild the Chroma vector database.

Run:

    python scripts/ingest.py

Optional:

    python scripts/ingest.py --reset

The script:

1. Optionally clears the existing Chroma database.
2. Loads all supported documents.
3. Chunks the documents.
4. Generates embeddings.
5. Stores them in Chroma.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.config import get_settings
from app.core.logger import get_logger
from app.rag.ingestion import DocIngestor

logger = get_logger(__name__)

LINE = "=" * 80


def reset_chroma(chroma_dir: Path) -> None:
    """Delete the persisted Chroma database."""

    if not chroma_dir.exists():
        return

    shutil.rmtree(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Rebuild the Chroma vector database."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma database before ingestion.",
    )

    args = parser.parse_args()

    settings = get_settings()

    chroma_dir = Path(settings.chroma_dir)

    print(LINE)
    print("Enterprise AI Knowledge Assistant")
    print("Document Ingestion Utility")
    print(LINE)

    print(f"Collection : {settings.collection_name}")
    print(f"Directory  : {chroma_dir.resolve()}")
    print(f"Chunk Size : {settings.chunk_size}")
    print(f"Overlap    : {settings.chunk_overlap}")
    print(f"Embedding  : {settings.embedding_model}")

    if args.reset:
        print("\nResetting existing vector database...")
        reset_chroma(chroma_dir)

    try:

        ingestor = DocIngestor(settings)

        results = ingestor.ingestDirectory("data/uploads")

        total_chunks = sum(r["chunks"] for r in results)

        print()
        print(f"Ingested Files : {len(results)}")
        print(f"Total Chunks   : {total_chunks}")

        for item in results:
            print(
                f"  • {item['file']} "
                f"({item['pages']} pages → {item['chunks']} chunks)"
            )

        # Verify vector store contents
        collection = ingestor._store.get()

        print()
        print(f"Indexed Chunks : {len(collection['ids'])}")

    except Exception:

        logger.exception("Document ingestion failed.")
        raise

    print("\n" + LINE)
    print("Document ingestion completed successfully.")
    print(LINE)

    print("\nNext steps:")

    print("1. python scripts/inspect_vectorstore.py")
    print("2. python scripts/check_retriever.py")
    print("3. python scripts/check_chain.py")
    print("4. python scripts/evaluate.py")


if __name__ == "__main__":
    main()