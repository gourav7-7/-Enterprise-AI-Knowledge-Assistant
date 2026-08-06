"""
Inspect the Chroma vector store.

Run:

    python scripts/inspect_vectorstore.py

Displays useful information about the current vector database.

Useful for verifying:

- Collection name
- Persist directory
- Embedding model
- Retriever configuration
- Number of indexed chunks
"""

from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.rag.retriever import Retriever

LINE = "=" * 80


def main() -> None:

    settings = get_settings()

    retriever = Retriever(
        settings,
        top_k=settings.chat.top_k,
    )

    chroma_dir = Path(settings.chroma_dir)

    if not chroma_dir.exists():
        print(LINE)
        print("Vector Store Inspection")
        print(LINE)
        print("❌ Chroma directory does not exist.")
        print(f"Expected location:\n{chroma_dir.resolve()}")
        print("\nRun:")
        print("    python scripts/ingest.py")
        return

    retriever = Retriever(
        settings,
        top_k=settings.chat.top_k,
    )

    collection = retriever._store._collection

    print(LINE)
    print("Vector Store Inspection")
    print(LINE)

    print(f"Collection Name   : {settings.collection_name}")
    print(f"Persist Directory : {Path(settings.chroma_dir).resolve()}")
    print(f"Embedding Model   : {settings.embedding_model}")

    print("\nRetriever Configuration")

    print(f"Search Type       : MMR")
    print(f"Top K             : {settings.chat.top_k}")
    print(
        f"Fetch K           : {max(20, settings.chat.top_k * 4)}"
    )
    print("Lambda Mult       : 0.40")

    print("\nCollection Statistics")

    try:

        count = collection.count()

        print(f"Indexed Chunks    : {count}")

    except Exception as exc:

        print(f"Unable to read collection statistics.")
        print(exc)

    print("\nStatus")

    if collection.count() == 0:
        print("❌ Collection is empty.")
        print("Run:")
        print("\n    python scripts/ingest.py\n")
    else:
        print("✅ Collection looks healthy.")

    print(LINE)


if __name__ == "__main__":
    main()