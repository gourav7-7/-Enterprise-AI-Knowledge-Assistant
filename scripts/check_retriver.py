"""
Debug utility for the Retriever.

Run:

    python scripts/check_retriever.py

This script lets you inspect exactly what chunks are retrieved from Chroma
before they are sent to the LLM.

Useful for debugging:
- Low RAGAS Context Precision
- Wrong answers
- Missing context
- Chunking issues
"""

from __future__ import annotations

from app.config import get_settings
from app.rag.retriever import Retriever

LINE = "=" * 80
SUBLINE = "-" * 80


def print_document(index: int, doc) -> None:
    """Pretty-print a retrieved document."""

    print(f"\n{sub_header(index)}")

    print(f"Source : {doc.metadata.get('source', 'Unknown')}")
    print(f"Page   : {doc.metadata.get('page', 'N/A')}")

    if "score" in doc.metadata:
        print(f"Score  : {doc.metadata['score']:.4f}")

    print("\nContent\n")
    print(doc.page_content.strip())

    print("\n" + SUBLINE)


def sub_header(index: int) -> str:
    return f"{'-' * 30} Chunk {index} {'-' * 30}"


def main() -> None:
    settings = get_settings()

    retriever = Retriever(
        settings,
        top_k=settings.chat.top_k,
    )

    print(LINE)
    print("Retriever Debug Utility")
    print(LINE)

    print(f"Collection : {settings.collection_name}")
    print(f"Top K      : {settings.chat.top_k}")
    print(f"Embedding  : {settings.embedding_model}")
    print(f"Search     : MMR")
    print(LINE)

    while True:
        question = input(
            "\nQuestion (type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye.")
            break

        if not question:
            continue

        docs = retriever.retrieve(question)

        print("\n" + LINE)
        print("Question")
        print(LINE)
        print(question)

        print("\n" + LINE)
        print(f"Retrieved {len(docs)} document(s)")
        print(LINE)

        if not docs:
            print("No relevant documents found.\n")
            continue

        for index, doc in enumerate(docs, start=1):
            print_document(index, doc)


if __name__ == "__main__":
    main()