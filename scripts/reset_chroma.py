"""
Reset the Chroma vector database.

Run:

    python scripts/reset_chroma.py

This utility safely deletes the persisted Chroma database so that it can
be rebuilt from scratch using the ingestion pipeline.

Typical workflow:

    python scripts/reset_chroma.py
    python scripts/ingest.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.config import get_settings

LINE = "=" * 80


def get_directory_size(directory: Path) -> float:
    """Return directory size in MB."""

    if not directory.exists():
        return 0.0

    total = sum(
        file.stat().st_size
        for file in directory.rglob("*")
        if file.is_file()
    )

    return total / (1024 * 1024)


def main() -> None:

    settings = get_settings()

    chroma_dir = Path(settings.chroma_dir)

    print(LINE)
    print("Chroma Reset Utility")
    print(LINE)

    print(f"Collection : {settings.collection_name}")
    print(f"Directory  : {chroma_dir.resolve()}")

    if not chroma_dir.exists():
        print("\nChroma directory does not exist.")
        print("Nothing to delete.")
        return

    size = get_directory_size(chroma_dir)

    print(f"Size       : {size:.2f} MB")

    confirmation = input(
        "\nDelete the Chroma database? (yes/no): "
    ).strip().lower()

    if confirmation != "yes":
        print("\nOperation cancelled.")
        return

    try:

        shutil.rmtree(chroma_dir)

        chroma_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    except Exception as exc:

        print(f"\nError: {exc}")
        return

    print("\nChroma database successfully reset.")
    print("You can now rebuild embeddings using:")
    print("\n    python scripts/ingest.py")


if __name__ == "__main__":
    main()