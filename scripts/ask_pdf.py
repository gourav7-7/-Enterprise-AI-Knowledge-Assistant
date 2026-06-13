from __future__ import annotations
import sys
from pathlib import Path

from app.core.exception import AppError
from app.rag.chain import RAGChain
from app.rag.ingestion import DocIngestor

def _format_page(page) -> str:
    if isinstance(page, int):
        return f"p.{page + 1}"
    return "p.?"

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ask_pdf.py <path-to-pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    try:
        ingestor = DocIngestor()
        ingestor.reset()
        print("Ingesting PDF...")
        summary = ingestor.ingest(pdf_path)
        print(
            f"Ingested '{summary['file']}': "
            f"{summary['pages']} pages -> {summary['chunks']} chunks\n"
        )
    except AppError as e:
        print(f"Ingestion failed : {e.message}")
        sys.exit(1)

    chain = RAGChain()
    print("Ask question about the document. Type /quit to exit\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye, tata, gaya, khatam, Goodbye")
            break

        if not question:
            continue
        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye, tata, gaya, khatam, Goodbye")
            break

        res = chain.answer(question)
        print(f"\nAnswer: {res['answer']}\n")

        if res["sources"]:
            print("sources: ")
            seen = set()
            for s in res["sources"]:
                tag = (s["source"], s["page"])
                if tag in seen:
                    continue
                seen.add(tag)
                print(f" - {s['source']} {_format_page(s['page'])}")
            print()

if __name__ == "__main__":
    main()