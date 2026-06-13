#!/usr/bin/env python
"""
Task 1 — Run the LangChain LLMChain (PromptTemplate + ChatOpenAI).

Usage (from project root, venv activated):
    python scripts/run_chain.py
    python scripts/run_chain.py "What is RAG?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chains.qa_chain import run_qa_chain


def main() -> None:
    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "In one sentence, what is retrieval-augmented generation (RAG)?"
    )
    context = (
        "RAG combines a retriever that fetches relevant documents with an LLM "
        "that generates answers grounded in those documents."
    )

    print("Question:", question)
    print("Running LangChain LLMChain...\n")
    answer = run_qa_chain(question=question, context=context)
    print("Answer:", answer)
    print("\nLangChain chain completed successfully.")


if __name__ == "__main__":
    main()
