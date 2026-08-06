"""
Debug utility for the complete Conversational RAG pipeline.

Run:

    python scripts/check_chain.py

This utility executes the production ConversationalRAGChain without
running FastAPI or Streamlit.

Useful for debugging:
- Retriever quality
- Prompt behaviour
- LLM answers
- Source snippets
- Conversation history
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from app.config import get_settings
from app.rag.chain import ConversationalRAGChain

LINE = "=" * 80
SUBLINE = "-" * 80


def print_sources(sources: list[dict]) -> None:
    """Pretty-print retrieved source documents."""

    if not sources:
        print("No sources returned.")
        return

    for index, source in enumerate(sources, start=1):

        print(f"\n{sub_header(index)}")

        print(
            f"Source : {source.get('source', 'Unknown')}"
        )

        print(
            f"Page   : {source.get('page', 'N/A')}"
        )

        print("\nSnippet\n")

        print(source.get("snippet", ""))

        print("\n" + SUBLINE)


def sub_header(index: int) -> str:
    return f"{'-' * 30} Source {index} {'-' * 30}"


async def interactive_chat() -> None:
    """Run an interactive chat session using the production RAG chain."""

    settings = get_settings()

    chain = ConversationalRAGChain(settings)

    chat_history = []

    print(LINE)
    print("Enterprise AI Knowledge Assistant")
    print("Conversation Debug Utility")
    print(LINE)

    print(f"Model       : {settings.openai_model}")
    print(f"Temperature : {settings.chat.temperature}")
    print(f"Top K       : {settings.chat.top_k}")
    print(f"History     : Enabled")
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

        print("\nGenerating answer...\n")

        try:

            result = await chain.aanswer(
                question,
                chat_history=chat_history,
            )

        except Exception as exc:

            print(f"\nError: {exc}")
            continue

        print(LINE)
        print("Answer")
        print(LINE)

        print(result["answer"])

        print("\n" + LINE)
        print("Retrieved Sources")
        print(LINE)

        print_sources(result["sources"])

        #
        # Maintain conversation history exactly like production.
        #
        chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=result["answer"]),
            ]
        )


def main() -> None:
    asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()