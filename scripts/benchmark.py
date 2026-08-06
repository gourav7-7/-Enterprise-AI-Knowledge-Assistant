"""
Benchmark the RAG pipeline.

Run:

    python scripts/benchmark.py

Measures:
- Retrieval latency
- Generation latency
- Total latency

Useful before deployment and after changing chunking,
retrieval settings, or LLM configuration.
"""

from __future__ import annotations

import asyncio
import statistics
import time

from app.config import get_settings
from app.rag.chain import ConversationalRAGChain
from app.rag.retriever import Retriever

LINE = "=" * 80


async def benchmark_question(
    retriever: Retriever,
    chain: ConversationalRAGChain,
    question: str,
) -> dict:

    start = time.perf_counter()

    retrieval_start = time.perf_counter()
    docs = retriever.retrieve(question)
    retrieval_time = time.perf_counter() - retrieval_start

    generation_start = time.perf_counter()

    result = await chain.aanswer(
        question,
        chat_history=[],
    )

    generation_time = time.perf_counter() - generation_start

    total_time = time.perf_counter() - start

    return {
        "question": question,
        "documents": len(docs),
        "retrieval": retrieval_time,
        "generation": generation_time,
        "total": total_time,
        "answer": result["answer"],
    }


async def main() -> None:

    settings = get_settings()

    retriever = Retriever(
        settings,
        top_k=settings.chat.top_k,
    )

    chain = ConversationalRAGChain(settings)

    print(LINE)
    print("Enterprise AI Knowledge Assistant")
    print("Benchmark Utility")
    print(LINE)

    print(f"Model       : {settings.openai_model}")
    print(f"Temperature : {settings.chat.temperature}")
    print(f"Top K       : {settings.chat.top_k}")

    print(LINE)

    total_times = []

    while True:

        question = input(
            "\nQuestion (type 'exit' to quit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        benchmark = await benchmark_question(
            retriever,
            chain,
            question,
        )

        total_times.append(
            benchmark["total"]
        )

        print("\n" + LINE)
        print("Benchmark Results")
        print(LINE)

        print(
            f"Retrieved Documents : {benchmark['documents']}"
        )

        print(
            f"Retrieval Time      : "
            f"{benchmark['retrieval']:.3f} sec"
        )

        print(
            f"Generation Time     : "
            f"{benchmark['generation']:.3f} sec"
        )

        print(
            f"Total Time          : "
            f"{benchmark['total']:.3f} sec"
        )

        print("\nAnswer\n")
        print(benchmark["answer"])

    if total_times:

        print("\n" + LINE)
        print("Session Summary")
        print(LINE)

        print(
            f"Queries Executed : {len(total_times)}"
        )

        print(
            f"Average Time     : "
            f"{statistics.mean(total_times):.3f} sec"
        )

        print(
            f"Fastest          : "
            f"{min(total_times):.3f} sec"
        )

        print(
            f"Slowest          : "
            f"{max(total_times):.3f} sec"
        )


if __name__ == "__main__":
    asyncio.run(main())