"""
Run RAGAS evaluation against the production RAG pipeline.

Run:

    python scripts/evaluate.py

This utility:

1. Loads the evaluation dataset.
2. Executes the production Conversational RAG chain.
3. Runs RAGAS evaluation.
4. Prints metric averages.
5. Saves detailed results to CSV.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.core.logger import get_logger
from app.rag.evaluation import RAGEvaluator

logger = get_logger(__name__)

LINE = "=" * 80


def main() -> None:

    dataset_path = Path("data/eval/qa_dataset.json")
    output_path = Path("data/eval/ragas_results.csv")

    print(LINE)
    print("Enterprise AI Knowledge Assistant")
    print("RAGAS Evaluation Utility")
    print(LINE)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path.resolve()}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()

    try:

        evaluator = RAGEvaluator()

        df = evaluator.evaluate_dataset(
            dataset_path
        )

        evaluator.print_summary(df)

        evaluator.save_csv(
            df,
            output_path,
        )

    except Exception:

        logger.exception(
            "Evaluation failed."
        )

        raise

    elapsed = time.perf_counter() - start

    print("\n" + LINE)
    print("Evaluation Complete")
    print(LINE)

    print(f"Dataset : {dataset_path.resolve()}")
    print(f"Results : {output_path.resolve()}")
    print(f"Elapsed : {elapsed:.2f} seconds")

    print("\nDone.")


if __name__ == "__main__":
    main()