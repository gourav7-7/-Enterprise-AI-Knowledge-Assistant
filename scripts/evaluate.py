#!/usr/bin/env python
"""Run RAGAS over data/eval/qa_pairs.json.

For each {question, ground_truth} pair it runs the REAL retriever (for full
contexts) and the chain (for the answer), then scores faithfulness + answer
relevancy. Costs tokens and takes a few minutes — run it occasionally.

Usage:  python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chain import RAGChain
from app.rag.evaluation import RAGEvaluator
from app.rag.retriever import Retriver

EVAL_FILE = Path("data/eval/qa_pairs.json")


def main() -> None:
    if not EVAL_FILE.exists():
        print(f"Missing {EVAL_FILE}. Create it with your 20 question/ground_truth pairs.")
        sys.exit(1)

    pairs = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    print(f"Evaluating {len(pairs)} pairs...\n")

    retriever = Retriver()
    chain = RAGChain()

    samples = []
    for i, pair in enumerate(pairs, start=1):
        question = pair["question"]
        print(f"[{i}/{len(pairs)}] {question}")
        docs = retriever.retrieve(question)          
        contexts = [d.page_content for d in docs]
        answer = chain.answer(question)["answer"]
        samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": pair.get("ground_truth", ""),
            }
        )

    print("\nScoring with RAGAS (this calls the LLM for each metric)...\n")
    scores = RAGEvaluator().evaluate(samples)
    print("\n=== RAGAS scores ===")
    print(scores)


if __name__ == "__main__":
    main()