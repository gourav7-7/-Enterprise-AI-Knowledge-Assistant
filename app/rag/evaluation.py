
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.rag import _vertexai_shim
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from app.config import Settings, get_settings
from app.core.logger import get_logger
from app.rag.chain import ConversationalRAGChain

logger = get_logger(__name__)


class RAGEvaluator:
    """Runs RAGAS evaluation against the production ConversationalRAGChain."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        self.chain = ConversationalRAGChain(self.settings,top_k=self.settings.evaluation.top_k)

        self.llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self.settings.openai_model,
                temperature=self.settings.evaluation.temperature,
                api_key=self.settings.openai_api_key,
                max_retries=self.settings.openai_max_retries,
            )
        )

        self.embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.openai_api_key,
            )
        )

        self.metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ]

    def load_dataset(self, dataset_path: str | Path) -> list[dict]:
        dataset_path = Path(dataset_path)
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array.")

        return data

    def prepare_dataset(self, samples: list[dict]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        chat_history: list = []

        for sample in samples:
            result = self.chain._chain.invoke(
                {
                    "input": sample["question"],
                    "chat_history": chat_history,
                }
            )

            documents = result.get("context", [])

            rows.append(
                {
                    "user_input": sample["question"],
                    "response": result["answer"],
                    "retrieved_contexts": [
                        doc.page_content for doc in documents
                    ],
                    "reference": sample["ground_truth"],
                }
            )

        return rows

    def evaluate_dataset(self, dataset_path: str | Path) -> pd.DataFrame:
        logger.info("Loading evaluation dataset from %s", dataset_path)

        qa_pairs = self.load_dataset(dataset_path)
        rows = self.prepare_dataset(qa_pairs)

        dataset = Dataset.from_list(rows)

        logger.info("Running RAGAS evaluation...")
        # for checking ---
        print(self.metrics)
        for metric in self.metrics:
            print(type(metric), getattr(metric, "name", None))
        # ---
        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
        except Exception:
            logger.exception("RAGAS evaluation failed.")
            raise

        logger.info("Evaluation complete.")

        return result.to_pandas()

    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict[str, float]:
        summary: dict[str, float] = {}

        for column in df.select_dtypes(include="number").columns:
            summary[column] = float(df[column].mean())

        return summary

    @staticmethod
    def print_summary(df: pd.DataFrame) -> None:
        print(df.to_string(index=False))
        print("\nAverage Scores\n")

        for metric, value in RAGEvaluator.get_summary(df).items():
            print(f"{metric:<35}{value:.3f}")

    @staticmethod
    def save_csv(df: pd.DataFrame, output_path: str | Path) -> None:
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        logger.info("Saved evaluation results to %s", output_path)
