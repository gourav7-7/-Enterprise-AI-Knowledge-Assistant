# RAGAS evaluation — faithfulness + answer relevancy.

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy

from app.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        # temperature=0 for deterministic judging
        self._eval_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=self._settings.openai_model,
                temperature=0,
                api_key=self._settings.openai_api_key,
            )
        )
        self._eval_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self._settings.embedding_model,
                api_key=self._settings.openai_api_key,
            )
        )

    def evaluate(self, samples: list[dict]) -> dict:
        """samples: list of {question, answer, contexts: list[str], ground_truth}."""
        eval_samples = [
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s.get("ground_truth", ""),
            )
            for s in samples
        ]
        dataset = EvaluationDataset(samples=eval_samples)

        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy()],
            llm=self._eval_llm,
            embeddings=self._eval_embeddings,
        )
        logger.info("RAGAS evaluation complete: %s", result)
        return result