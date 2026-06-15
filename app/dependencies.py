from __future__ import annotations
 
from functools import lru_cache
 
from app.db.feedback_store import FeedbackStore
from app.rag.chain import RAGChain
from app.rag.ingestion import DocIngestor

@lru_cache
def get_rag_chain() -> RAGChain:
    return RAGChain()

@lru_cache
def get_ingestor() -> DocIngestor:
    return DocIngestor()

@lru_cache
def get_feedback_store() -> FeedbackStore:
    return FeedbackStore()