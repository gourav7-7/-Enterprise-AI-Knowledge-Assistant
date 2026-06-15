"""POST /query : ask a question, get a grounded answer with sources.
 
Async endpoint : awaits the LLM call so the event loop stays free during the
network round-trip.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends
from app.dependencies import get_rag_chain
from app.rag.chain import RAGChain
from app.schemas.query import QueryResponse, QueryRequest

router = APIRouter(tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, chain: RAGChain = Depends(get_rag_chain)) -> QueryResponse:
    result = await chain.aanswer(payload.question)
    return QueryResponse(answer=result["answer"], sources=result["sources"])