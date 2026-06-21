"""POST /query : ask a question, get a grounded answer with sources.
 
Async endpoint : awaits the LLM call so the event loop stays free during the
network round-trip.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_rag_chain, get_current_user
from app.rag.chain import RAGChain
from app.schemas.query import QueryResponse, QueryRequest

router = APIRouter(tags=["query"])

@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest,current_user: User = Depends(get_current_user),
                chain: RAGChain = Depends(get_rag_chain),db: Session = Depends(get_db)) -> QueryResponse:
    result = await chain.aanswer(payload.question)
 
    await run_in_threadpool(
        crud.save_chat,
        db,
        current_user.id,
        payload.question,
        result["answer"],
        result["sources"],
    )
 
    return QueryResponse(answer=result["answer"], sources=result["sources"])