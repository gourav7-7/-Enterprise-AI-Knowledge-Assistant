"""POST /query : ask a question, get a grounded answer with sources.
 
Async endpoint : awaits the LLM call so the event loop stays free during the
network round-trip.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import crud
from app.db.database import get_db
from app.db.models import ChatHistory, User
from app.dependencies import get_conversational_rag_chain, get_current_user
from app.rag.chain import ConversationalRAGChain
from app.schemas.query import QueryResponse, QueryRequest

router = APIRouter(tags=["query"])

def _to_chat_history(records: list[ChatHistory]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for record in records:
        messages.append(HumanMessage(content = record.question))
        messages.append(AIMessage(content=record.answer))
    return messages

@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest,current_user: User = Depends(get_current_user),
                chain: ConversationalRAGChain = Depends(get_conversational_rag_chain),db: Session = Depends(get_db)) -> QueryResponse:
    settings = get_settings()
    recent_records = await run_in_threadpool(
        crud.get_recent_chat_history,
        db,
        current_user.id,
        settings.conversation_history_turns
    )
    chat_history = _to_chat_history(recent_records)

    result = await chain.aanswer(payload.question, chat_history = chat_history)

    await run_in_threadpool(
        crud.save_chat,
        db,
        current_user.id,
        payload.question,
        result["answer"],
        result["sources"],
    )

    return QueryResponse(answer=result["answer"], sources=result["sources"])