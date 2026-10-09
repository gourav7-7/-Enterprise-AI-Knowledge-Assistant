"""POST /query : ask a question, get a grounded answer with sources.

Async endpoint : awaits the LLM call so the event loop stays free during the
network round-trip.

History is scoped to one conversation: pass session_id (from POST /sessions) to
continue it; omit it to use the legacy single thread.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.exception import GenerationError, NotFoundError
from app.db import crud
from app.db.database import get_db
from app.db.models import ChatHistory, User
from app.dependencies import get_conversational_rag_chain, get_current_user
from app.rag.chain import ConversationalRAGChain
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


def _to_chat_history(records: list[ChatHistory]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for record in records:
        messages.append(HumanMessage(content=record.question))
        messages.append(AIMessage(content=record.answer))
    return messages


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    current_user: User = Depends(get_current_user),
    chain: ConversationalRAGChain = Depends(get_conversational_rag_chain),
    db: Session = Depends(get_db),
) -> QueryResponse:
    settings = get_settings()
    session_id = payload.session_id

    # Ownership check: a user can only continue their own conversations.
    if session_id is not None:
        session = await run_in_threadpool(
            crud.get_chat_session, db, session_id, current_user.id
        )
        if session is None:
            raise NotFoundError("Conversation not found.")

    recent_records = await run_in_threadpool(
        crud.get_recent_chat_history,
        db,
        current_user.id,
        settings.conversation_history_turns,
        session_id,
    )
    chat_history = _to_chat_history(recent_records)

    result = await chain.aanswer(payload.question, chat_history=chat_history)

    # Failed generations are surfaced as an error and never persisted, or the
    # apology text would be replayed to the LLM as prior conversation.
    if result.get("error"):
        raise GenerationError(
            "The assistant could not generate an answer right now. Please try again."
        )

    await run_in_threadpool(
        crud.save_chat,
        db,
        current_user.id,
        payload.question,
        result["answer"],
        result["sources"],
        session_id,
    )

    return QueryResponse(
        answer=result["answer"], sources=result["sources"], session_id=session_id
    )