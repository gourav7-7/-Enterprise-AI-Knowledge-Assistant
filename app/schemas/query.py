# Request/response models for the /query endpoint.

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000, description="The user's question"
    )
    session_id: int | None = Field(
        None,
        ge=1,
        description="Conversation to continue. Omit to use the legacy single thread.",
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise PydanticCustomError("question_blank", "Question must not be empty.")
        return value


class Source(BaseModel):
    source: str = Field(..., description="Source document filename")
    page: int | None = Field(None, description="0-indexed page number, if known")
    snippet: str = Field(..., description="Short excerpt of the cited chunk")


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    session_id: int | None = None