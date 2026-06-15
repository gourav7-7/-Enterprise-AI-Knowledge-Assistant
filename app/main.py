"""FastAPI application entry point : the composition root.
 
Run locally:
    uvicorn app.main:app --reload --port 8000
Then open http://localhost:8000/docs for the interactive API docs.
""" 

from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api import documents, feedback, query
from app.core.exception import register_exception_handlers
from app.core.logger import get_logger
from app.schemas.health import HealthResponse
 
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Knowledge Assistant API")
    yield
    logger.info("Shutting down AI Knowledge Assistant API")
 
 
app = FastAPI(
    title="AI Knowledge Assistant",
    version="0.1.0",
    lifespan=lifespan,
)
 

register_exception_handlers(app)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(feedback.router)

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


