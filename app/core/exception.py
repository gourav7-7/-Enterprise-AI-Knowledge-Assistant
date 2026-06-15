# Custom exception hierarchy + FastAPI handlers.


from __future__ import annotations
from typing import Any
from app.core.logger import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    status_code: int = 500
    error_code: str = "app_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "type": self.__class__.__name__,
                "code": self.error_code,
                "message": self.message,
            }
        }
        if self.details:
            body["error"]["details"] = self.details
        return body


class ConfigError(AppError):
    status_code = 500
    error_code = "config_error"


class DocumentNotFoundError(AppError):
    status_code = 404
    error_code = "document_not_found"


class IngestionError(AppError):
    status_code = 400
    error_code = "ingestion_error"


class RetrievalError(AppError):
    status_code = 500
    error_code = "retrieval_error"


def register_exception_handlers(app: Any) -> None:

    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        # Expected error -> log a warning, return its declared status + body.
        logger.warning("%s on %s -> %s", exc.error_code, request.url.path, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # Unexpected bug -> log full traceback, return a generic 500 (no leak).
        logger.exception("Unhandled error on %s", request.url.path)
        generic = AppError("An internal error occurred.")
        generic.error_code = "internal_error"
        return JSONResponse(status_code=500, content=generic.to_dict())