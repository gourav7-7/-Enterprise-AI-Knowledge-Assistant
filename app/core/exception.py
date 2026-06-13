from __future__ import annotations

class AppError(Exception):

    status_code: int = 500
    def __init__(self, message:str) -> None:
        super().__init__(message)
        self.message = message

class ConfigError(AppError):

    status_code= 500

class DocumentNotFoundError(AppError):

    status_code = 404

class IngestionError(AppError):
    status_code = 400

class RetrievalError(AppError):
    status_code = 500                