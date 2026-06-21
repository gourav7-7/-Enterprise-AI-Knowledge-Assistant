from __future__ import annotations
 
from functools import lru_cache

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.exception import AuthError
from app.core.security import decode_access_token
from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.rag.chain import RAGChain
from app.rag.ingestion import DocIngestor

@lru_cache
def get_rag_chain() -> RAGChain:
    return RAGChain()

@lru_cache
def get_ingestor() -> DocIngestor:
    return DocIngestor()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user(token: str = Depends(oauth2_scheme),db: Session = Depends(get_db)) -> User:
    user_id = decode_access_token(token)  # raises AuthError if missing/invalid/expired
    user = crud.get_user_by_id(db, int(user_id))
    if user is None:
        raise AuthError("User not found")
    return user