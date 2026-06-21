from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from app.config import Settings, get_settings
from app.core.logger import get_logger
from app.core.exception import AuthError

logger = get_logger(__name__)

_password_hash = PasswordHash.recommended()

def hash_password(plain: str) -> str:
    return _password_hash.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _password_hash.verify(plain, hashed)

def create_access_token(subject: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

def decode_access_token(token: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:  # base class for all PyJWT decode errors
        raise AuthError("Invalid token") from exc
 
    subject = payload.get("sub")
    if subject is None:
        raise AuthError("Invalid token payload")
    return subject