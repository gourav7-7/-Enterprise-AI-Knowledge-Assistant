# Auth endpoints: register, login (JWT), logout, me

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.exception import AuthError, ConflictError
from app.core.rate_limit import client_ip, login_failure_limiter, register_limiter
from app.core.security import create_access_token, hash_password, verify_password
from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_current_user
from app.schemas.auth import RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the username doesn't exist, so unknown users and wrong
# passwords take the same time (no username enumeration via response timing).
_DUMMY_HASH = hash_password("dummy-password-for-constant-time-check")


@router.post("/register", response_model=UserResponse, status_code=201)
def register(
    payload: RegisterRequest, request: Request, db: Session = Depends(get_db)
) -> UserResponse:
    register_limiter.consume(client_ip(request))
    if crud.get_user_by_username(db, payload.username) is not None:
        raise ConflictError("Username already taken")
    user = crud.create_user(db, payload.username, hash_password(payload.password))
    return UserResponse(id=user.id, username=user.username)


@router.post("/login", response_model=TokenResponse)
def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    key = f"{client_ip(request)}|{form.username.strip().lower()}"
    login_failure_limiter.check(key)

    user = crud.get_user_by_username(db, form.username)
    hashed = user.hashed_password if user is not None else _DUMMY_HASH
    password_ok = verify_password(form.password, hashed)

    if user is None or not password_ok:
        login_failure_limiter.hit(key)
        raise AuthError("Incorrect username or password")

    login_failure_limiter.reset(key)
    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token)


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> dict:
    """Stateless JWT: no server-side session to destroy. "Logout" = client discards
    its token. For true server-side revocation, add a token blocklist (e.g. Redis)
    checked in get_current_user."""

    return {"message": "Logged out. Discard your access token on the client."}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=current_user.id, username=current_user.username)