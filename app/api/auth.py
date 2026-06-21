"""Auth endpoints: register, login (JWT), logout, me."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.exception import AuthError, ConflictError
from app.core.security import create_access_token, hash_password, verify_password
from app.db import crud
from app.db.database import get_db
from app.db.models import User
from app.dependencies import get_current_user
from app.schemas.auth import RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> UserResponse:
    if crud.get_user_by_username(db, payload.username) is not None:
        raise ConflictError("Username already taken")
    user = crud.create_user(db, payload.username, hash_password(payload.password))
    return UserResponse(id=user.id, username=user.username)


@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(),db: Session = Depends(get_db)) -> TokenResponse:
    user = crud.get_user_by_username(db, form.username)
    if user is None or not verify_password(form.password, user.hashed_password):
        raise AuthError("Incorrect username or password")
    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token)


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> dict:
    ''' Stateless JWT: no server-side session to destroy. "Logout" = client discards
        its token. For true server-side revocation, add a token blocklist (e.g. Redis)
        checked in get_current_user.'''
    
    return {"message": "Logged out. Discard your access token on the client."}


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse(id=current_user.id, username=current_user.username)