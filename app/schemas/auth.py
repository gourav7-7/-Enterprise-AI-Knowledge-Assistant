from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

_USERNAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Small static blocklist (only entries >= 10 chars matter, the minimum length).
# For stronger protection use zxcvbn or the HIBP k-anonymity API.
_COMMON_PASSWORDS = frozenset(
    {
        "1234567890", "0123456789", "12345678910", "123456789a", "a123456789",
        "1234512345", "qwertyuiop", "qwerty1234", "qwerty12345", "qwerty123456",
        "password12", "password123", "password1234", "password12345",
        "passw0rd123", "p@ssw0rd123", "p@ssword123", "iloveyou123",
        "welcome123", "welcome1234", "letmein1234", "admin123456",
        "administrator", "changeme123", "abcdefghij", "abcd123456",
        "abc1234567", "1q2w3e4r5t", "1qaz2wsx3edc", "qazwsxedc123",
        "testtest123", "test1234567", "trustno1234", "monkey12345",
        "dragon12345", "football123", "baseball123", "superman123",
        "sunshine123", "princess123",
    }
)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=10, max_length=128)

    @field_validator("username", mode="before")
    @classmethod
    def _strip_username(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("username")
    @classmethod
    def _check_username(cls, value: str) -> str:
        if not _USERNAME_RE.fullmatch(value):
            raise PydanticCustomError(
                "username_chars",
                "Username may only contain letters, numbers, '.', '_' and '-'.",
            )
        return value

    @model_validator(mode="after")
    def _check_password(self) -> "RegisterRequest":
        lowered = self.password.lower()
        if lowered in _COMMON_PASSWORDS or len(set(self.password)) < 4:
            raise PydanticCustomError(
                "password_weak",
                "Password is too common or too repetitive. Choose a less guessable one.",
            )
        if self.username.lower() in lowered:
            raise PydanticCustomError(
                "password_username",
                "Password must not contain your username.",
            )
        return self


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str