"""
Frozen auth module (template). Provided by platform, not AI-generated.
"""

import secrets
from typing import Optional

from passlib.context import CryptContext
import bcrypt

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    if not plain_password:
        raise ValueError("Password cannot be empty")
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not plain_password or not hashed_password:
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def validate_csrf_token(request: Request, token: str) -> None:
    session_token = getattr(request, "session", {}).get("csrf_token") if hasattr(request, "session") else None
    if not session_token or not secrets.compare_digest(session_token, token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


class TokenData(BaseModel):
    username: Optional[str] = None
    exp: Optional[int] = None


def get_current_user(token: str = Depends(oauth2_scheme)):
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def bcrypt_hash_password(plain_password: str) -> str:
    if isinstance(plain_password, str):
        plain_password = plain_password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password, salt)
    return hashed.decode("utf-8")
