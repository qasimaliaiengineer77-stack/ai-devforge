"""
AI Software Company — Auth routes (Step 3)

Endpoints:
  POST /login   -> verify credentials, create session, set HttpOnly cookie
  POST /logout  -> delete session, clear cookie (F17)
  GET  /me      -> return current user (protected, 401 if not logged in)

Security:
  - Cookie: HttpOnly + SameSite=Lax + (Secure in production)
  - CSRF token returned in login response for the client to use
  - All POSTs require CSRF token (login itself is the exception since
    there's no session yet; but logout requires CSRF)
  - No secrets in error responses (Section 30)
"""

from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Form, Header, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from auth.deps import require_user
from auth.passwords import verify_password
from auth.sessions import (
    SESSION_COOKIE_NAME,
    SESSION_LIFETIME_HOURS,
    check_csrf,
    create_session,
    delete_session,
    get_session,
)
from db.audit import (
    ACTION_CSRF_FAILED,
    ACTION_LOGIN_FAILED,
    ACTION_USER_LOGIN,
    ACTION_USER_LOGOUT,
    log_action,
)
from db.database import get_db
from db.models import User

router = APIRouter(tags=["auth"])


@router.post("/login")
async def login(
    response: Response,
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: OrmSession = Depends(get_db),
):
    """
    Verify credentials and create session.
    Returns CSRF token in JSON body — client must send it on subsequent POSTs.
    Cookie is HttpOnly so JS cannot read it.
    """
    stmt = select(User).where(User.username == username)
    user = db.execute(stmt).scalar_one_or_none()

    if user is None or not verify_password(password, user.password_hash):
        log_action(
            db,
            ACTION_LOGIN_FAILED,
            status="fail",
            message=f"login failed for username={username!r}",
        )
        # Generic error — do not reveal whether user exists
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    s = create_session(db, user.id)

    # Set HttpOnly cookie (SameSite=Lax). Secure=False for local dev (http).
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=s.session_id,
        max_age=SESSION_LIFETIME_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
    )

    log_action(
        db,
        ACTION_USER_LOGIN,
        status="ok",
        user_id=user.id,
        message="login success",
    )

    return {
        "status": "ok",
        "user": {"id": user.id, "username": user.username, "role": user.role},
        "csrf_token": s.csrf_token,
    }


@router.post("/logout")
async def logout(
    response: Response,
    request: Request,
    csrf_token: Optional[str] = Form(default=None),
        x_csrf_token: Optional[str] = Header(default=None, alias="X-CSRF-Token"),
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """
    Delete session, clear cookie. Requires valid session + CSRF token.
    Accepts CSRF either as form field `csrf_token` or header `X-CSRF-Token`.
    """
    s = get_session(db, session_cookie) if session_cookie else None
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    provided = x_csrf_token or csrf_token
    if not check_csrf(s, provided):
        log_action(
            db,
            ACTION_CSRF_FAILED,
            status="fail",
            user_id=s.user_id,
            message="logout csrf mismatch",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF check failed",
        )

    user_id = s.user_id
    delete_session(db, s.session_id)

    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")

    log_action(
        db,
        ACTION_USER_LOGOUT,
        status="ok",
        user_id=user_id,
        message="logout success",
    )

    return {"status": "ok", "message": "logged out"}


@router.get("/me")
async def me(user: User = Depends(require_user)):
    """Return current logged-in user. 401 if not authenticated."""
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
    }