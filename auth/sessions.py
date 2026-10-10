"""
AI Software Company — Session + CSRF management (Step 3)

Architecture rules (Section 7 + F15 + F17 + F18):
  - Sessions stored in SQLite `sessions` table
  - Cookie: HttpOnly + SameSite=Lax
  - CSRF token: generated at login, stored in sessions table,
    required on every POST form (hidden field)
  - Logout: POST /logout → delete session + clear cookie + audit

Security:
  - Session id and CSRF token generated with `secrets` module
  - No secrets in logs
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session as OrmSession

from db.models import Session as SessionModel

SESSION_COOKIE_NAME = "ai_company_session"
SESSION_LIFETIME_HOURS = 12


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _generate_token(nbytes: int = 32) -> str:
    """URL-safe random token."""
    return secrets.token_urlsafe(nbytes)


def create_session(db: OrmSession, user_id: int) -> SessionModel:
    """
    Create a new session row for the given user.
    Returns the SessionModel (with session_id + csrf_token set).
    """
    session_id = _generate_token(32)
    csrf_token = _generate_token(32)
    expires_at = _now() + timedelta(hours=SESSION_LIFETIME_HOURS)

    s = SessionModel(
        session_id=session_id,
        user_id=user_id,
        csrf_token=csrf_token,
        expires_at=expires_at,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_session(db: OrmSession, session_id: str) -> Optional[SessionModel]:
    """
    Fetch a session by id. Returns None if not found or expired.
    Expired sessions are NOT auto-deleted here (call cleanup_expired_sessions).
    """
    if not session_id:
        return None
    stmt = select(SessionModel).where(SessionModel.session_id == session_id)
    s = db.execute(stmt).scalar_one_or_none()
    if s is None:
        return None
    # Ensure tz-aware comparison works even with naive SQLite datetimes
    exp = s.expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp < _now():
        return None
    return s


def delete_session(db: OrmSession, session_id: str) -> bool:
    """Delete a session by id. Returns True if a row was removed."""
    if not session_id:
        return False
    stmt = delete(SessionModel).where(SessionModel.session_id == session_id)
    result = db.execute(stmt)
    db.commit()
    return result.rowcount > 0


def delete_user_sessions(db: OrmSession, user_id: int) -> int:
    """Delete all sessions for a user (e.g., on password change)."""
    stmt = delete(SessionModel).where(SessionModel.user_id == user_id)
    result = db.execute(stmt)
    db.commit()
    return result.rowcount or 0


def cleanup_expired_sessions(db: OrmSession) -> int:
    """Remove expired sessions. Returns count deleted."""
    now = _now()
    stmt = delete(SessionModel).where(SessionModel.expires_at < now)
    result = db.execute(stmt)
    db.commit()
    return result.rowcount or 0


def check_csrf(session: SessionModel, provided_token: Optional[str]) -> bool:
    """
    Constant-time compare of provided CSRF token against stored token.
    Returns False if session or provided token is missing.
    """
    if session is None or not provided_token:
        return False
    return secrets.compare_digest(session.csrf_token, provided_token)