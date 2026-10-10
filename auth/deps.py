"""
AI Software Company — FastAPI auth dependencies (Step 3)

Provides:
  - get_current_user: reads session cookie, returns User or None
  - require_user: raises 401 if not logged in
  - require_role: raises 403 if user role not allowed
  - require_csrf: raises 403 if CSRF token missing/invalid (for POST forms)

Security notes:
  - Cookie is HttpOnly + SameSite=Lax (set in login route)
  - CSRF token is stored in sessions table and compared in constant time
  - No secrets in error responses (Section 30)
"""

from typing import Optional

from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session as OrmSession

from auth.sessions import SESSION_COOKIE_NAME, check_csrf, get_session
from db.database import get_db
from db.models import Session as SessionModel
from db.models import User


def _session_from_cookie(
    db: OrmSession,
    session_cookie: Optional[str],
) -> Optional[SessionModel]:
    if not session_cookie:
        return None
    return get_session(db, session_cookie)


def get_current_user(
    request: Request,
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> Optional[User]:
    """
    Returns the logged-in User, or None if not logged in.
    Does NOT raise — use require_user for protected routes.
    """
    s = _session_from_cookie(db, session_cookie)
    if s is None:
        return None
    user = db.get(User, s.user_id)
    if user is None:
        return None
    # Attach session to request.state so routes can access CSRF token
    request.state.session = s
    return user


def require_user(
    request: Request,
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> User:
    """Raises 401 if not authenticated."""
    s = _session_from_cookie(db, session_cookie)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    user = db.get(User, s.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    request.state.session = s
    return user


def require_role(*allowed_roles: str):
    """
    Factory: returns a dependency that raises 403 if user's role
    is not in allowed_roles.
    """
    def _checker(user: User = Depends(require_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden",
            )
        return user

    return _checker


def require_csrf(
    request: Request,
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    x_csrf_token: Optional[str] = Header(default=None, alias="X-CSRF-Token"),
) -> SessionModel:
    """
    CSRF check for POST endpoints.

    Token may be provided either:
      - via header 'X-CSRF-Token' (for API/AJAX calls), or
      - via form field 'csrf_token' (parsed inside the route).

    Since FastAPI doesn't parse form bodies here (that's route-specific),
    we accept the header form as the primary mechanism. HTML forms in the
    platform UI will submit via fetch() with the header set, OR routes can
    call verify_csrf_form() themselves after reading the form.

    Raises 403 if session missing or CSRF mismatch.
    """
    s = _session_from_cookie(db, session_cookie)
    if s is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF check failed",
        )
    if not check_csrf(s, x_csrf_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF check failed",
        )
    return s


def verify_csrf_form(db: OrmSession, session_cookie: Optional[str], form_token: Optional[str]) -> bool:
    """
    Helper for HTML-form routes that read `csrf_token` from the form body.
    Returns True if valid, False otherwise. Callers should raise 403 on False.
    """
    s = _session_from_cookie(db, session_cookie)
    return check_csrf(s, form_token)