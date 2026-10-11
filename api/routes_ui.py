"""
AI Software Company — Platform UI routes (Step 17)

Architecture references:
  Section 18  Platform UI (4 pages: login, submit, jobs, job_status)
  Section 7   Auth + roles
  Section 8   CSRF + session cookie
  F13         Platform UI (4 pages)
  F19         Build failed error box (shown on job_status page)

Page routes (render HTML):
  GET  /login              — login form
  GET  /submit             — new build form
  GET  /jobs               — jobs list (admin sees all)
  GET  /jobs/{job_id}      — job detail with 3s polling

Form-action routes (POST, redirect on success):
  POST /ui/login           — login form handler (sets session cookie)
  POST /ui/logout          — logout form handler (clears cookie)
  POST /ui/build           — build submit handler (creates job + enqueue)

Design:
  - Uses Jinja2Templates pointed at templates/.
  - CSRF token is generated per request when no session exists yet
    (for the login form), and loaded from the session when a session
    exists (for submit/logout forms).
  - Templates receive: current_user, csrf_token, flash_error, flash_message.
  - All POST form handlers require CSRF (except /ui/login, which is the
    entry point and has no session yet).
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from auth.deps import get_current_user, require_user
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
from db.jobs_store import create_job, get_job, list_jobs
from db.models import User
from security import prompt_guard as pg
from workers.queue import enqueue_build

router = APIRouter(tags=["ui"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Template context helper ---------------------------------------------
def _ctx(
    request: Request,
    *,
    current_user: Optional[User] = None,
    csrf_token: str = "",
    flash_error: Optional[str] = None,
    flash_message: Optional[str] = None,
    **extra,
) -> dict:
    base = {
        "request": request,
        "current_user": current_user,
        "csrf_token": csrf_token,
        "flash_error": flash_error,
        "flash_message": flash_message,
    }
    base.update(extra)
    return base


# --- Page routes ---------------------------------------------------------
@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    error: Optional[str] = None,
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Render login form. If already logged in, redirect to /jobs."""
    user = get_current_user(request=request, db=db, session_cookie=session_cookie)
    if user is not None:
        return RedirectResponse(url="/jobs", status_code=303)

    # No session yet: generate an ephemeral CSRF token for the login form.
    # The server-side verify step (POST /ui/login) does NOT enforce CSRF
    # on the login form, because there is no session to bind it to yet.
    csrf_token = secrets.token_urlsafe(32)
    flash_error = None
    if error == "invalid":
        flash_error = "Invalid username or password."
    elif error == "csrf":
        flash_error = "Session expired. Please try again."
    return templates.TemplateResponse(
        request,
        "platform/login.html",
        _ctx(request, csrf_token=csrf_token, flash_error=flash_error),
    )


@router.get("/submit", response_class=HTMLResponse)
async def submit_page(
    request: Request,
    user: User = Depends(require_user),
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Render the new-build form (requires login)."""
    s = get_session(db, session_cookie) if session_cookie else None
    csrf = s.csrf_token if s else secrets.token_urlsafe(32)
    return templates.TemplateResponse(
        request,
        "platform/submit.html",
        _ctx(request, current_user=user, csrf_token=csrf),
    )


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(
    request: Request,
    user: User = Depends(require_user),
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Jobs list. Admin sees all; others see only their own."""
    s = get_session(db, session_cookie) if session_cookie else None
    csrf = s.csrf_token if s else secrets.token_urlsafe(32)

    if user.role == "admin":
        jobs = list_jobs(user_id=None, limit=100)
    else:
        jobs = list_jobs(user_id=user.id, limit=100)

    # Light view-model
    rows = []
    for j in jobs:
        rows.append({
            "job_id": j.job_id,
            "requirement": j.requirement or "",
            "status": j.status or "unknown",
            "created_at": j.created_at.strftime("%Y-%m-%d %H:%M") if j.created_at else "",
        })

    return templates.TemplateResponse(
        request,
        "platform/jobs.html",
        _ctx(request, current_user=user, csrf_token=csrf, jobs=rows),
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_status_page(
    request: Request,
    job_id: str,
    user: User = Depends(require_user),
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Job detail page (3-second polling JS is in the template)."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if user.role != "admin" and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="not allowed")

    s = get_session(db, session_cookie) if session_cookie else None
    csrf = s.csrf_token if s else secrets.token_urlsafe(32)

    return templates.TemplateResponse(
        request,
        "platform/job_status.html",
        _ctx(
            request,
            current_user=user,
            csrf_token=csrf,
            job_id=job.job_id,
            status=job.status or "unknown",
            requirement=job.requirement or "",
        ),
    )


# --- Form-action routes --------------------------------------------------
@router.post("/ui/login")
async def ui_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(default=""),
    db: OrmSession = Depends(get_db),
):
    """
    Form login. Creates session, sets cookie, redirects to /jobs.
    On failure redirects back to /login with an error code.
    """
    stmt = select(User).where(User.username == username)
    user = db.execute(stmt).scalar_one_or_none()

    if user is None or not verify_password(password, user.password_hash):
        try:
            log_action(
                db, ACTION_LOGIN_FAILED, status="fail",
                message=f"ui login failed for username={username!r}",
            )
        except Exception:
            pass
        return RedirectResponse(url="/login?error=invalid", status_code=303)

    s = create_session(db, user.id)
    try:
        log_action(
            db, ACTION_USER_LOGIN, status="ok",
            user_id=user.id, message="ui login ok",
        )
    except Exception:
        pass

    resp = RedirectResponse(url="/jobs", status_code=303)
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=s.session_id,
        max_age=SESSION_LIFETIME_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=False,  # local dev
        path="/",
    )
    return resp


@router.post("/ui/logout")
async def ui_logout(
    request: Request,
    csrf_token: str = Form(default=""),
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Form logout. Requires CSRF. Clears cookie and redirects to /login."""
    s = get_session(db, session_cookie) if session_cookie else None
    if s is None:
        resp = RedirectResponse(url="/login", status_code=303)
        resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return resp

    if not check_csrf(s, csrf_token):
        try:
            log_action(
                db, ACTION_CSRF_FAILED, status="fail",
                user_id=s.user_id, message="ui logout csrf mismatch",
            )
        except Exception:
            pass
        raise HTTPException(status_code=403, detail="CSRF check failed")

    user_id = s.user_id
    delete_session(db, s.session_id)
    try:
        log_action(
            db, ACTION_USER_LOGOUT, status="ok",
            user_id=user_id, message="ui logout ok",
        )
    except Exception:
        pass

    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp


@router.post("/ui/build")
async def ui_build(
    request: Request,
    requirement: str = Form(...),
    csrf_token: str = Form(default=""),
    user: User = Depends(require_user),
    db: OrmSession = Depends(get_db),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Form build submit. Validates, creates job, enqueues, redirects."""
    s = get_session(db, session_cookie) if session_cookie else None
    if s is None or not check_csrf(s, csrf_token):
        raise HTTPException(status_code=403, detail="CSRF check failed")

    # L1 sanitize (prompt guard)
    san = pg.sanitize_user_input(requirement)
    if not san.ok:
        return RedirectResponse(
            url=f"/submit?error={san.reason}", status_code=303
        )

    # Create job row
    job_id = "job-" + uuid.uuid4().hex[:16]
    try:
        create_job(job_id=job_id, user_id=user.id, requirement=san.cleaned)
    except Exception:
        raise HTTPException(status_code=500, detail="could not create job")

    # Enqueue
    try:
        await enqueue_build(job_id)
    except Exception:
        # Job row remains; worker won't pick it up. Surface to user.
        return RedirectResponse(
            url=f"/jobs/{job_id}", status_code=303
        )

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)