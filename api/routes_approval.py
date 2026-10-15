"""
AI Software Company — Approval routes (Step 15)

Architecture references:
  Section 7   Roles: admin, user, approver (Approval Authority)
  Section 17  POST /approve/{job_id} with {"decision": "yes/no"}
  Section 18  Platform UI polls status every 3 s
  Section 23  Human approval gate (AI cannot self-approve)
  Section 24  Audit actions handled in security.approval

Endpoints:
  POST /approve/{job_id}   (Approval Authority only; CSRF required)
      body: {"decision": "yes"} or {"decision": "no"}
      returns: {"status": "approved"|"rejected", "job_id": str, ...}

  GET  /approval/{job_id}  (any authenticated user)
      returns: {"job_id": str, "status": "pending"|"approved"|"rejected",
                "decision": str|None, "requested_by": int|None,
                "decided_by": int|None, "decided_at": iso|None}
      returns {"status": "none"} if no row exists yet.

Security:
  - Cookie: HttpOnly (set by login) — read via auth.deps
  - CSRF: required on POST via X-CSRF-Token header
  - No secrets in error responses
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as OrmSession

from auth.deps import require_user
from auth.sessions import SESSION_COOKIE_NAME, check_csrf, get_session
from db.database import get_db
from db.models import User
from security.approval import (
    ApprovalAlreadyDecidedError,
    ApprovalNotFoundError,
    decide_approval,
    get_approval,
)



# --- Trigger deploy after approval --------------------------------------
async def _trigger_deploy_after_approval(job_id: str, db) -> dict:
    """Rebuild WorkflowState from disk files + DB, then run deploy_node."""
    import json as _json
    from pathlib import Path as _P

    # 1. Load job from DB
    from db.jobs_store import get_job
    job = get_job(job_id)
    if job is None:
        return {"trigger": "deploy", "status": "skipped", "reason": "job_not_found"}

    # 2. Read all files from disk
    out_root = _P("output") / job_id
    if not out_root.exists():
        return {"trigger": "deploy", "status": "skipped", "reason": "no_output_dir"}

    code_files = []
    for p in out_root.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(out_root)).replace("\\", "/")
        if rel in ("Dockerfile", "test_runner.py"):
            continue
        try:
            code_files.append({"path": rel, "content": p.read_text(encoding="utf-8", errors="replace")})
        except Exception:
            continue

    if not code_files:
        return {"trigger": "deploy", "status": "skipped", "reason": "no_files"}

    # 3. Rebuild minimal WorkflowState
    try:
        features = _json.loads(job.features_json) if job.features_json else []
    except Exception:
        features = []

    state = {
        "job_id": job_id,
        "user_id": str(job.user_id),
        "trace_id": job_id,
        "requirement": job.requirement,
        "app_name": (features[0] if features else "App"),
        "features": features,
        "code_files": code_files,
        "approved": True,
        "approval_status": "approved",
    }

    # 4. Run deploy_node
    from graph.nodes import deploy_node
    result = await deploy_node(state)
    return {
        "trigger": "deploy",
        "status": "ok",
        "github_url": result.get("github_url"),
        "live_url": result.get("live_url"),
        "port": result.get("port"),
    }


router = APIRouter(tags=["approval"])


# --- Request body --------------------------------------------------------
class ApproveBody(BaseModel):
    decision: str = Field(..., description='"yes" or "no"')


# --- POST /approve/{job_id} ---------------------------------------------
@router.post("/approve/{job_id}")
async def approve_job(
    job_id: str,
    body: ApproveBody,
    request: Request,
    db: OrmSession = Depends(get_db),
    user: User = Depends(require_user),
    session_cookie: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    x_csrf_token: Optional[str] = Header(default=None, alias="X-CSRF-Token"),
):
    """
    Approve or reject a job. Only Approval Authority may call this.
    CSRF token (header X-CSRF-Token) is required.
    """
    # 1. Role check
    if user.role != "approver":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Approval Authority role required",
        )

    # 2. CSRF check
    s = get_session(db, session_cookie) if session_cookie else None
    if s is None or not check_csrf(s, x_csrf_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF check failed",
        )

    # 3. Decide
    try:
        req = decide_approval(
            db=db,
            job_id=job_id,
            decision=body.decision,
            decider_user_id=user.id,
        )
    except ApprovalNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ApprovalAlreadyDecidedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    # 3.5 ? If approved, trigger deploy (non-blocking)
    deploy_result = None
    if body.decision == "yes":
        try:
            deploy_result = await _trigger_deploy_after_approval(job_id, db)
        except Exception as exc:
            import logging as _log
            _log.getLogger("approval").warning(
                "deploy_after_approval_failed job=%s err=%s", job_id, type(exc).__name__
            )
            deploy_result = {"trigger": "deploy", "status": "failed",
                             "reason": type(exc).__name__}

    return {
        "job_id": req.job_id,
        "status": req.status,
        "decision": req.decision,
        "decided_by": req.decided_by,
        "decided_at": req.decided_at.isoformat() if req.decided_at else None,
        "deploy": deploy_result,
    }


# --- GET /approval/{job_id} ---------------------------------------------
@router.get("/approval/{job_id}")
async def approval_status(
    job_id: str,
    db: OrmSession = Depends(get_db),
    user: User = Depends(require_user),
):
    """Return the current approval status for a job (any logged-in user)."""
    req = get_approval(db, job_id)
    if req is None:
        return {
            "job_id": job_id,
            "status": "none",
            "decision": None,
            "requested_by": None,
            "decided_by": None,
            "decided_at": None,
        }
    return {
        "job_id": req.job_id,
        "status": req.status,
        "decision": req.decision,
        "requested_by": req.requested_by,
        "decided_by": req.decided_by,
        "decided_at": req.decided_at.isoformat() if req.decided_at else None,
    }