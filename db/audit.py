"""
AI Software Company — Audit log helper (Step 3)

Architecture rules (Section 24 + F28):
  - Every important action is logged to `audit_log` table
  - Fields: timestamp, trace_id, job_id, user_id, action, status,
            message, duration_ms
  - No secrets in messages (regex filter later in security layer)
"""

from typing import Optional

from sqlalchemy.orm import Session as OrmSession

from db.models import AuditLog


# ---------------------------------------------------------------------------
# Action names (subset relevant to Step 3; more added in later steps)
# ---------------------------------------------------------------------------
ACTION_USER_LOGIN = "user_login"
ACTION_USER_LOGOUT = "user_logout"
ACTION_LOGIN_FAILED = "login_failed"
ACTION_CSRF_FAILED = "csrf_failed"
ACTION_AUTH_DENIED = "auth_denied"
ACTION_USER_CREATED = "user_created"
ACTION_SEED_RUN = "seed_run"
ACTION_APPROVAL_REQUESTED = "approval_requested"
ACTION_APPROVAL_GRANTED = "approval_granted"
ACTION_APPROVAL_REJECTED = "approval_rejected"


def log_action(
    db: OrmSession,
    action: str,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    user_id: Optional[int] = None,
    job_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    duration_ms: Optional[int] = None,
    commit: bool = True,
) -> AuditLog:
    """
    Write one audit log row.

    commit=True  -> commit immediately (safe default for short-lived audit writes)
    commit=False -> caller commits as part of a bigger transaction
    """
    entry = AuditLog(
        action=action,
        status=status,
        message=message,
        user_id=user_id,
        job_id=job_id,
        trace_id=trace_id,
        duration_ms=duration_ms,
    )
    db.add(entry)
    if commit:
        db.commit()
        db.refresh(entry)
    return entry