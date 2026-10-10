"""
AI Software Company — Approval business logic (Step 15)

Architecture references:
  Section 23  Human approval gate (AI cannot self-approve)
  Section 24  Audit actions: approval_requested, approval_granted,
              approval_rejected
  Section 7   Roles: Admin, User/Developer, Approval Authority

Design:
  - One ApprovalRequest row per job (UNIQUE on job_id).
  - Status transitions:
        (none)  -> pending    via request_approval()
        pending -> approved   via decide_approval(decision="yes")
        pending -> rejected   via decide_approval(decision="no")
  - Once decided, cannot be changed (idempotency enforced by state check).
  - Audit log is written on every state change.
  - Commit only on success, rollback on exception (Section 22 note).

Public API:
  request_approval(db, job_id, requested_by) -> ApprovalRequest
  decide_approval(db, job_id, decision, decider_user_id) -> ApprovalRequest
  get_approval(db, job_id) -> ApprovalRequest | None
  is_approved(db, job_id) -> bool
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from db.audit import (
    ACTION_APPROVAL_GRANTED,
    ACTION_APPROVAL_REJECTED,
    ACTION_APPROVAL_REQUESTED,
    log_action,
)
from db.models import ApprovalRequest


class ApprovalError(RuntimeError):
    """Base error for approval flow."""


class ApprovalAlreadyDecidedError(ApprovalError):
    """Raised when trying to decide an already-decided request."""


class ApprovalNotFoundError(ApprovalError):
    """Raised when trying to decide a request that does not exist."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --- Request -------------------------------------------------------------
def request_approval(
    db: OrmSession,
    job_id: str,
    requested_by: Optional[int] = None,
) -> ApprovalRequest:
    """
    Create a pending approval request for `job_id`.

    Idempotent: if a row already exists for this job_id, returns it
    unchanged (does NOT reset an already-decided request).
    """
    if not job_id or not str(job_id).strip():
        raise ValueError("job_id required")
    job_id = str(job_id).strip()

    existing = db.execute(
        select(ApprovalRequest).where(ApprovalRequest.job_id == job_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    try:
        req = ApprovalRequest(
            job_id=job_id,
            requested_by=requested_by,
            status="pending",
            decision=None,
            decided_by=None,
            decided_at=None,
        )
        db.add(req)
        db.flush()  # so req.id is set; commit happens below

        log_action(
            db,
            ACTION_APPROVAL_REQUESTED,
            status="pending",
            user_id=requested_by,
            job_id=job_id,
            message=f"approval requested for job {job_id}",
            commit=False,
        )
        db.commit()
        db.refresh(req)
        return req
    except Exception:
        db.rollback()
        raise


# --- Decide --------------------------------------------------------------
def decide_approval(
    db: OrmSession,
    job_id: str,
    decision: str,
    decider_user_id: int,
) -> ApprovalRequest:
    """
    Approve or reject an existing pending request.

    decision must be "yes" or "no".
    Raises ApprovalNotFoundError if no row exists.
    Raises ApprovalAlreadyDecidedError if status is not "pending".
    """
    if not job_id:
        raise ValueError("job_id required")
    d = str(decision).strip().lower()
    if d not in ("yes", "no"):
        raise ValueError("decision must be 'yes' or 'no'")

    job_id = str(job_id).strip()

    req = db.execute(
        select(ApprovalRequest).where(ApprovalRequest.job_id == job_id)
    ).scalar_one_or_none()

    if req is None:
        raise ApprovalNotFoundError(f"no approval request for job {job_id!r}")

    if req.status != "pending":
        raise ApprovalAlreadyDecidedError(
            f"approval for job {job_id!r} already {req.status!r}"
        )

    try:
        req.status = "approved" if d == "yes" else "rejected"
        req.decision = d
        req.decided_by = decider_user_id
        req.decided_at = _utcnow()

        action = (
            ACTION_APPROVAL_GRANTED if d == "yes" else ACTION_APPROVAL_REJECTED
        )
        log_action(
            db,
            action,
            status=req.status,
            user_id=decider_user_id,
            job_id=job_id,
            message=f"approval {req.status} for job {job_id}",
            commit=False,
        )
        db.commit()
        db.refresh(req)
        return req
    except Exception:
        db.rollback()
        raise


# --- Queries -------------------------------------------------------------
def get_approval(db: OrmSession, job_id: str) -> Optional[ApprovalRequest]:
    """Return the ApprovalRequest for job_id, or None."""
    if not job_id:
        return None
    return db.execute(
        select(ApprovalRequest).where(
            ApprovalRequest.job_id == str(job_id).strip()
        )
    ).scalar_one_or_none()


def is_approved(db: OrmSession, job_id: str) -> bool:
    """True iff an approved request exists for this job."""
    req = get_approval(db, job_id)
    return req is not None and req.status == "approved"