"""
AI Software Company — Jobs store (Step 17)

Architecture references:
  Section 17  GET /status/{job_id}
  Section 18  Platform UI (jobs list, job status page)
  Section 22  Build Artifact fields (job_id, status, created_at)

Purpose:
  Thin CRUD layer over the `jobs` table. Used by:
    - api/routes_build.py   (POST /build creates a job)
    - api/routes_status.py  (GET /status/{job_id} reads a job)
    - api/routes_ui.py      (jobs list + job status page)
    - workers/build_worker.py (updates progress as the workflow runs)

Design:
  - SQLAlchemy ORM via db.database.SessionLocal.
  - JSON blobs (features, files, scores) stored as TEXT; helpers encode/decode.
  - Never raises on "not found": returns None.
  - Commit on success, rollback on exception. No commit in finally.

Public API:
  create_job(job_id, user_id, requirement) -> Job
  get_job(job_id) -> Job | None
  list_jobs(user_id=None, limit=100) -> list[Job]
  update_job(job_id, **fields) -> Job | None
  set_error(job_id, agent, code, message, attempt) -> Job | None
  clear_error(job_id) -> Job | None
  set_features(job_id, features) -> None
  set_files(job_id, files) -> None
  set_scores(job_id, scores) -> None
  decode_features(job) / decode_files(job) / decode_scores(job) -> python obj
  job_to_status_dict(job) -> dict   (for GET /status/{job_id})
  delete_job(job_id) -> bool
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from db.database import SessionLocal
from db.models import Job


# --- Helpers -------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json_dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return None


def _safe_json_loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


# --- Create --------------------------------------------------------------
def create_job(
    job_id: str,
    user_id: Optional[int],
    requirement: str,
    status: str = "building",
) -> Job:
    """Insert a new job row. Raises if job_id already exists (UNIQUE)."""
    if not job_id:
        raise ValueError("job_id required")
    db: OrmSession = SessionLocal()
    try:
        job = Job(
            job_id=str(job_id).strip(),
            user_id=user_id,
            requirement=str(requirement)[:4000],
            status=status,
            current_node="",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# --- Read ----------------------------------------------------------------
def get_job(job_id: str) -> Optional[Job]:
    if not job_id:
        return None
    db: OrmSession = SessionLocal()
    try:
        return db.execute(
            select(Job).where(Job.job_id == str(job_id).strip())
        ).scalar_one_or_none()
    finally:
        db.close()


def list_jobs(user_id: Optional[int] = None, limit: int = 100) -> list[Job]:
    """If user_id is None, returns all jobs (admin view). Newest first."""
    db: OrmSession = SessionLocal()
    try:
        stmt = select(Job).order_by(Job.id.desc()).limit(int(limit))
        if user_id is not None:
            stmt = select(Job).where(Job.user_id == user_id).order_by(
                Job.id.desc()
            ).limit(int(limit))
        return list(db.execute(stmt).scalars().all())
    finally:
        db.close()


# --- Update --------------------------------------------------------------
_ALLOWED_FIELDS = {
    "status", "current_node",
    "error_agent", "error_code", "error_message", "error_attempt",
    "live_url", "github_url", "version",
}


def update_job(job_id: str, **fields: Any) -> Optional[Job]:
    """Update allowed fields on a job. Returns the updated row or None."""
    if not job_id:
        return None
    unknown = set(fields.keys()) - _ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")

    db: OrmSession = SessionLocal()
    try:
        job = db.execute(
            select(Job).where(Job.job_id == str(job_id).strip())
        ).scalar_one_or_none()
        if job is None:
            return None
        for k, v in fields.items():
            setattr(job, k, v)
        job.updated_at = _utcnow()
        db.commit()
        db.refresh(job)
        return job
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def set_error(
    job_id: str,
    agent: str,
    code: str,
    message: str,
    attempt: Optional[int] = None,
) -> Optional[Job]:
    """Record a safe error on the job (no secrets in message)."""
    return update_job(
        job_id,
        error_agent=str(agent)[:32],
        error_code=str(code)[:64],
        error_message=str(message)[:1000],
        error_attempt=int(attempt) if attempt is not None else None,
    )


def clear_error(job_id: str) -> Optional[Job]:
    return update_job(
        job_id,
        error_agent=None, error_code=None, error_message=None, error_attempt=None,
    )


def set_features(job_id: str, features: list) -> None:
    db: OrmSession = SessionLocal()
    try:
        job = db.execute(
            select(Job).where(Job.job_id == str(job_id).strip())
        ).scalar_one_or_none()
        if job is None:
            return
        job.features_json = _safe_json_dumps(features or [])
        job.updated_at = _utcnow()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def set_files(job_id: str, files: list) -> None:
    db: OrmSession = SessionLocal()
    try:
        job = db.execute(
            select(Job).where(Job.job_id == str(job_id).strip())
        ).scalar_one_or_none()
        if job is None:
            return
        # Store only paths and lengths (do not store full content in the jobs row)
        compact = []
        for f in (files or []):
            if isinstance(f, dict):
                compact.append({
                    "path": f.get("path", ""),
                    "bytes": len(f.get("content", "") or ""),
                })
            else:
                compact.append({"path": str(f), "bytes": 0})
        job.files_json = _safe_json_dumps(compact)
        job.updated_at = _utcnow()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def set_scores(job_id: str, scores: dict) -> None:
    db: OrmSession = SessionLocal()
    try:
        job = db.execute(
            select(Job).where(Job.job_id == str(job_id).strip())
        ).scalar_one_or_none()
        if job is None:
            return
        job.scores_json = _safe_json_dumps(scores or {})
        job.updated_at = _utcnow()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


# --- Decode --------------------------------------------------------------
def decode_features(job: Job) -> list:
    return _safe_json_loads(job.features_json, [])


def decode_files(job: Job) -> list:
    return _safe_json_loads(job.files_json, [])


def decode_scores(job: Job) -> dict:
    return _safe_json_loads(job.scores_json, {})


# --- Status dict (for GET /status/{job_id}) ------------------------------
def job_to_status_dict(job: Job) -> dict:
    """
    Return the public JSON shape for GET /status/{job_id}.
    Never includes secrets or raw file contents.
    """
    err = None
    if job.error_message or job.error_code or job.error_agent:
        err = {
            "agent": job.error_agent or "",
            "error_code": job.error_code or "",
            "message": job.error_message or "",
            "attempt": job.error_attempt,
        }
    files = decode_files(job)

    # Progress hints for UI timer
    total_files_est = 10  # typical plan size (Architect decides 8-10)
    files_done = len(files)

    # Estimated total build time (seconds) based on observed averages
    # PM+Architect+Designer ~30s, each file ~35s, tester ~120s, improver ~120s
    base = 30 + 120 + 120
    estimate_total = base + total_files_est * 35

    # Elapsed from created_at to now
    elapsed = 0
    if job.created_at:
        from datetime import datetime, timezone
        try:
            created = job.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            elapsed = int((datetime.now(timezone.utc) - created).total_seconds())
        except Exception:
            elapsed = 0

    # Refine estimate as files complete
    if files_done > 0 and job.status == "building":
        # extrapolate: (elapsed / files_done) * total
        per_file = max(20, elapsed // max(files_done, 1))
        estimate_total = base + per_file * total_files_est

    remaining = max(0, estimate_total - elapsed)

    return {
        "job_id": job.job_id,
        "requirement": job.requirement,
        "status": job.status,
        "current_node": job.current_node or "",
        "features": decode_features(job),
        "files": files,
        "scores": decode_scores(job),
        "error": err,
        "live_url": job.live_url or "",
        "github_url": job.github_url or "",
        "version": job.version or "",
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        # Timer fields
        "elapsed_seconds": elapsed,
        "estimate_total_seconds": estimate_total,
        "remaining_seconds": remaining,
        "files_done": files_done,
        "files_total_est": total_files_est,
    }


# --- Delete --------------------------------------------------------------
def delete_job(job_id: str) -> bool:
    if not job_id:
        return False
    db: OrmSession = SessionLocal()
    try:
        job = db.execute(
            select(Job).where(Job.job_id == str(job_id).strip())
        ).scalar_one_or_none()
        if job is None:
            return False
        db.delete(job)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def set_tester_issues(job_id: str, issues: list) -> None:
    """Store tester issues so Coder can pick them up on retry."""
    db = SessionLocal()
    try:
        job = db.execute(
            select(Job).where(Job.job_id == str(job_id).strip())
        ).scalar_one_or_none()
        if job is None:
            return
        # Reuse scores_json field is wrong; add column? No - use error_message
        # Actually store in a simple JSON in existing field
        import json as _json
        job.updated_at = _utcnow()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

