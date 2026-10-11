"""Failure store (Phase 1 of checkpoint system)."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import select
from db.database import SessionLocal
from db.models import JobFailure


def _utcnow():
    return datetime.now(timezone.utc)


def score_confidence(error_code: str) -> tuple[str, int]:
    """Return (severity, confidence_percent 0-100)."""
    if not error_code:
        return ("unknown", 50)
    ec = error_code
    if ec.startswith("file_too_small"):
        try:
            parts = ec.split(":", 1)[1]
            got_s, need_s = parts.split("<")
            got, need = int(got_s), int(need_s)
            ratio = got / need
            if ratio > 0.95: return ("minor", 95)
            if ratio > 0.85: return ("minor", 85)
            if ratio > 0.70: return ("moderate", 65)
            return ("severe", 30)
        except Exception:
            return ("moderate", 50)
    if ec.startswith("syntax_error"):  return ("minor", 80)
    if ec.startswith("bad_path"):      return ("minor", 85)
    if ec.startswith("not_json"):      return ("moderate", 55)
    if ec.startswith("dangerous"):     return ("severe", 25)
    if ec.startswith("secret"):        return ("severe", 20)
    if ec.startswith("path_blocked"):  return ("severe", 15)
    return ("unknown", 50)


def save_failure(job_id, file_path, error_code, error_message, ai_content):
    severity, confidence = score_confidence(error_code or "")
    db = SessionLocal()
    try:
        f = JobFailure(
            job_id=str(job_id), file_path=str(file_path),
            error_code=str(error_code or "")[:128],
            error_message=str(error_message or "")[:1000],
            severity=severity, confidence=confidence,
            ai_content=ai_content or "",
        )
        db.add(f); db.commit(); db.refresh(f)
        return f
    finally:
        db.close()


def get_pending(job_id):
    db = SessionLocal()
    try:
        rows = db.execute(
            select(JobFailure).where(
                JobFailure.job_id == str(job_id),
                JobFailure.resolved == 0,
            ).order_by(JobFailure.id)
        ).scalars().all()
        return [{
            "id": r.id, "file_path": r.file_path,
            "error_code": r.error_code, "error_message": r.error_message,
            "severity": r.severity, "confidence": r.confidence,
            "ai_content": r.ai_content,
        } for r in rows]
    finally:
        db.close()


def resolve_failure(failure_id, resolution, manual_content=None):
    db = SessionLocal()
    try:
        f = db.get(JobFailure, int(failure_id))
        if f is None:
            return None
        f.resolved = 1
        f.resolution = str(resolution)[:32]
        if manual_content is not None:
            f.manual_content = manual_content
        f.resolved_at = _utcnow()
        db.commit(); db.refresh(f)
        return {"id": f.id, "file_path": f.file_path,
                "resolution": f.resolution, "manual_content": f.manual_content}
    finally:
        db.close()
