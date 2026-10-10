"""
AI Software Company — Status routes (Step 17)

Architecture references:
  Section 17  GET /status/{job_id}
  Section 25  Rate limiting: GET /status 60/min per user
  Section 29  GET /ready

Endpoints:
  GET /status/{job_id}   — owner or admin only (Section 17)
  GET /ready             — platform readiness (DB, Redis, dirs, Docker network)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import text
from sqlalchemy.orm import Session as OrmSession

from auth.deps import require_user
from db.database import SessionLocal, get_db
from db.jobs_store import get_job, job_to_status_dict
from db.models import User
from workers.queue import ping as redis_ping

router = APIRouter(tags=["status"])

limiter = Limiter(key_func=get_remote_address)


@router.get("/status/{job_id}")
@limiter.limit("60/minute")
async def get_status(
    request: Request,
    job_id: str,
    user: User = Depends(require_user),
    db: OrmSession = Depends(get_db),
):
    """Owner or admin can view job status. Returns 404 if not found."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="job not found",
        )
    if user.role != "admin" and job.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="not allowed to view this job",
        )
    return job_to_status_dict(job)


@router.get("/ready")
async def ready():
    """
    Readiness check used by Step 18 staging/load tests.
    Returns 200 with a checks dict. If a critical check fails, returns 503.
    """
    checks: dict = {}

    # 1. Database: trivial query
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            checks["database"] = True
        finally:
            db.close()
    except Exception as exc:
        checks["database"] = False
        checks["database_error"] = type(exc).__name__

    # 2. Environment: required keys
    required_env = ["SECRET_KEY", "DATABASE_URL", "REDIS_URL", "GROQ_API_KEY"]
    missing = [k for k in required_env if not (os.getenv(k) or "").strip()]
    checks["environment"] = len(missing) == 0
    if missing:
        checks["environment_missing"] = missing

    # 3. Directories
    root = Path(__file__).resolve().parent.parent
    for d in ["output", "logs"]:
        p = root / d
        try:
            p.mkdir(parents=True, exist_ok=True)
            checks[f"dir_{d}"] = True
        except Exception:
            checks[f"dir_{d}"] = False

    # 4. Redis (ARQ queue)
    try:
        checks["queue"] = await redis_ping()
    except Exception:
        checks["queue"] = False

    # 5. Chroma (RAG)
    try:
        from rag import chroma_client as cc
        checks["rag"] = cc.is_available()
    except Exception:
        checks["rag"] = False

    # 6. Groq key present
    checks["ai_key_present"] = bool((os.getenv("GROQ_API_KEY") or "").strip())

    # 7. Docker network (best-effort)
    try:
        import subprocess
        r = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5,
        )
        checks["docker_network"] = "ai_company_network" in (r.stdout or "")
    except Exception:
        checks["docker_network"] = False

    critical_ok = (
        bool(checks.get("database"))
        and bool(checks.get("environment"))
        and bool(checks.get("queue"))
    )

    payload = {
        "status": "ready" if critical_ok else "not_ready",
        "checks": checks,
    }
    if not critical_ok:
        return JSONResponse(status_code=503, content=payload)
    return payload