"""
AI Software Company — ARQ queue helper (Step 17)

Architecture references:
  Section 26  ARQ (Async Redis Queue)
  F1          ARQ + Redis (Docker container)
  Config: max_concurrent_builds=3, queue_size=20, per_user_active_jobs=1,
          job_timeout=15 min

Public API:
  get_pool() -> ArqRedis        (lazy singleton)
  close_pool() -> None
  enqueue_build(job_id) -> None (raises on connection failure)
  ping() -> bool                (used by GET /ready)

Design:
  - Redis URL comes from .env (REDIS_URL).
  - The worker function name must match workers/build_worker.py's
    `build_workflow` function.
  - Enqueue uses _job_id=job_id for idempotency: if the same job_id is
    enqueued twice, ARQ will not create a duplicate entry.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from dotenv import load_dotenv

from observability.logger import get_logger

load_dotenv()

logger = get_logger("workers.queue")

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")

# Must match the function name in workers/build_worker.py
BUILD_WORKER_FUNCTION = "build_workflow"

_pool: Optional[ArqRedis] = None


def _redis_settings_from_url(url: str) -> RedisSettings:
    """Convert redis://[:pass@]host:port/db into ARQ RedisSettings."""
    p = urlparse(url)
    path = (p.path or "/0").lstrip("/") or "0"
    try:
        db = int(path)
    except Exception:
        db = 0
    return RedisSettings(
        host=p.hostname or "127.0.0.1",
        port=p.port or 6379,
        database=db,
        password=p.password or None,
    )


async def get_pool() -> ArqRedis:
    """Return a shared ARQ Redis pool. Creates it on first use."""
    global _pool
    if _pool is None:
        _pool = await create_pool(_redis_settings_from_url(REDIS_URL))
        logger.info("arq_pool_created", url=REDIS_URL.split("@")[-1])
    return _pool


async def close_pool() -> None:
    """Close the pool (used on shutdown and in tests)."""
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
        _pool = None


async def enqueue_build(job_id: str) -> None:
    """
    Enqueue a build job. Raises RuntimeError if Redis is unreachable.
    Idempotent by _job_id.
    """
    if not job_id:
        raise ValueError("job_id required")
    pool = await get_pool()
    await pool.enqueue_job(BUILD_WORKER_FUNCTION, job_id, _job_id=str(job_id))
    logger.info("enqueue_build", job_id=job_id)


async def ping() -> bool:
    """Return True if Redis is reachable (used by GET /ready)."""
    try:
        pool = await get_pool()
        pong = await pool.ping()
        return bool(pong)
    except Exception as exc:
        logger.warning("redis_ping_failed", err=type(exc).__name__)
        return False