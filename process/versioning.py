"""
AI Software Company — Versioning (Step 14)

Architecture references:
  Section 21  Versioning + Rollback
              Successful -> version save
              Version/job mapping
              Previous working identify
              Rollback audit log
              GitHub tag
  Section 22  Build Artifact fields:
              job_id, project_id, version, artifact_path, status, created_at
  Section 24  Audit actions: rollback, deployment, build_failed

Design:
  - SQLite `versions` table: version -> job_id mapping per project_id.
  - `projects` table: logical grouping of versions for the same generated app.
  - Status enum: working|failed|rolled_back
  - "current working" = the latest version with status='working'
  - Rollback: mark a failed version as rolled_back and return the
    previous working version's job_id + artifact_path.
  - All DB writes use explicit transactions (BEGIN EXCLUSIVE / COMMIT /
    ROLLBACK) and never commit in finally.

Public API:
    init_versions_table()
    async def save_version(project_id, job_id, artifact_path, status="working")
        -> dict
    def get_versions(project_id) -> list[dict]
    def get_current_working(project_id) -> dict | None
    def get_version(project_id, version) -> dict | None
    async def rollback(project_id, failed_version) -> dict | None
        -> returns the previous working version dict, or None if none exists
    def list_all_versions() -> list[dict]
    def reset_versions() -> int  (tests only)
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from observability.logger import get_logger

logger = get_logger("process.versioning")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "ai_company.db"

_version_lock = asyncio.Lock()

VALID_STATUSES = {"working", "failed", "rolled_back"}


# --- Connection ----------------------------------------------------------
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Schema --------------------------------------------------------------
def init_versions_table() -> None:
    """
    Create `projects` and `versions` tables if they do not exist.
    Idempotent.
    """
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                job_id TEXT NOT NULL,
                artifact_path TEXT,
                status TEXT NOT NULL DEFAULT 'working',
                created_at TEXT NOT NULL,
                UNIQUE (project_id, version)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_project "
            "ON versions(project_id, version)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_job ON versions(job_id)"
        )


# --- Helpers -------------------------------------------------------------
def _row_to_dict(row: sqlite3.Row | tuple) -> dict:
    # Row order: id, project_id, version, job_id, artifact_path, status, created_at
    return {
        "id": int(row[0]),
        "project_id": row[1],
        "version": int(row[2]),
        "job_id": row[3],
        "artifact_path": row[4],
        "status": row[5],
        "created_at": row[6],
    }


def _ensure_project(conn: sqlite3.Connection, project_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO projects (project_id, created_at) VALUES (?, ?)",
        (project_id, _utcnow_iso()),
    )


# --- Save ----------------------------------------------------------------
async def save_version(
    project_id: str,
    job_id: str,
    artifact_path: Optional[str] = None,
    status: str = "working",
) -> dict:
    """
    Save a new version for a project.

    - If project_id does not exist, it is created.
    - version is `max(existing) + 1` for this project.
    - status must be one of: working | failed | rolled_back.

    Returns the saved row as a dict.
    """
    if not project_id or not job_id:
        raise ValueError("project_id and job_id must be non-empty")
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}")

    project_id = str(project_id).strip()
    job_id = str(job_id).strip()

    async with _version_lock:
        conn = _connect()
        try:
            conn.execute("BEGIN EXCLUSIVE")
            try:
                _ensure_project(conn, project_id)

                row = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM versions "
                    "WHERE project_id = ?",
                    (project_id,),
                ).fetchone()
                next_version = int(row[0]) + 1

                conn.execute(
                    "INSERT INTO versions "
                    "(project_id, version, job_id, artifact_path, status, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (project_id, next_version, job_id, artifact_path, status,
                     _utcnow_iso()),
                )
                conn.execute("COMMIT")

                logger.info(
                    "version_saved",
                    project_id=project_id,
                    version=next_version,
                    job_id=job_id,
                    status=status,
                )
                return {
                    "project_id": project_id,
                    "version": next_version,
                    "job_id": job_id,
                    "artifact_path": artifact_path,
                    "status": status,
                    "created_at": _utcnow_iso(),
                }
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        finally:
            try:
                conn.close()
            except Exception:
                pass


# --- Queries -------------------------------------------------------------
def get_versions(project_id: str) -> list[dict]:
    """Return all versions for a project ordered by version ascending."""
    if not project_id:
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, project_id, version, job_id, artifact_path, status, created_at "
            "FROM versions WHERE project_id = ? ORDER BY version ASC",
            (str(project_id).strip(),),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_current_working(project_id: str) -> Optional[dict]:
    """
    Return the newest version with status='working' for a project, or None.
    """
    if not project_id:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, project_id, version, job_id, artifact_path, status, created_at "
            "FROM versions WHERE project_id = ? AND status = 'working' "
            "ORDER BY version DESC LIMIT 1",
            (str(project_id).strip(),),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_version(project_id: str, version: int) -> Optional[dict]:
    """Return a specific version row, or None."""
    if not project_id:
        return None
    try:
        v = int(version)
    except Exception:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id, project_id, version, job_id, artifact_path, status, created_at "
            "FROM versions WHERE project_id = ? AND version = ? LIMIT 1",
            (str(project_id).strip(), v),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_all_versions() -> list[dict]:
    """Return every version row across all projects (for /metrics/tests)."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, project_id, version, job_id, artifact_path, status, created_at "
            "FROM versions ORDER BY project_id, version"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --- Rollback ------------------------------------------------------------
async def rollback(project_id: str, failed_version: int) -> Optional[dict]:
    """
    Mark `failed_version` as 'rolled_back' and return the previous
    'working' version for the same project.

    If `failed_version` is not found, or its status is not 'failed',
    raises ValueError. Returns None if there is no earlier working version.
    """
    if not project_id:
        raise ValueError("project_id required")
    try:
        fv = int(failed_version)
    except Exception as exc:
        raise ValueError("failed_version must be int") from exc

    project_id = str(project_id).strip()

    async with _version_lock:
        conn = _connect()
        try:
            conn.execute("BEGIN EXCLUSIVE")
            try:
                row = conn.execute(
                    "SELECT status FROM versions "
                    "WHERE project_id = ? AND version = ?",
                    (project_id, fv),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise ValueError(f"version {fv} not found for {project_id}")
                if row[0] != "failed":
                    conn.execute("ROLLBACK")
                    raise ValueError(
                        f"version {fv} status is {row[0]!r}; only 'failed' can be rolled back"
                    )

                # Mark as rolled back
                conn.execute(
                    "UPDATE versions SET status = 'rolled_back' "
                    "WHERE project_id = ? AND version = ?",
                    (project_id, fv),
                )

                # Find previous working version
                prev = conn.execute(
                    "SELECT id, project_id, version, job_id, artifact_path, status, created_at "
                    "FROM versions WHERE project_id = ? AND status = 'working' "
                    "AND version < ? ORDER BY version DESC LIMIT 1",
                    (project_id, fv),
                ).fetchone()

                conn.execute("COMMIT")

                if prev is None:
                    logger.warning(
                        "rollback_no_previous_working",
                        project_id=project_id, failed_version=fv,
                    )
                    return None

                prev_dict = _row_to_dict(prev)
                logger.info(
                    "rollback_done",
                    project_id=project_id,
                    failed_version=fv,
                    restored_version=prev_dict["version"],
                    restored_job_id=prev_dict["job_id"],
                )
                return prev_dict
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        finally:
            try:
                conn.close()
            except Exception:
                pass


# --- Tests-only helpers --------------------------------------------------
def reset_versions() -> int:
    """Delete all rows in versions + projects. Tests only."""
    conn = _connect()
    try:
        conn.execute("BEGIN EXCLUSIVE")
        try:
            r1 = conn.execute("DELETE FROM versions").rowcount or 0
            conn.execute("DELETE FROM projects")
            conn.execute("COMMIT")
            return r1
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
    finally:
        try:
            conn.close()
        except Exception:
            pass