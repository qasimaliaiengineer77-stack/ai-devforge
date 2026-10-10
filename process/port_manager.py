"""
AI Software Company — Port Manager (Step 14)

Architecture references:
  Section 22  Port Manager (SQLite `ports` table, range 8001-8100)
  F5          Generated apps ke ports 8001+
  F16         Port manager = SQLite UNIQUE
  F24         Port Manager race fix (asyncio.Lock)
  F27         Port Manager multi-process safe (BEGIN EXCLUSIVE)
  Section 37  Generated app ports bound to 127.0.0.1 (localhost only)

Implementation hardening (Section 22 note):
  - COMMIT only on success, ROLLBACK on exception.
  - NEVER commit unconditionally in `finally`.
  - Connection/cursor reliably closed.
  - BEGIN EXCLUSIVE wraps the INSERT to make reservation atomic even
    across processes.
  - Port allocation tests must cover: exception path, duplicate port,
    process crash, and port release.

Public API:
    async def allocate_port(job_id: str) -> int
        returns port in [8001, 8100] reserved for job_id, or raises
        NoFreePortError.

    def release_port(port: int, job_id: str | None = None) -> bool
        frees the port. If job_id is given, only frees if it matches.

    def release_ports_for_job(job_id: str) -> int
        frees all ports for a job. Returns count released.

    def get_port(job_id: str) -> int | None
        returns the port currently reserved for job_id, or None.

    def list_ports() -> list[dict]
        returns every reservation row.

    def init_ports_table() -> None
        creates the `ports` table if it does not exist (idempotent).
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from observability.logger import get_logger

logger = get_logger("process.port_manager")

# --- Config --------------------------------------------------------------
PORT_RANGE_START = 8001
PORT_RANGE_END = 8100

# DB file: same folder as the platform's ai_company.db (project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "ai_company.db"

# --- Single-process lock (F24) -------------------------------------------
# asyncio.Lock handles the common case within one running event loop
# (one process). For multi-process safety we also use BEGIN EXCLUSIVE.
_port_lock = asyncio.Lock()


class NoFreePortError(RuntimeError):
    """Raised when no port is free in the configured range."""


# --- Connection helpers --------------------------------------------------
def _connect() -> sqlite3.Connection:
    """
    Open a connection to the platform SQLite DB.
    `isolation_level=None` puts us in autocommit mode; we manage
    transactions explicitly with BEGIN EXCLUSIVE / COMMIT / ROLLBACK.
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_ports_table() -> None:
    """
    Create the `ports` table if it does not exist. Idempotent.
    Called at platform startup (api/main.py lifespan, later step).
    """
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ports (
                port INTEGER PRIMARY KEY,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_use',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ports_job_id ON ports(job_id)"
        )
        # UNIQUE constraint is implicit because port is PRIMARY KEY.


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Allocation ----------------------------------------------------------
async def allocate_port(job_id: str) -> int:
    """
    Reserve a free port in [8001, 8100] for `job_id`.

    Raises:
        NoFreePortError if no port is free.

    Guarantees:
        - On success: exactly one row was inserted and committed.
        - On failure: no partial state, connection closed.
        - Multi-process safe via BEGIN EXCLUSIVE.
    """
    if not job_id or not str(job_id).strip():
        raise ValueError("job_id must be a non-empty string")

    job_id = str(job_id).strip()

    async with _port_lock:
        conn = _connect()
        try:
            # BEGIN EXCLUSIVE gives us a write lock across processes.
            conn.execute("BEGIN EXCLUSIVE")

            try:
                # If this job already has a port, return it (idempotent).
                row = conn.execute(
                    "SELECT port FROM ports WHERE job_id = ? ORDER BY port LIMIT 1",
                    (job_id,),
                ).fetchone()
                if row is not None:
                    conn.execute("COMMIT")
                    logger.info("port_reused", job_id=job_id, port=row[0])
                    return int(row[0])

                # Find the first free port in the range.
                for candidate in range(PORT_RANGE_START, PORT_RANGE_END + 1):
                    try:
                        conn.execute(
                            "INSERT INTO ports (port, job_id, status, created_at) "
                            "VALUES (?, ?, 'in_use', ?)",
                            (candidate, job_id, _utcnow_iso()),
                        )
                        # Success: commit and return.
                        conn.execute("COMMIT")
                        logger.info("port_allocated", job_id=job_id, port=candidate)
                        return candidate
                    except sqlite3.IntegrityError:
                        # Port already taken by another row: try the next one.
                        continue

                # No free port found.
                conn.execute("ROLLBACK")
                logger.warning("port_exhausted", job_id=job_id)
                raise NoFreePortError(
                    f"no free port in range {PORT_RANGE_START}-{PORT_RANGE_END}"
                )
            except NoFreePortError:
                # Already rolled back above; re-raise.
                raise
            except Exception:
                # Roll back on any unexpected error and re-raise.
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        finally:
            # Never commit here. Just close.
            try:
                conn.close()
            except Exception:
                pass


# --- Release -------------------------------------------------------------
def release_port(port: int, job_id: Optional[str] = None) -> bool:
    """
    Free a port. If `job_id` is given, only frees it when it belongs to
    that job (safety). Returns True if a row was removed.
    """
    try:
        port_i = int(port)
    except Exception:
        return False

    conn = _connect()
    try:
        conn.execute("BEGIN EXCLUSIVE")
        try:
            if job_id is not None:
                cur = conn.execute(
                    "DELETE FROM ports WHERE port = ? AND job_id = ?",
                    (port_i, str(job_id)),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM ports WHERE port = ?",
                    (port_i,),
                )
            removed = cur.rowcount or 0
            conn.execute("COMMIT")
            if removed:
                logger.info("port_released", port=port_i, job_id=job_id)
            return removed > 0
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


def release_ports_for_job(job_id: str) -> int:
    """Free every port reserved for `job_id`. Returns count released."""
    if not job_id:
        return 0
    conn = _connect()
    try:
        conn.execute("BEGIN EXCLUSIVE")
        try:
            cur = conn.execute(
                "DELETE FROM ports WHERE job_id = ?",
                (str(job_id).strip(),),
            )
            removed = cur.rowcount or 0
            conn.execute("COMMIT")
            if removed:
                logger.info("ports_released_for_job",
                            job_id=job_id, count=removed)
            return removed
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
def get_port(job_id: str) -> Optional[int]:
    """Return the port reserved for `job_id`, or None."""
    if not job_id:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT port FROM ports WHERE job_id = ? ORDER BY port LIMIT 1",
            (str(job_id).strip(),),
        ).fetchone()
        return int(row[0]) if row else None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def list_ports() -> list[dict]:
    """Return every row in the ports table (for /metrics, tests, admin)."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT port, job_id, status, created_at FROM ports ORDER BY port"
        ).fetchall()
        return [
            {
                "port": int(r[0]),
                "job_id": r[1],
                "status": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]
    finally:
        try:
            conn.close()
        except Exception:
            pass


def reset_all_ports() -> int:
    """
    Delete every row. Used only by tests. Returns count removed.
    """
    conn = _connect()
    try:
        conn.execute("BEGIN EXCLUSIVE")
        try:
            cur = conn.execute("DELETE FROM ports")
            removed = cur.rowcount or 0
            conn.execute("COMMIT")
            return removed
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