r"""
AI Software Company â€” ARQ build worker (Step 17)

Architecture references:
  Section 26  ARQ job queue, max_concurrent_builds=3, job_timeout=15min
  Section 4   Full flow (up to approval)
  Section 18  Status page polls every 3s
  Section 23  Human approval gate (worker stops at approval node)

Flow:
  1. ARQ delivers (ctx, job_id) from the queue.
  2. We load the job row, mark it "building".
  3. We build the LangGraph StateGraph and astream it â€” each node's
     partial updates are merged into the running state, and the jobs
     table is updated after every node so the UI can display progress.
  4. When the graph reaches the approval node (or an abort), the worker
     sets the final status and exits. The platform then waits for the
     Approval Authority to decide via POST /approve/{job_id}.
  5. After approval, the API calls graph.workflow.resume_deploy() which
     runs the deploy_node directly (not through the worker).

Run:
  In a dedicated PowerShell window:
      cd $env:USERPROFILE\Desktop\ai-company
      .\\venv\\Scripts\\Activate.ps1
      python -m arq workers.build_worker.WorkerSettings
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from arq.connections import RedisSettings
from dotenv import load_dotenv

from db import jobs_store
from db import failures_store
from graph.state import new_workflow_state
from graph.workflow import build_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from observability.logger import bind_job, get_logger
from workers.queue import _redis_settings_from_url

load_dotenv()

logger = get_logger("workers.build_worker")

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")


# --- Node -> job status mapping -----------------------------------------
_NODE_TO_STATUS = {
    "sanitize": "building",
    "pm": "building",
    "architect": "building",
    "designer": "building",
    "coder": "building",
    "tester": "testing",
    "improver": "improving",
    "approval": "awaiting_approval",
    "deploy": "deploying",
    "abort": "failed",
}


# --- DB update after each node -------------------------------------------
def _update_job_from_state(
    job_id: str,
    node_name: str,
    state: dict,
) -> None:
    """Push the current workflow state into the jobs table for the UI."""
    try:
        status = _NODE_TO_STATUS.get(node_name, "building")

        next_route = state.get("next_route", "") or ""
        if next_route == "abort":
            status = "failed"

        # Terminal statuses win over node mapping
        if state.get("approval_status") == "pending":
            status = "awaiting_approval"
        if state.get("finished_at"):
            # Deploy completed inside the workflow (not typical â€” deploy
            # is normally a separate resume step).
            status = "done"

        jobs_store.update_job(job_id, status=status, current_node=node_name)

        # Persist features (once available)
        features = state.get("features") or []
        if features:
            jobs_store.set_features(job_id, features)

        # Persist file list (paths + sizes only)
        code_files = state.get("code_files") or []
        if code_files:
            jobs_store.set_files(job_id, code_files)

        # Persist scores (before_scores is what Improver produces)
        scores = state.get("before_scores") or state.get("after_scores") or {}
        if scores:
            jobs_store.set_scores(job_id, scores)

        # Persist the most recent error, if any
        errors = state.get("errors") or []
        if errors:
            last = errors[-1] or {}
            jobs_store.set_error(
                job_id,
                agent=str(last.get("node", node_name)),
                code=str(last.get("error_code", "unknown")),
                message=str(last.get("message", "")),
                attempt=last.get("attempt"),
            )
    except Exception as exc:
        # Never let a DB hiccup kill the workflow; log and continue.
        logger.warning(
            "worker_state_update_failed",
            job_id=job_id, node=node_name, err=type(exc).__name__,
        )


# --- Worker entry point --------------------------------------------------
async def build_workflow(ctx: dict, job_id: str) -> dict:
    """
    ARQ task. Runs the workflow up to the approval node, updating the
    jobs table after each node so the platform UI can poll progress.
    """
    t0 = time.perf_counter()
    log = bind_job(logger, job_id=job_id)
    log.info("worker_start")

    job = jobs_store.get_job(job_id)
    if job is None:
        log.error("worker_job_not_found")
        return {"status": "not_found", "job_id": job_id}

    user_id = str(job.user_id) if job.user_id else "0"
    requirement = job.requirement or ""
    trace_id = job_id

    try:
        jobs_store.update_job(
            job_id, status="building", current_node="starting"
        )

        initial = new_workflow_state(
            requirement=requirement,
            job_id=job_id,
            user_id=user_id,
            trace_id=trace_id,
        )

        # Create SQLite checkpointer (WAL mode + integrity check for crash safety)
        from pathlib import Path as _Path
        _cp = _Path("checkpoints.db")

        # Check if existing DB is corrupt
        if _cp.exists():
            try:
                _test = sqlite3.connect(str(_cp), check_same_thread=False)
                _test.execute("PRAGMA integrity_check").fetchone()
                _test.close()
            except Exception as _e:
                logger.warning("checkpointer_corrupt_deleting", err=str(_e))
                try:
                    _cp.unlink()
                except Exception:
                    pass

        import aiosqlite
        conn = await aiosqlite.connect("checkpoints.db")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": job_id}}

        # Running state: we merge each node's partial update here and use
        # it to update the jobs table. `astream(stream_mode="updates")`
        # yields {node_name: partial_dict} per node completion.
        running: dict[str, Any] = dict(initial)

        async for event in graph.astream(initial, config, stream_mode="updates"):
            for node_name, partial in event.items():
                if partial:
                    running.update(partial)
                _update_job_from_state(job_id, node_name, running)

        # Failure detection: file failures, workflow errors,
        # OR tester fail with all files generated.
        failed_files = running.get("failed_files") or []
        errors = running.get("errors") or []
        test_result = running.get("test_result", "")
        test_error = running.get("test_error", "")

        tester_failed_alone = (
            test_result == "fail"
            and not failed_files
            and len(running.get("code_files") or []) > 0
        )

        if failed_files or errors or tester_failed_alone:
            saved = 0
            for ff in failed_files:
                try:
                    failures_store.save_failure(
                        job_id=job_id,
                        file_path=ff.get("path", ""),
                        error_code=ff.get("reason", "unknown"),
                        error_message=ff.get("reason", ""),
                        ai_content=ff.get("content", "") or "",
                    )
                    saved += 1
                except Exception as exc:
                    logger.warning("save_failure_failed",
                                   file_path=ff.get("path"),
                                   err=type(exc).__name__)

            # Errors without file failures (Groq rate limit etc.) still
            # need a single job-level failure row so UI shows a panel.
            if errors and not failed_files:
                last_err = errors[-1] or {}
                try:
                    failures_store.save_job_failure(
                        job_id=job_id,
                        error_code=last_err.get("error_code", "unknown"),
                        error_message=last_err.get("message", ""),
                    )
                    saved += 1
                except Exception as exc:
                    logger.warning("save_job_failure_failed",
                                   err=type(exc).__name__)

            # Tester failed but files are OK: save a job-level row with the
            # tester's error so user sees the reason (L1/L2/L3 details).
            if tester_failed_alone:
                l1_ok = running.get("tester_l1_ok", False)
                l2_ok = running.get("tester_l2_ok", False)
                l3_ok = running.get("tester_l3_ok", False)
                issues = running.get("tester_issues") or []
                detail_lines = []
                if not l1_ok: detail_lines.append("L1: static security checks failed")
                if not l2_ok: detail_lines.append("L2: syntax check failed")
                if not l3_ok: detail_lines.append("L3: HTTP tests failed")
                if issues:
                    for it in issues[:5]:
                        detail_lines.append("  - " + str(it)[:120])
                detail = " | ".join(detail_lines) or (test_error or "tester failed")
                try:
                    failures_store.save_job_failure(
                        job_id=job_id,
                        error_code="tester_failed",
                        error_message=detail[:900],
                    )
                    saved += 1
                except Exception as exc:
                    logger.warning("save_tester_failure_failed",
                                   err=type(exc).__name__)

            jobs_store.update_job(
                job_id,
                status="awaiting_user_input",
                current_node=running.get("current_node", ""),
            )
            dur = int((time.perf_counter() - t0) * 1000)
            log.info("worker_stopped_for_user_input",
                     failed_count=len(failed_files),
                     error_count=len(errors),
                     saved=saved,
                     duration_ms=dur)
            return {"status": "awaiting_user_input", "job_id": job_id,
                    "failed_count": len(failed_files),
                    "error_count": len(errors)}

        # Final status determination
        next_route = running.get("next_route", "") or ""
        approval_status = running.get("approval_status", "none")
        test_result = running.get("test_result", "")
        coder_attempts = int(running.get("coder_attempts", 0) or 0)

        if approval_status == "pending":
            final_status = "awaiting_approval"
        elif next_route == "abort" or running.get("errors"):
            final_status = "failed"
        elif test_result == "fail":
            # Tester failed and attempts are exhausted -> workflow aborted
            # even though the abort node set next_route itself.
            final_status = "failed"
        elif not running.get("code_files"):
            # Coder never produced files
            final_status = "failed"
        else:
            final_status = "done"

        jobs_store.update_job(
            job_id,
            status=final_status,
            current_node=running.get("current_node", ""),
        )

        dur = int((time.perf_counter() - t0) * 1000)
        log.info("worker_end", status=final_status, duration_ms=dur)
        return {"status": final_status, "job_id": job_id}

    except Exception as exc:
        log.error(
            "worker_exception",
            err=f"{type(exc).__name__}: {exc}",
        )
        try:
            jobs_store.set_error(
                job_id, "worker", "worker_exception", str(exc)[:500]
            )
            jobs_store.update_job(job_id, status="failed")
        except Exception:
            pass
        return {"status": "failed", "error": str(exc)[:200], "job_id": job_id}


# --- ARQ lifecycle hooks -------------------------------------------------
async def on_startup(ctx: dict) -> None:
    logger.info("worker_startup", redis=REDIS_URL.split("@")[-1])


async def on_shutdown(ctx: dict) -> None:
    logger.info("worker_shutdown")


# --- ARQ WorkerSettings --------------------------------------------------
class WorkerSettings:
    """Referenced by `python -m arq workers.build_worker.WorkerSettings`."""
    redis_settings = _redis_settings_from_url(REDIS_URL)
    functions = [build_workflow]

    # Section 26 config
    max_jobs = 3            # max_concurrent_builds
    job_timeout = 1800      # 30 minutes (raised from 15 after real-world timing)
    keep_result = 3600      # keep result 1 hour
    max_tries = 1           # internal retries are handled by the workflow
    allow_abort_jobs = True

    on_startup = on_startup
    on_shutdown = on_shutdown


