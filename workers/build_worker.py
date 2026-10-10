r"""
AI Software Company — ARQ build worker (Step 17)

Architecture references:
  Section 26  ARQ job queue, max_concurrent_builds=3, job_timeout=15min
  Section 4   Full flow (up to approval)
  Section 18  Status page polls every 3s
  Section 23  Human approval gate (worker stops at approval node)

Flow:
  1. ARQ delivers (ctx, job_id) from the queue.
  2. We load the job row, mark it "building".
  3. We build the LangGraph StateGraph and astream it — each node's
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
import time
from typing import Any

from arq.connections import RedisSettings
from dotenv import load_dotenv

from db import jobs_store
from graph.state import new_workflow_state
from graph.workflow import build_graph
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
            # Deploy completed inside the workflow (not typical — deploy
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

        graph = build_graph()

        # Running state: we merge each node's partial update here and use
        # it to update the jobs table. `astream(stream_mode="updates")`
        # yields {node_name: partial_dict} per node completion.
        running: dict[str, Any] = dict(initial)

        async for event in graph.astream(initial, stream_mode="updates"):
            for node_name, partial in event.items():
                if partial:
                    running.update(partial)
                _update_job_from_state(job_id, node_name, running)

        # Final status determination
        next_route = running.get("next_route", "") or ""
        approval_status = running.get("approval_status", "none")

        if approval_status == "pending":
            final_status = "awaiting_approval"
        elif next_route == "abort" or running.get("errors"):
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
    job_timeout = 900       # 15 minutes
    keep_result = 3600      # keep result 1 hour
    max_tries = 1           # internal retries are handled by the workflow
    allow_abort_jobs = True

    on_startup = on_startup
    on_shutdown = on_shutdown