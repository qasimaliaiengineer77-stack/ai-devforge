"""
AI Software Company — LangGraph nodes (Step 16)

Architecture references:
  Section 4   Full flow (PM -> Architect -> Designer -> Coder ->
              Tester -> Improver -> Approval -> Deploy)
  Section 5   Repair loops (Coder max 3, Improver max 2)
  Section 16  AppState / WorkflowState
  Section 23  Human approval gate

Design:
  - Each node is an async function:
        async def name_node(state: WorkflowState) -> dict
    returning a PARTIAL dict that LangGraph merges into the state.
  - Nodes never mutate the input state dict.
  - Nodes never raise on agent failure; they record to state["errors"]
    and let the graph's conditional edges decide the next step.
  - Logger is bound per job so every line carries job_id/user_id/trace_id.
  - current_node is set at entry for observability.

Public API:
  ALL_NODE_NAMES — for wiring the graph
  sanitize_node, pm_node, architect_node, designer_node,
  coder_node, tester_node, improver_node, approval_node, deploy_node
"""

from __future__ import annotations

from typing import Any

from agents.architect_agent import ArchitectAgentError, run_architect_agent
from agents.coder_agent import CoderAgentError, run_coder_agent
from agents.designer_agent import DesignerAgentError, run_designer_agent
from agents.improver_agent import ImproverAgentError, run_improver_agent
from agents.pm_agent import PMAgentError, run_pm_agent
from agents.tester_agent import TesterAgentError, run_tester_agent
from observability.logger import bind_job, get_logger
from process.deploy_agent import DeployError, NotApprovedError, deploy
from security import prompt_guard as pg
from security.approval import request_approval

from graph.state import WorkflowState, add_error, mark_finished

logger = get_logger("graph.nodes")


ALL_NODE_NAMES = [
    "sanitize",
    "pm",
    "architect",
    "designer",
    "coder",
    "tester",
    "improver",
    "approval",
    "deploy",
]


def _log_for(state: WorkflowState):
    return bind_job(
        logger,
        job_id=state.get("job_id"),
        user_id=state.get("user_id"),
        trace_id=state.get("trace_id"),
    )


# --- Node 1: sanitize ----------------------------------------------------
async def sanitize_node(state: WorkflowState) -> dict:
    """
    L1 prompt guard on the raw requirement. Sets requirement to the
    cleaned text and, on failure, records an error so the graph can stop.
    """
    log = _log_for(state)
    log.info("node_enter", node="sanitize")
    raw = state.get("requirement", "") or ""
    san = pg.sanitize_user_input(raw)
    if not san.ok:
        log.warning("sanitize_failed", reason=san.reason)
        return {
            "current_node": "sanitize",
            "next_route": "abort",
            "errors": add_error(
                state, "sanitize", f"prompt_guard:{san.reason}",
                "requirement failed L1 sanitization",
            ),
        }
    log.info("node_exit", node="sanitize", cleaned_len=len(san.cleaned))
    return {
        "current_node": "sanitize",
        "requirement": san.cleaned,
        "next_route": "pm",
    }


# --- Node 2: PM ----------------------------------------------------------
async def pm_node(state: WorkflowState) -> dict:
    log = _log_for(state)
    log.info("node_enter", node="pm")
    try:
        brief = await run_pm_agent(
            requirement=state["requirement"],
            job_id=state["job_id"],
            user_id=str(state["user_id"]),
            trace_id=state["trace_id"],
        )
        log.info("node_exit", node="pm", features=len(brief["features"]))
        return {
            "current_node": "pm",
            "app_name": brief["app_name"],
            "features": brief["features"],
            "pages": brief["pages"],
            "next_route": "architect",
        }
    except PMAgentError as exc:
        log.warning("node_failed", node="pm", err=str(exc))
        return {
            "current_node": "pm",
            "next_route": "abort",
            "errors": add_error(state, "pm", "pm_failed", str(exc)),
        }


# --- Node 3: Architect ---------------------------------------------------
async def architect_node(state: WorkflowState) -> dict:
    log = _log_for(state)
    log.info("node_enter", node="architect")
    pm_brief = {
        "app_name": state.get("app_name", ""),
        "features": state.get("features", []),
        "pages": state.get("pages", []),
    }
    try:
        plan = await run_architect_agent(
            pm_brief=pm_brief,
            job_id=state["job_id"],
            user_id=str(state["user_id"]),
            trace_id=state["trace_id"],
        )
        log.info(
            "node_exit", node="architect",
            files=len(plan["files"]),
            tables=len(plan["database"]["tables"]),
        )
        return {
            "current_node": "architect",
            "files": plan["files"],
            "database": plan["database"],
            "routes": plan["routes"],
            "next_route": "designer",
        }
    except ArchitectAgentError as exc:
        log.warning("node_failed", node="architect", err=str(exc))
        return {
            "current_node": "architect",
            "next_route": "abort",
            "errors": add_error(state, "architect", "architect_failed", str(exc)),
        }


# --- Node 4: Designer ----------------------------------------------------
async def designer_node(state: WorkflowState) -> dict:
    log = _log_for(state)
    log.info("node_enter", node="designer")
    pm_brief = {
        "app_name": state.get("app_name", ""),
        "features": state.get("features", []),
        "pages": state.get("pages", []),
    }
    plan = {
        "files": state.get("files", []),
        "database": state.get("database", {}),
        "routes": state.get("routes", []),
    }
    try:
        design = await run_designer_agent(
            pm_brief=pm_brief,
            plan=plan,
            job_id=state["job_id"],
            user_id=str(state["user_id"]),
            trace_id=state["trace_id"],
        )
        log.info("node_exit", node="designer",
                 chosen=design["colors"]["chosen"])
        return {
            "current_node": "designer",
            "colors": design["colors"],
            "fonts": design["fonts"],
            "design_brief": design,
            "next_route": "coder",
        }
    except DesignerAgentError as exc:
        log.warning("node_failed", node="designer", err=str(exc))
        return {
            "current_node": "designer",
            "next_route": "abort",
            "errors": add_error(state, "designer", "designer_failed", str(exc)),
        }


# --- Node 5: Coder -------------------------------------------------------
async def coder_node(state: WorkflowState) -> dict:
    log = _log_for(state)
    attempt = int(state.get("coder_attempts", 0) or 0) + 1
    log.info("node_enter", node="coder", attempt=attempt)
    pm_brief = {
        "app_name": state.get("app_name", ""),
        "features": state.get("features", []),
        "pages": state.get("pages", []),
    }
    plan = {
        "files": state.get("files", []),
        "database": state.get("database", {}),
        "routes": state.get("routes", []),
    }
    design = state.get("design_brief") or {
        "colors": state.get("colors", {}),
        "fonts": state.get("fonts", {}),
    }
    try:
        result = await run_coder_agent(
            pm_brief=pm_brief,
            plan=plan,
            design=design,
            job_id=state["job_id"],
            user_id=str(state["user_id"]),
            trace_id=state["trace_id"],
        )
        log.info(
            "node_exit", node="coder",
            files_ok=len(result["files"]),
            files_failed=len(result["failed_files"]),
        )
        return {
            "current_node": "coder",
            "code_files": result["files"],
            "failed_files": result["failed_files"],
            "rag_tokens_used": result["rag_tokens_used"],
            "coder_attempts": attempt,
            "next_route": "tester",
        }
    except CoderAgentError as exc:
        log.warning("node_failed", node="coder", err=str(exc))
        return {
            "current_node": "coder",
            "coder_attempts": attempt,
            "next_route": "abort",
            "errors": add_error(state, "coder", "coder_failed", str(exc),
                                attempt=attempt),
        }


# --- Node 6: Tester ------------------------------------------------------
async def tester_node(state: WorkflowState) -> dict:
    log = _log_for(state)
    log.info("node_enter", node="tester")
    attempt = int(state.get("tester_attempts", 0) or 0) + 1
    files = state.get("code_files", []) or []
    try:
        res = await run_tester_agent(
            files=files,
            job_id=state["job_id"],
            user_id=str(state["user_id"]),
            trace_id=state["trace_id"],
        )
        status = "pass" if res.ok else "fail"
        issues = list(res.l1_issues) + list(res.l2_issues) + list(res.l3_issues)
        log.info("node_exit", node="tester", status=status,
                 tests_run=res.tests_run, tests_passed=res.tests_passed)
        return {
            "current_node": "tester",
            "test_result": status,
            "test_error": res.error or "",
            "tester_attempts": attempt,
            "tester_l1_ok": res.l1_ok,
            "tester_l2_ok": res.l2_ok,
            "tester_l3_ok": res.l3_ok,
            "tester_issues": issues,
            "next_route": "improver" if res.ok else "coder",
        }
    except TesterAgentError as exc:
        log.warning("node_failed", node="tester", err=str(exc))
        return {
            "current_node": "tester",
            "test_result": "fail",
            "test_error": str(exc),
            "tester_attempts": attempt,
            "next_route": "abort",
            "errors": add_error(state, "tester", "tester_failed", str(exc),
                                attempt=attempt),
        }


# --- Node 7: Improver ----------------------------------------------------
async def improver_node(state: WorkflowState) -> dict:
    log = _log_for(state)
    attempt = int(state.get("improver_attempts", 0) or 0) + 1
    log.info("node_enter", node="improver", attempt=attempt)
    files = state.get("code_files", []) or []
    try:
        res = await run_improver_agent(
            files=files,
            job_id=state["job_id"],
            user_id=str(state["user_id"]),
            trace_id=state["trace_id"],
            run_lighthouse=True,
        )
        log.info(
            "node_exit", node="improver",
            ok=res.ok, scores=res.before_scores,
        )
        return {
            "current_node": "improver",
            "before_scores": res.before_scores,
            "after_scores": res.after_scores,
            "improver_issues": res.issues,
            "improver_attempts": attempt,
            "next_route": "approval" if res.ok else "abort",
            "errors": (
                add_error(state, "improver", "lighthouse_failed", res.error)
                if not res.ok else state.get("errors", [])
            ),
        }
    except ImproverAgentError as exc:
        log.warning("node_failed", node="improver", err=str(exc))
        return {
            "current_node": "improver",
            "improver_attempts": attempt,
            "next_route": "abort",
            "errors": add_error(state, "improver", "improver_failed",
                                str(exc), attempt=attempt),
        }


# --- Node 8: Approval ----------------------------------------------------
async def approval_node(state: WorkflowState) -> dict:
    """
    Create (or refresh) a pending approval request. Does NOT block — the
    graph stops at this node when next_route != 'deploy'. The platform UI
    (Step 17) polls /approval/{job_id} and calls /approve/{job_id}.
    """
    log = _log_for(state)
    log.info("node_enter", node="approval")
    job_id = state["job_id"]
    try:
        user_id = int(state["user_id"])
    except Exception:
        user_id = None
    try:
        from db.database import SessionLocal  # local import to avoid cycle
        db = SessionLocal()
        try:
            req = request_approval(db, job_id, requested_by=user_id)
            status = req.status
        finally:
            db.close()
    except Exception as exc:
        log.warning("approval_request_failed", err=str(exc))
        return {
            "current_node": "approval",
            "approval_status": "none",
            "next_route": "abort",
            "errors": add_error(state, "approval", "approval_request_failed",
                                str(exc)),
        }
    log.info("node_exit", node="approval", status=status)
    return {
        "current_node": "approval",
        "approval_status": status,
        "approved": False,  # becomes True once decide_approval runs
        "next_route": "wait_approval",  # graph stops; UI drives the rest
    }


# --- Node 9: Deploy ------------------------------------------------------
async def deploy_node(state: WorkflowState) -> dict:
    """
    Deploy the generated app locally and push to GitHub (if creds exist).
    Only callable when approval_status == 'approved'.
    """
    log = _log_for(state)
    log.info("node_enter", node="deploy")
    approved = (state.get("approval_status") or "none") == "approved"
    if not approved:
        log.warning("deploy_blocked_not_approved")
        return {
            "current_node": "deploy",
            "next_route": "abort",
            "errors": add_error(state, "deploy", "not_approved",
                                "deploy called without approval"),
        }
    files = state.get("code_files", []) or []
    project_id = state.get("app_name", "app").lower().replace(" ", "-") or "app"
    try:
        res = await deploy(
            files=files,
            job_id=state["job_id"],
            project_id=project_id,
            user_id=str(state["user_id"]),
            trace_id=state["trace_id"],
            approval=True,
        )
        log.info("node_exit", node="deploy", ok=res.ok,
                 port=res.port, version=res.version)
        return {
            "current_node": "deploy",
            "approved": True,
            "port": res.port or 0,
            "live_url": res.host_url or "",
            "container_id": res.container_name or "",
            "version": str(res.version or ""),
            "github_url": res.github.repo_url if res.github.ok else "",
            "next_route": "done" if res.ok else "abort",
            "errors": (
                add_error(state, "deploy", res.error or "deploy_failed",
                          "deploy failed")
                if not res.ok else state.get("errors", [])
            ),
            "finished_at": mark_finished(state),
        }
    except NotApprovedError as exc:
        return {
            "current_node": "deploy",
            "next_route": "abort",
            "errors": add_error(state, "deploy", "not_approved", str(exc)),
        }
    except DeployError as exc:
        return {
            "current_node": "deploy",
            "next_route": "abort",
            "errors": add_error(state, "deploy", "deploy_error", str(exc)),
        }