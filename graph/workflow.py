"""
AI Software Company — LangGraph workflow (Step 16)

Architecture references:
  Section 4   Full flow
  Section 5   Repair loops (Coder max 3, Improver max 2)
  Section 16  AppState / WorkflowState
  Section 23  Human approval gate

Graph shape:
    START -> sanitize -> pm -> architect -> designer -> coder -> tester
                                       ^                          |
                                       |                          v
                                       +--- coder  <---- fail --- +--- pass ---> improver -> approval -> END
                                                                                     ^
                                                                                     |
                                                                                     +-- abort -> END

After the graph reaches "approval" and ends, the platform UI (Step 17)
shows the job to an Approval Authority. When they call
POST /approve/{job_id} with decision="yes", the API invokes
resume_deploy(job_id) which rebuilds the state from the DB / checkpointer
and runs the deploy_node.

Public API:
  build_graph() -> CompiledStateGraph
  run_to_approval(requirement, job_id, user_id, trace_id) -> WorkflowState
  resume_deploy(job_id) -> WorkflowState   (requires approval_status='approved'
                                            stored in the platform DB)
"""

from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    approval_node,
    architect_node,
    coder_node,
    deploy_node,
    designer_node,
    improver_node,
    pm_node,
    sanitize_node,
    tester_node,
)
from graph.state import (
    MAX_CODER_ATTEMPTS,
    MAX_IMPROVER_ATTEMPTS,
    WorkflowState,
    new_workflow_state,
)
from observability.logger import bind_job, get_logger

logger = get_logger("graph.workflow")


# --- Conditional edge functions ------------------------------------------
def _after_sanitize(state: WorkflowState) -> str:
    return state.get("next_route") or "abort"


def _after_pm(state: WorkflowState) -> str:
    return state.get("next_route") or "abort"


def _after_architect(state: WorkflowState) -> str:
    return state.get("next_route") or "abort"


def _after_designer(state: WorkflowState) -> str:
    return state.get("next_route") or "abort"


def _after_coder(state: WorkflowState) -> str:
    # If the coder node set next_route to abort, honor it.
    nr = state.get("next_route") or "tester"
    if nr == "abort":
        return "abort"
    return "tester"


def _after_tester(state):
    result = state.get("test_result") or "unknown"
    if result == "pass":
        return "improver"

    # If files are all generated (no failed_files) but Tester failed,
    # retrying Coder won't help - same prompt = same output.
    # Stop and let user decide (failure panel will show).
    failed_files = state.get("failed_files") or []
    code_files = state.get("code_files") or []
    if not failed_files and len(code_files) > 0:
        logger.warning("tester_fail_but_files_ok_abort",
                       files=len(code_files))
        return "abort"

    # Only retry Coder if files were actually missing/failed
    attempts = int(state.get("coder_attempts", 0) or 0)
    if attempts < MAX_CODER_ATTEMPTS:
        return "coder"
    return "abort"

def _after_improver(state: WorkflowState) -> str:
    """
    ok  -> approval
    fail with retries remaining -> coder (regenerate)
    fail exhausted -> abort
    """
    nr = state.get("next_route") or "abort"
    if nr == "approval":
        return "approval"

    # If improver failed, try coder again while attempts remain.
    coder_attempts = int(state.get("coder_attempts", 0) or 0)
    if coder_attempts < MAX_CODER_ATTEMPTS:
        return "coder"
    return "abort"


# --- Node name lookup ----------------------------------------------------
_NODE_FUNCS = {
    "sanitize": sanitize_node,
    "pm": pm_node,
    "architect": architect_node,
    "designer": designer_node,
    "coder": coder_node,
    "tester": tester_node,
    "improver": improver_node,
    "approval": approval_node,
    "deploy": deploy_node,
}


def _abort_node(state: WorkflowState) -> dict:
    """Terminal node that stops the graph cleanly."""
    return {"next_route": "aborted"}


# --- Graph builder -------------------------------------------------------
def build_graph(checkpointer=None) -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence.
                      Use SqliteSaver for local dev, PostgresSaver for prod.

    Returns the compiled graph. It is safe to reuse across jobs.
    """
    g = StateGraph(WorkflowState)

    # Add nodes
    for name, fn in _NODE_FUNCS.items():
        g.add_node(name, fn)
    g.add_node("abort", _abort_node)

    # START
    g.add_edge(START, "sanitize")

    # Linear/conditional edges for each agent
    g.add_conditional_edges(
        "sanitize", _after_sanitize,
        {"pm": "pm", "abort": "abort"},
    )
    g.add_conditional_edges(
        "pm", _after_pm,
        {"architect": "architect", "abort": "abort"},
    )
    g.add_conditional_edges(
        "architect", _after_architect,
        {"designer": "designer", "abort": "abort"},
    )
    g.add_conditional_edges(
        "designer", _after_designer,
        {"coder": "coder", "abort": "abort"},
    )
    g.add_conditional_edges(
        "coder", _after_coder,
        {"tester": "tester", "abort": "abort"},
    )
    g.add_conditional_edges(
        "tester", _after_tester,
        {"improver": "improver", "coder": "coder", "abort": "abort"},
    )
    g.add_conditional_edges(
        "improver", _after_improver,
        {"approval": "approval", "coder": "coder", "abort": "abort"},
    )

    # Approval reaches END; deploy is a separate resume step.
    g.add_edge("approval", END)

    # Abort reaches END.
    g.add_edge("abort", END)

    return g.compile(checkpointer=checkpointer)


# --- Convenience runners -------------------------------------------------
async def run_to_approval(
    requirement: str,
    job_id: str,
    user_id: str,
    trace_id: str,
) -> WorkflowState:
    """
    Run the workflow up to the approval node. Returns the final state.

    After this call, the platform DB will have a pending ApprovalRequest
    for job_id (created by approval_node). The caller should:
      - persist the state somewhere (checkpointer in a later substep), and
      - show the approval UI (Step 17).
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)
    log.info("workflow_start")
    initial = new_workflow_state(
        requirement=requirement,
        job_id=job_id,
        user_id=str(user_id),
        trace_id=trace_id,
    )
    graph = build_graph()
    final_state: WorkflowState = await graph.ainvoke(initial)
    log.info(
        "workflow_end",
        final_node=final_state.get("current_node"),
        next_route=final_state.get("next_route"),
        errors=len(final_state.get("errors", []) or []),
    )
    return final_state


async def resume_deploy(prev_state: WorkflowState) -> WorkflowState:
    """
    Resume from the approval node by running the deploy node on a state
    whose approval_status is 'approved'.

    Caller is responsible for ensuring:
      - the job's approval_status in the DB is 'approved',
      - prev_state has code_files, app_name, etc. (persisted from the
        earlier run_to_approval call).

    Here we simply run the deploy node directly (no graph traversal) so we
    do not need a checkpointer for V1.
    """
    job_id = prev_state.get("job_id", "")
    log = bind_job(
        logger,
        job_id=job_id,
        user_id=prev_state.get("user_id"),
        trace_id=prev_state.get("trace_id"),
    )
    log.info("workflow_resume_deploy")
    partial = await deploy_node(prev_state)
    merged: WorkflowState = {**prev_state, **partial}  # type: ignore[assignment]
    log.info(
        "workflow_resume_deploy_end",
        next_route=merged.get("next_route"),
        port=merged.get("port"),
    )
    return merged


async def resume_workflow(job_id: str, resume_value: dict) -> WorkflowState:
    """
    Resume a paused workflow from the last checkpoint.

    Call this after the user resolves a failure via /resolve-failure.
    The checkpointer loads the exact state where the workflow stopped.
    """
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Create checkpointer (same DB as worker)
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    log = bind_job(logger, job_id=job_id)
    log.info("workflow_resume_start", job_id=job_id)

    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": job_id}}

    try:
        # Get current state
        state = graph.get_state(config)
        log.info("resume_state_found", next_steps=state.next)

        # Resume by passing the updated state
        # We update the state with the resume_value (which may include
        # corrected code files, deleted files, etc.)
        graph.update_state(config, resume_value)

        # Continue execution from where it left off
        result = await graph.ainvoke(None, config)
        log.info("workflow_resume_done", status=result.get("status"))
        return result
    except Exception as exc:
        log.error("workflow_resume_failed", err=str(exc))
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

