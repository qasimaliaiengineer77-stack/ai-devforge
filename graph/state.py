"""
AI Software Company — LangGraph workflow state (Step 16)

Architecture references:
  Section 16  AppState schema
  Section 4   Full flow (PM -> Architect -> Designer -> Coder ->
              Tester -> Improver -> Approval -> Deploy)
  Section 5   Repair loops (Coder max 3, Improver max 2)
  Section 23  Human approval gate

Design:
  - Extends the base AppState from ai/state.py (Section 16).
  - Adds LangGraph-specific routing fields:
        current_node       — which agent is running (for logging/UI)
        next_route         — what the graph should do next (for conditional edges)
        coder_attempts     — number of times Coder ran for this job
        improver_attempts  — number of times Improver ran
        approval_status    — "pending" | "approved" | "rejected" | "none"
        errors             — list of {node, error_code, message, attempt}
        started_at         — ISO timestamp when the workflow started
        finished_at        — ISO timestamp when it finished (or None)

  - We keep the base AppState's typed fields (requirement, job_id, ...
    before_scores, after_scores, port, live_url, etc.) so all existing
    agents can consume/return them unchanged.

  - LangGraph requires a plain TypedDict. We use total=False so partial
    updates work naturally from node returns.

Public API:
    WorkflowState — TypedDict for LangGraph
    new_workflow_state(requirement, job_id, user_id, trace_id) -> WorkflowState
    bump_attempt(state, key) -> dict  (helper for node returns)
    add_error(state, node, error_code, message, attempt) -> list
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, TypedDict


MAX_CODER_ATTEMPTS = 3
MAX_IMPROVER_ATTEMPTS = 2

VALID_APPROVAL_STATUSES = {"none", "pending", "approved", "rejected"}


class WorkflowState(TypedDict, total=False):
    # --- Input (Section 16) ---
    requirement: str
    job_id: str
    user_id: str
    trace_id: str

    # --- PM agent ---
    app_name: str
    features: list[str]
    pages: list[str]

    # --- Architect agent ---
    files: list[str]
    database: dict
    routes: list[dict]

    # --- Designer agent ---
    colors: dict
    fonts: dict
    design_brief: dict          # full design brief (colors+fonts+spacing+responsive)

    # --- Coder agent ---
    code_files: list[dict]      # [{"path": str, "content": str}, ...]
    rag_tokens_used: int
    failed_files: list[dict]    # [{"path": str, "reason": str}, ...]

    # --- Tester agent ---
    test_result: str            # "pass" | "fail" | "unknown"
    test_error: str
    tester_attempts: int
    tester_l1_ok: bool
    tester_l2_ok: bool
    tester_l3_ok: bool
    tester_issues: list

    # --- Improver agent ---
    before_scores: dict
    after_scores: dict
    improver_issues: list
    improver_attempts: int

    # --- Deploy agent ---
    approved: bool
    approval_status: str        # "none" | "pending" | "approved" | "rejected"
    github_url: str
    live_url: str
    version: str
    container_id: str
    port: int
    db_url: str

    # --- Cost tracking (Section 16 + Section 20) ---
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    model: str
    latency_ms: int

    # --- Observability (Section 16) ---
    logs: list[dict]

    # --- LangGraph routing / meta ---
    current_node: str
    next_route: str
    coder_attempts: int
    errors: list[dict]
    started_at: str
    finished_at: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_workflow_state(
    requirement: str,
    job_id: str,
    user_id: str,
    trace_id: str,
) -> WorkflowState:
    """
    Create a fresh WorkflowState with only the input and defaults populated.
    All agent output fields start empty.
    """
    return WorkflowState(
        # Input
        requirement=requirement,
        job_id=job_id,
        user_id=user_id,
        trace_id=trace_id,
        # PM
        app_name="",
        features=[],
        pages=[],
        # Architect
        files=[],
        database={},
        routes=[],
        # Designer
        colors={},
        fonts={},
        design_brief={},
        # Coder
        code_files=[],
        rag_tokens_used=0,
        failed_files=[],
        # Tester
        test_result="unknown",
        test_error="",
        tester_attempts=0,
        tester_l1_ok=False,
        tester_l2_ok=False,
        tester_l3_ok=False,
        tester_issues=[],
        # Improver
        before_scores={},
        after_scores={},
        improver_issues=[],
        improver_attempts=0,
        # Deploy
        approved=False,
        approval_status="none",
        github_url="",
        live_url="",
        version="",
        container_id="",
        port=0,
        db_url="",
        # Cost
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        estimated_cost=0.0,
        model="",
        latency_ms=0,
        # Observability
        logs=[],
        # LangGraph meta
        current_node="",
        next_route="",
        coder_attempts=0,
        errors=[],
        started_at=_utcnow_iso(),
        finished_at="",
    )


def add_error(
    state: WorkflowState,
    node: str,
    error_code: str,
    message: str,
    attempt: Optional[int] = None,
) -> list[dict]:
    """
    Return a NEW errors list with one entry appended. Do not mutate state.
    LangGraph merges partial node returns into the state.
    """
    errors = list(state.get("errors") or [])
    entry: dict[str, Any] = {
        "node": node,
        "error_code": error_code,
        "message": str(message)[:500],
        "at": _utcnow_iso(),
    }
    if attempt is not None:
        entry["attempt"] = int(attempt)
    errors.append(entry)
    return errors


def bump(state: WorkflowState, key: str, amount: int = 1) -> int:
    """Return the new integer value for `key` after adding `amount`."""
    return int(state.get(key, 0) or 0) + int(amount)


def is_approved(state: WorkflowState) -> bool:
    """True iff approval_status is 'approved'."""
    return (state.get("approval_status") or "none") == "approved"


def mark_finished(state: WorkflowState) -> str:
    """Return the ISO timestamp to set as finished_at."""
    return _utcnow_iso()