"""
AI Software Company — LangGraph AppState (Step 4)

Architecture reference: Section 16 (LangGraph State).
Only the state schema is defined here. Agents and the graph are added
in later steps (Step 7 onwards). This file is the single source of truth
for state keys and their types.
"""

from typing import TypedDict


class AppState(TypedDict, total=False):
    # --- Input -----------------------------------------------------------
    requirement: str
    job_id: str
    user_id: str
    trace_id: str

    # --- PM agent --------------------------------------------------------
    app_name: str
    features: list[str]
    pages: list[str]

    # --- Architect agent -------------------------------------------------
    files: list[str]
    database: dict
    routes: list[dict]

    # --- Designer agent --------------------------------------------------
    colors: dict
    fonts: dict

    # --- Coder agent -----------------------------------------------------
    code_files: list[dict]
    rag_tokens_used: int

    # --- Tester agent ----------------------------------------------------
    test_result: str
    error: str
    attempts: int

    # --- Improver agent --------------------------------------------------
    before_scores: dict
    after_scores: dict

    # --- Deploy agent ----------------------------------------------------
    approved: bool
    github_url: str
    live_url: str
    version: str
    container_id: str
    port: int
    db_url: str

    # --- Cost tracking ---------------------------------------------------
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    model: str
    latency_ms: int

    # --- Observability ---------------------------------------------------
    logs: list[dict]


def new_state(
    requirement: str,
    job_id: str,
    user_id: str,
    trace_id: str,
) -> AppState:
    """Create a fresh AppState with only the input fields populated."""
    return AppState(
        requirement=requirement,
        job_id=job_id,
        user_id=user_id,
        trace_id=trace_id,
        features=[],
        pages=[],
        files=[],
        database={},
        routes=[],
        colors={},
        fonts={},
        code_files=[],
        rag_tokens_used=0,
        attempts=0,
        before_scores={},
        after_scores={},
        approved=False,
        logs=[],
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        estimated_cost=0.0,
    )