"""
AI Software Company — Tester Agent (Step 12)

Architecture references:
  Section 5   Tester: Docker sandbox (256MB, 0.5 CPU), 3-level check, repair max 3
  Section 27  Timeouts: pip 120s, tests 10s
  Section 35  2-phase network, no subprocess fallback
  Section 16  AppState keys: test_result, error, attempts
  Section 24  Audit actions: test_started, test_failed, test_passed, repair_started

Public API:
  async def run_tester_agent(files, job_id, user_id, trace_id, output_root,
                             max_repairs=3)
      -> TestResult
      raises TesterAgentError on unexpected internal failure.

Note: repair itself (regenerating code) is delegated back to the Coder in
Step 16 (LangGraph workflow). This agent only tests and produces a report.
The repair loop is a placeholder that records the failure reason and returns
so the graph can route back to the Coder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from observability.logger import bind_job, get_logger
from security.static_checks import run_all as run_static_checks
from sandbox.docker_runner import (
    SandboxResult,
    cleanup_sandbox,
    run_sandbox,
)

logger = get_logger("agents.tester")

# Path where sandbox/test_runner.py lives (host side)
_TEST_SCRIPT = Path(__file__).resolve().parent.parent / "sandbox" / "test_runner.py"

# Where generated apps are written for sandbox runs
_DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"


class TesterAgentError(RuntimeError):
    """Raised when the tester cannot run at all (e.g., test script missing)."""


@dataclass
class TestResult:
    ok: bool = False
    attempts: int = 0
    error: str = ""
    sandbox: SandboxResult = field(default_factory=SandboxResult)
    l1_ok: bool = False
    l2_ok: bool = False
    l3_ok: bool = False
    l1_issues: list = field(default_factory=list)
    l2_issues: list = field(default_factory=list)
    l3_issues: list = field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0
    duration_ms: int = 0
    cleanup_done: bool = False


def _summarize_issues(prefix: str, issues: list, limit: int = 5) -> str:
    """Short human-readable summary of first N issues."""
    if not issues:
        return ""
    head = issues[:limit]
    joined = "; ".join(str(x) for x in head)
    if len(issues) > limit:
        joined += f"; +{len(issues) - limit} more"
    return f"{prefix}: {joined}"


async def run_tester_agent(
    files: list[dict],
    job_id: str,
    user_id: str,
    trace_id: str,
    output_root: Optional[Path] = None,
    max_repairs: int = 3,
) -> TestResult:
    """
    Run the Tester agent on generated files.

    Returns a TestResult with ok=True if all 3 levels passed.
    Never raises on test failure (only on missing test runner file).
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)
    log.info("agent_started", agent="tester", files=len(files or []))

    if not _TEST_SCRIPT.exists():
        raise TesterAgentError(f"test_runner_missing:{_TEST_SCRIPT}")

    if not files:
        r = TestResult(ok=False, error="no_files_to_test")
        log.warning("tester_no_files")
        return r

    root = Path(output_root) if output_root else _DEFAULT_OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    result = TestResult()
    t0 = time.perf_counter()

    # In this step, we run the sandbox once and report the outcome.
    # The repair loop belongs to the graph (Step 16), so max_repairs is
    # recorded but the actual re-generation is delegated upstream.
    log.info("test_started", phase=1)

    # -------- Dump files to disk FIRST (so we can inspect on failure) --------
    import shutil as _sh
    _dump_dir = root / job_id
    if _dump_dir.exists():
        _sh.rmtree(_dump_dir, ignore_errors=True)
    _dump_dir.mkdir(parents=True, exist_ok=True)
    for _f in files:
        _fp = _dump_dir / _f["path"]
        _fp.parent.mkdir(parents=True, exist_ok=True)
        _fp.write_text(_f["content"], encoding="utf-8")
    log.info("tester_dumped_files", count=len(files), path=str(_dump_dir))

    # -------- Static checks BEFORE Docker (fast, no build) --------
    static_issues = run_static_checks(files)
    if static_issues:
        log.warning("static_check_failed", count=len(static_issues),
                    sample=static_issues[:5])
        result.ok = False
        result.error = "static_check_failed: " + "; ".join(static_issues[:8])
        result.l2_issues = list(static_issues)
        result.duration_ms = int((time.perf_counter() - t0) * 1000)
        log.info("agent_completed", agent="tester", ok=False,
                 stage="static", duration_ms=result.duration_ms)
        return result

    log.info("static_check_passed")

    sandbox_result = run_sandbox(
        files=files,
        job_id=job_id,
        output_root=root,
        test_script_path=_TEST_SCRIPT,
    )
    result.sandbox = sandbox_result
    result.attempts = 1

    report = sandbox_result.test_report or {}
    result.l1_ok = bool(report.get("l1_ok", False))
    result.l2_ok = bool(report.get("l2_ok", False))
    result.l3_ok = bool(report.get("l3_ok", False))
    result.l1_issues = list(report.get("l1_issues") or [])
    result.l2_issues = list(report.get("l2_issues") or [])
    result.l3_issues = list(report.get("l3_issues") or [])
    result.tests_run = int(report.get("tests_run", 0) or 0)
    result.tests_passed = int(report.get("tests_passed", 0) or 0)

    if not sandbox_result.ok:
        # Compose a short reason combining phase and issues
        reason_parts = [sandbox_result.error or "sandbox_failed"]
        for prefix, issues in (
            ("L1", result.l1_issues),
            ("L2", result.l2_issues),
            ("L3", result.l3_issues),
        ):
            s = _summarize_issues(prefix, issues)
            if s:
                reason_parts.append(s)
        result.error = " | ".join(reason_parts)[:800]
        log.warning("test_failed", error=result.error,
                    l1=result.l1_ok, l2=result.l2_ok, l3=result.l3_ok)
    else:
        result.ok = True
        log.info(
            "test_passed",
            tests_run=result.tests_run,
            tests_passed=result.tests_passed,
            l1=result.l1_ok, l2=result.l2_ok, l3=result.l3_ok,
        )

    # Cleanup sandbox workspace (skip if KEEP_WORKSPACE=1 for debugging)
    import os as _os
    if _os.getenv("KEEP_WORKSPACE", "1") == "1":
        log.info("sandbox_workspace_kept", path=str(root / job_id))
    else:
        try:
            cleanup_sandbox(job_id, root)
            result.cleanup_done = True
        except Exception as exc:
            log.warning("sandbox_cleanup_failed", err=str(exc))

    result.duration_ms = int((time.perf_counter() - t0) * 1000)
    log.info(
        "agent_completed",
        agent="tester",
        ok=result.ok,
        attempts=result.attempts,
        duration_ms=result.duration_ms,
    )
    return result