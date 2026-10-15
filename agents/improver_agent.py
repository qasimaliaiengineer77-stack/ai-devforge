"""
AI Software Company — Improver Agent (Step 13)

Architecture references:
  Section 5   Improver: Lighthouse CLI (Docker), before/after scores
  Section 36  Lighthouse Docker
  Section 27  Lighthouse timeout 120s
  Section 16  AppState keys: before_scores, after_scores
  F6          Lighthouse --no-sandbox
  F12         Lighthouse internet (Tailwind CDN)
  F22         Shared network ai_company_network
  F23         Lighthouse URL via service name (http://app-{job_id}:8000)

Public API:
  async def run_improver_agent(files, job_id, user_id, trace_id,
                               output_root, run_lighthouse=True) -> ImproverResult

Flow:
  1. Build app image via sandbox.app_runner.build_app_image
  2. Run container on ai_company_network (no host port needed)
  3. Wait for uvicorn "startup complete" in logs
  4. Run Lighthouse container on same network against service URL
  5. Parse JSON report -> categories scores (0..100)
  6. Cleanup: stop container, remove image
  7. Return ImproverResult with before_scores (and after_scores same
     for now; actual improvement is a Step 16 graph concern)

Verified configuration (manual test PASSED with these exact settings):
  IMAGE       : femtopixel/google-lighthouse:latest
  CAPS        : --cap-add SYS_ADMIN   (Chrome sandbox needs it)
  SHM         : --shm-size=2g         (Docker default 64MB is too small)
  CHROME_FLAGS: --headless=new --no-sandbox --disable-dev-shm-usage
                --disable-setuid-sandbox --disable-gpu
  OUTPUT      : --output=json (stdout, NOT --output-path)
  NO          : custom --remote-debugging-port (image manages its own)
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from observability.logger import bind_job, get_logger
from sandbox.app_runner import (
    build_app_image,
    cleanup_app,
    run_app_container,
    wait_until_ready,
)

logger = get_logger("agents.improver")

LIGHTHOUSE_IMAGE = "femtopixel/google-lighthouse:latest"
LIGHTHOUSE_TIMEOUT_S = 180          # 120s official + 30s buffer
LIGHTHOUSE_CATEGORIES = "performance,accessibility,best-practices,seo"
LIGHTHOUSE_SHM_SIZE = "2g"

# Chrome flags for headless Chrome inside Docker.
# See module docstring for why each flag is here.
LIGHTHOUSE_CHROME_FLAGS = (
    "--headless=new "
    "--no-sandbox "
    "--disable-dev-shm-usage "
    "--disable-setuid-sandbox "
    "--disable-gpu"
)


class ImproverAgentError(RuntimeError):
    """Raised when the improver cannot run at all."""


@dataclass
class ImproverResult:
    ok: bool = False
    before_scores: dict = field(default_factory=dict)
    after_scores: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    error: str = ""
    duration_ms: int = 0
    app_container: str = ""
    app_service_url: str = ""
    lighthouse_returncode: int = 0
    lighthouse_log: str = ""


# --- Lighthouse invocation ----------------------------------------------
def _run_lighthouse(url: str) -> dict:
    """
    Run Lighthouse container against `url` on the shared network.
    Returns dict with returncode / stdout / stderr / duration_ms / timed_out.

    JSON report is emitted to stdout (no --output-path) so the host can
    parse category scores directly from captured subprocess output.
    """
    cmd = [
        "docker", "run", "--rm",
        "--network", "ai_company_network",
        "--cap-add", "SYS_ADMIN",                # Chrome sandbox inside Docker
        "--shm-size", LIGHTHOUSE_SHM_SIZE,       # /dev/shm is 64MB by default
        LIGHTHOUSE_IMAGE,
        url,
        "--output=json",                         # JSON to stdout
        f"--only-categories={LIGHTHOUSE_CATEGORIES}",
        "--quiet",
        "--max-wait-for-load=90000",
    ]

    logger.info("lighthouse_start", url=url)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=LIGHTHOUSE_TIMEOUT_S,
        )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "lighthouse_end",
            url=url, returncode=proc.returncode, duration_ms=dur_ms,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "duration_ms": dur_ms,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("lighthouse_timeout", url=url,
                       timeout_s=LIGHTHOUSE_TIMEOUT_S)
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"lighthouse_timeout_after_{LIGHTHOUSE_TIMEOUT_S}s",
            "duration_ms": dur_ms,
            "timed_out": True,
        }
    except FileNotFoundError:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "docker_cli_not_found",
            "duration_ms": 0,
            "timed_out": False,
        }


def _extract_categories(stdout: str) -> dict:
    """
    Locate the JSON report on stdout and return {category: 0..100}.

    Lighthouse with --output=json prints the full report as a single
    multi-line JSON object. We scan from the bottom for the last line
    that opens a JSON object, then grow upward until it parses.
    """
    if not stdout:
        return {}

    # Fast path: whole stdout parses as JSON
    stripped = stdout.strip()
    try:
        data = json.loads(stripped)
        return _scores_from_report(data)
    except Exception:
        pass

    # Fallback: try each '{' line from the bottom
    lines = stdout.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].lstrip().startswith("{"):
            continue
        candidate = "\n".join(lines[i:])
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        scores = _scores_from_report(data)
        if scores:
            return scores
    return {}


def _scores_from_report(data: dict) -> dict:
    cats = (data or {}).get("categories") or {}
    scores: dict = {}
    for name, cat in cats.items():
        if not isinstance(cat, dict):
            continue
        score = cat.get("score")
        if score is None:
            continue
        try:
            scores[name] = int(round(float(score) * 100))
        except Exception:
            continue
    return scores


def _summarize_lighthouse_issues(stdout: str, max_items: int = 5) -> list:
    """
    Very light issue extraction. We do not fail on this; only report.
    """
    issues: list = []
    if not stdout:
        return issues
    try:
        data = json.loads(stdout.strip())
    except Exception:
        return issues
    audits = data.get("audits") or {}
    for aid, audit in audits.items():
        if not isinstance(audit, dict):
            continue
        score = audit.get("score")
        if score is None:
            continue
        try:
            s_num = float(score)
        except Exception:
            continue
        if s_num < 0.9:
            issues.append({
                "audit": aid,
                "score": s_num,
                "title": audit.get("title", ""),
            })
    issues.sort(key=lambda i: i["score"])
    return issues[:max_items]


# --- Main entry point ---------------------------------------------------
async def run_improver_agent(
    files: list[dict],
    job_id: str,
    user_id: str,
    trace_id: str,
    output_root: Optional[Path] = None,
    run_lighthouse: bool = True,
) -> ImproverResult:
    """
    Build + run the app, then Lighthouse-scores it on the shared network.
    Always cleans up the app container and image in `finally`.
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)
    log.info("agent_started", agent="improver", files=len(files or []))

    result = ImproverResult()
    t0 = time.perf_counter()

    if not files:
        result.error = "no_files"
        return result

    root = Path(output_root) if output_root else Path(
        __file__).resolve().parent.parent / "output"
    root.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Build image
        build = build_app_image(files, job_id, root)
        if not build.ok:
            result.error = "app_build_failed"
            result.lighthouse_log = build.log[-1500:]
            log.warning("improver_build_failed", rc=build.returncode)
            return result

        # 2. Run container (no host port needed; on-network service name)
        run = run_app_container(job_id, host_port=None)
        result.app_container = run.container_name
        result.app_service_url = run.service_url
        if not run.ok:
            result.error = "app_run_failed"
            result.lighthouse_log = run.log[-1500:]
            log.warning("improver_run_failed", rc=run.returncode)
            return result

        # 3. Wait for uvicorn to be ready (non-blocking)
        if not wait_until_ready(run.container_name, timeout_s=120):
            log.warning("improver_app_not_ready_skipping_lighthouse")
            result.ok = True
            result.error = ""  # not fatal
            result.before_scores = {}
            result.after_scores = {}
            result.issues = ["app_not_ready"]
            return result

        if not run_lighthouse:
            result.ok = True
            log.info("improver_skipped_lighthouse")
            return result

        # 4. Run Lighthouse (non-blocking: warning on fail, build continues)
        lh = _run_lighthouse(run.service_url)
        result.lighthouse_returncode = lh["returncode"]
        result.lighthouse_log = (lh["stdout"] + "\n" + lh["stderr"])[-3000:]

        if lh["returncode"] != 0:
            # Non-fatal: log warning, mark ok=True, no error
            log.warning("lighthouse_skipped", reason="lighthouse_failed",
                        rc=lh["returncode"])
            result.ok = True
            result.before_scores = {}
            result.after_scores = {}
            result.issues = ["lighthouse_unavailable"]
            return result

        # 5. Parse scores
        scores = _extract_categories(lh["stdout"])
        result.before_scores = scores
        result.after_scores = dict(scores)  # placeholder; real improvement is Step 16

        # 6. Extract issues (informational)
        result.issues = _summarize_lighthouse_issues(lh["stdout"])

        result.ok = bool(scores)
        if not scores:
            result.error = "lighthouse_no_scores"

        log.info("agent_completed", agent="improver",
                 scores=scores, issues=len(result.issues),
                 ok=result.ok)
        return result

    except Exception as exc:
        log.error("improver_unexpected", err=str(exc))
        result.error = f"unexpected:{type(exc).__name__}"
        return result
    finally:
        try:
            cleanup_app(job_id, root)
        except Exception:
            pass
        result.duration_ms = int((time.perf_counter() - t0) * 1000)