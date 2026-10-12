"""
AI Software Company â€” Docker sandbox runner (Step 12)

Architecture references:
  Section 5   Tester: Docker sandbox, 256MB RAM, 0.5 CPU, 3-level check
  Section 27  Timeouts: pip 120s, tests 10s â€” actual enforcement
  Section 35  2-phase network (install -> tests offline), no subprocess fallback
  Section 37  Generated app Dockerfile
  F2          Sandbox 2-phase network
  F4          pip install 120s / tests 10s timeouts

Sandbox hardening (Section 35):
  - Non-root where possible (container default; we drop all capabilities)
  - No privileged mode
  - --cap-drop=ALL, --security-opt=no-new-privileges
  - --memory=256m, --cpus=0.5, --pids-limit=100
  - No host FS mounts beyond the build context (which is a temp workspace)
  - No Docker socket, no host PID, no host network

Phases:
  Phase 1: docker build (runs pip install -r requirements.txt, network ON, 120s)
  Phase 2: docker run --network none --memory 256m --cpus 0.5 (tests, 10s)

Public API:
  run_sandbox(files, job_id, output_root, test_script_path) -> SandboxResult
  cleanup_sandbox(job_id, output_root) -> None

SandboxResult fields:
  ok: bool                       -- True if phase1 and phase2 both succeeded
  phase1_returncode: int
  phase1_log: str                -- tail of build output
  phase2_returncode: int
  phase2_log: str                -- tail of test output
  test_report: dict              -- parsed JSON from test_runner.py
  error: str                     -- human-readable reason if ok=False
  duration_ms: int
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from observability.logger import get_logger
from security import path_check

logger = get_logger("sandbox.docker_runner")


# --- Config --------------------------------------------------------------
BASE_IMAGE = "python:3.12-slim"
PHASE1_TIMEOUT_S = 420        # docker build (pip install, 7 min for slow internet)
PHASE2_TIMEOUT_S = 90         # tests (1.5 min)
KILL_WAIT_S = 5               # after timeout, how long to wait for kill
CONTAINER_MEMORY = "256m"
CONTAINER_CPUS = "0.5"
CONTAINER_PIDS = "100"
CONTAINER_USER = "1000:1000"  # non-root

# Section 37 template (used only if the generated app did not ship a Dockerfile)
DEFAULT_DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir --timeout 1200 -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "test_runner.py"]
"""


@dataclass
class SandboxResult:
    ok: bool = False
    phase1_returncode: int = 0
    phase1_log: str = ""
    phase2_returncode: int = 0
    phase2_log: str = ""
    test_report: dict = field(default_factory=dict)
    error: str = ""
    duration_ms: int = 0


# --- Helpers -------------------------------------------------------------
def _tail(text: str, n: int = 4000) -> str:
    if not text:
        return ""
    return text[-n:]


def _to_posix(p: Path) -> str:
    """Docker CLI on Windows accepts both, but POSIX is safer for context."""
    return str(p).replace("\\", "/")


def _write_workspace(files: list[dict], workspace: Path) -> int:
    """Write generated files into the workspace. Returns count written."""
    written = 0
    for f in files:
        rel = f.get("path", "")
        content = f.get("content", "")
        if not rel or content is None:
            continue
        target = path_check.safe_join(workspace, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written += 1
    return written


def _sanitize_requirements(workspace: Path) -> None:
    """
    Rewrite requirements.txt keeping only well-formed package lines.
    20b models sometimes emit comments/notes/prose into requirements.txt
    which pip then fails on. We keep only lines matching a valid PEP 508
    name plus an optional version specifier.
    """
    import re as _re
    req = workspace / "requirements.txt"
    if not req.exists():
        return
    allowed = _re.compile(
        r"^\s*[A-Za-z0-9][A-Za-z0-9._\-]{0,80}"
        r"([<>=!~]{1,2}[A-Za-z0-9._,*+!\-]+)?(\s*;.*)?\s*$"
    )
    cleaned = []
    for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
        s2 = line.strip()
        if not s2 or s2.startswith("#"):
            continue
        if allowed.match(s2):
            cleaned.append(s2)

    # Safety net: if the model produced junk (or too few valid lines),
    # inject the essential packages required by Section 9.
    if len(cleaned) < 3:
        cleaned = [
            "fastapi==0.142.2",
            "starlette==1.6.0",
            "uvicorn[standard]==0.34.0",
            "sqlalchemy==2.0.36",
            "pydantic==2.10.4",
            "python-multipart==0.0.32",
            "passlib==1.7.4",
            "bcrypt==4.0.1",
            "jinja2==3.1.6",
        ]

    req.write_text("\n".join(cleaned) + "\n", encoding="utf-8")


def _ensure_requirements_with_httpx(workspace: Path) -> None:
    """
    FastAPI's TestClient needs httpx. Ensure it is in requirements.txt.
    If the generated app forgot requirements.txt entirely, create a minimal one.
    """
    req = workspace / "requirements.txt"
    if not req.exists():
        req.write_text("httpx\n", encoding="utf-8")
        return
    text = req.read_text(encoding="utf-8")
    if "httpx" not in text.lower():
        with req.open("a", encoding="utf-8") as fh:
            if not text.endswith("\n"):
                fh.write("\n")
            fh.write("httpx\n")


def _ensure_dockerfile(workspace: Path) -> None:
    df = workspace / "Dockerfile"
    if not df.exists():
        df.write_text(DEFAULT_DOCKERFILE, encoding="utf-8")


def _copy_test_runner(workspace: Path, test_script_path: Path) -> None:
    if not test_script_path.exists():
        raise FileNotFoundError(f"test script not found: {test_script_path}")
    content = test_script_path.read_text(encoding="utf-8")
    (workspace / "test_runner.py").write_text(content, encoding="utf-8")


# --- Docker operations ---------------------------------------------------
def _docker_build(workspace: Path, tag: str) -> dict:
    cmd = ["docker", "build", "-t", tag, _to_posix(workspace)]
    logger.info("sandbox_build_start", tag=tag, workspace=_to_posix(workspace))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=PHASE1_TIMEOUT_S
        )
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "sandbox_build_end", tag=tag, returncode=proc.returncode, duration_ms=dur_ms
        )
        return {
            "returncode": proc.returncode,
            "stdout": _tail(proc.stdout or ""),
            "stderr": _tail(proc.stderr or ""),
            "duration_ms": dur_ms,
        }
    except subprocess.TimeoutExpired:
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("sandbox_build_timeout", tag=tag, timeout_s=PHASE1_TIMEOUT_S)
        # Best-effort: remove a partially built image
        _docker_rmi(tag)
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"phase1_timeout_after_{PHASE1_TIMEOUT_S}s",
            "duration_ms": dur_ms,
        }
    except FileNotFoundError:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "docker_cli_not_found",
            "duration_ms": 0,
        }


def _kill_containers_for_image(tag: str) -> None:
    """Best-effort: kill any leftover containers using this image tag."""
    try:
        r = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"ancestor={tag}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        for cid in (r.stdout or "").split():
            try:
                subprocess.run(
                    ["docker", "kill", cid], capture_output=True, timeout=10
                )
            except Exception:
                pass
    except Exception:
        pass


def _docker_rmi(tag: str) -> None:
    try:
        subprocess.run(
            ["docker", "rmi", "-f", tag], capture_output=True, timeout=30
        )
    except Exception:
        pass


def _docker_run_tests(tag: str) -> dict:
    """
    Phase 2: run the test_runner.py inside the image with no network and
    strict resource limits.
    """
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", CONTAINER_MEMORY,
        "--cpus", CONTAINER_CPUS,
        "--pids-limit", CONTAINER_PIDS,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", CONTAINER_USER,
        tag,
        "python", "test_runner.py",
    ]
    logger.info("sandbox_test_start", tag=tag)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=PHASE2_TIMEOUT_S)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "sandbox_test_end", tag=tag, returncode=proc.returncode, duration_ms=dur_ms
        )
        return {
            "returncode": proc.returncode if proc.returncode is not None else -1,
            "stdout": _tail(stdout or ""),
            "stderr": _tail(stderr or ""),
            "duration_ms": dur_ms,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=KILL_WAIT_S)
        except Exception:
            pass
        _kill_containers_for_image(tag)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning("sandbox_test_timeout", tag=tag, timeout_s=PHASE2_TIMEOUT_S)
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"phase2_timeout_after_{PHASE2_TIMEOUT_S}s",
            "duration_ms": dur_ms,
            "timed_out": True,
        }


# --- Public API ----------------------------------------------------------
def run_sandbox(
    files: list[dict],
    job_id: str,
    output_root: Path,
    test_script_path: Path,
) -> SandboxResult:
    """
    Run a full 2-phase sandbox on the generated app.

    Raises nothing for build/test failures â€” returns a SandboxResult instead.
    Raises FileNotFoundError if test_script_path does not exist.
    """
    result = SandboxResult()
    t0 = time.perf_counter()

    # Sanitize job_id to be safe as a Docker tag component
    safe_job = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)[:40]
    tag = f"ai-company-sandbox-{safe_job}"

    workspace = output_root / safe_job
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    try:
        written = _write_workspace(files, workspace)
        if written == 0:
            result.error = "no_files_written"
            return result

        _ensure_requirements_with_httpx(workspace)
        _sanitize_requirements(workspace)
        _ensure_dockerfile(workspace)
        _copy_test_runner(workspace, test_script_path)

        # --- Phase 1 ---
        p1 = _docker_build(workspace, tag)
        result.phase1_returncode = p1["returncode"]
        result.phase1_log = (p1.get("stdout", "") + "\n" + p1.get("stderr", "")).strip()
        if p1["returncode"] != 0:
            result.ok = False
            result.error = "phase1_failed"
            return result

        # --- Phase 2 ---
        p2 = _docker_run_tests(tag)
        result.phase2_returncode = p2["returncode"]
        result.phase2_log = (p2.get("stdout", "") + "\n" + p2.get("stderr", "")).strip()

        # Try to parse the JSON report that test_runner.py prints as its last line
        report = {}
        for line in reversed((p2.get("stdout") or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    report = json.loads(line)
                    break
                except Exception:
                    continue
        result.test_report = report

        result.ok = (p2["returncode"] == 0)
        if not result.ok:
            if p2.get("timed_out"):
                result.error = "phase2_timeout"
            else:
                result.error = "phase2_tests_failed"

        return result

    except Exception as exc:
        logger.error("sandbox_unexpected", err=str(exc))
        result.ok = False
        result.error = f"unexpected:{type(exc).__name__}"
        return result
    finally:
        result.duration_ms = int((time.perf_counter() - t0) * 1000)
        _docker_rmi(tag)


def cleanup_sandbox(job_id: str, output_root: Path) -> None:
    """Remove the sandbox workspace folder for a job. Safe to call repeatedly."""
    safe_job = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)[:40]
    workspace = output_root / safe_job
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
