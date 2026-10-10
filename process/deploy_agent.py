"""
AI Software Company â€” Deploy Agent (Step 14)

Architecture references:
  Section 5   Deploy: approval -> GitPython push -> local Docker serve
  Section 21  Versioning + Rollback
  Section 22  Build Artifact + Port Manager
  Section 23  Human approval gate (AI cannot self-approve)
  Section 26  Shared network ai_company_network
  Section 37  Generated app port binding (127.0.0.1 only)
  F5          Generated apps ke ports 8001+
  F9          Generated app = Docker container
  F25/29/45   Cleanup behaviour comes later (Step 17); this module only deploys.

Flow:
  1. require approval=True (or raise NotApprovedError)
  2. allocate_port(job_id) via process.port_manager  (SQLite + asyncio.Lock)
  3. build_app_image(files, job_id, output_root)
  4. run_app_container(job_id, host_port=<allocated>)  -> 127.0.0.1 only
  5. save_version(project_id, job_id, artifact_path, status="working")
  6. OPTIONAL GitHub push via GitPython (only when GITHUB_TOKEN + GITHUB_USERNAME
     are set in .env). Push is skipped silently if credentials are missing.
  7. Return DeployResult

Notes:
  - Deploy NEVER self-approves. `approval` must be passed explicitly.
  - GitHub push writes to a per-job local working copy under
    output/<job>/.git_push_workspace so the platform repo stays clean.
  - Rollback is available through process.versioning.rollback(); this
    module does not call it automatically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from observability.logger import bind_job, get_logger
from process import port_manager
from process.versioning import save_version
from sandbox.app_runner import (
    build_app_image,
    cleanup_app,
    run_app_container,
    wait_until_ready,
)

load_dotenv()

logger = get_logger("process.deploy")


class DeployError(RuntimeError):
    """Base error for deploy failures."""


class NotApprovedError(DeployError):
    """Raised when deploy is called without explicit human approval."""


@dataclass
class GitHubPushResult:
    attempted: bool = False
    ok: bool = False
    repo_url: str = ""
    commit_hash: str = ""
    log: str = ""


@dataclass
class DeployResult:
    ok: bool = False
    error: str = ""
    job_id: str = ""
    project_id: str = ""
    port: Optional[int] = None
    host_url: Optional[str] = None
    service_url: Optional[str] = None
    container_name: str = ""
    version: int = 0
    artifact_path: str = ""
    duration_ms: int = 0
    github: GitHubPushResult = field(default_factory=GitHubPushResult)


# --- GitHub push (GitPython) ---------------------------------------------
def _github_creds() -> tuple[str, str]:
    """Return (username, token) from env, or ('', '')."""
    user = (os.getenv("GITHUB_USERNAME") or "").strip()
    token = (os.getenv("GITHUB_TOKEN") or "").strip()
    return user, token


def _prepare_push_workspace(files: list[dict], job_id: str, output_root: Path) -> Path:
    """
    Write generated files into a fresh workspace that will become its own
    git repo. Returns the workspace path.
    """
    from security import path_check  # local import to avoid cycle

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)[:40]
    ws = output_root / safe / ".git_push_workspace"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True, exist_ok=True)

    for f in files:
        rel = f.get("path", "")
        content = f.get("content", "")
        if not rel or content is None:
            continue
        target = path_check.safe_join(ws, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    # Ensure a minimal .gitignore in the pushed repo
    gi = ws / ".gitignore"
    if not gi.exists():
        gi.write_text(
            "__pycache__/\n*.pyc\n*.db\n.env\nvenv/\n",
            encoding="utf-8",
        )

    return ws


def _run_git(ws: Path, *args: str, timeout: int = 30) -> tuple[int, str]:
    """Run a git command in `ws`. Returns (returncode, combined output)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(ws),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, f"git timeout: {' '.join(args)}"
    except FileNotFoundError:
        return -1, "git_cli_not_found"


def _push_to_github(
    files: list[dict],
    job_id: str,
    project_id: str,
    version: int,
    output_root: Path,
) -> GitHubPushResult:
    """
    Push generated app to GitHub as its own repo. Skips silently when
    credentials are missing. Uses git CLI via subprocess for full control
    (GitPython is imported/used for repo ops in later steps; here we keep
    the flow explicit and inspectable).
    """
    result = GitHubPushResult()

    user, token = _github_creds()
    if not user or not token:
        logger.info("github_push_skipped", reason="missing_credentials")
        return result  # attempted stays False

    result.attempted = True
    repo_name = f"ai-company-{project_id}-{version}".replace("_", "-")[:60]
    repo_url = f"https://github.com/{user}/{repo_name}.git"
    auth_url = f"https://{user}:{token}@github.com/{user}/{repo_name}.git"

    try:
        ws = _prepare_push_workspace(files, job_id, output_root)

        # init repo
        rc, out = _run_git(ws, "init", "-b", "main")
        if rc != 0:
            result.log = f"git_init_failed: {out[-500:]}"
            return result

        # config local identity (avoid relying on global)
        _run_git(ws, "config", "user.name", user)
        _run_git(ws, "config", "user.email",
                 f"{user}@users.noreply.github.com")

        rc, out = _run_git(ws, "add", "-A")
        if rc != 0:
            result.log = f"git_add_failed: {out[-500:]}"
            return result

        rc, out = _run_git(ws, "commit", "-m",
                           f"Deploy {project_id} v{version} (job {job_id})")
        # Some git versions return 1 for "nothing to commit" â€” accept 0 only
        if rc != 0 and "nothing to commit" not in out.lower():
            result.log = f"git_commit_failed: {out[-500:]}"
            return result

        rc, out = _run_git(ws, "remote", "add", "origin", auth_url)
        # If origin already exists (re-run), update it
        if rc != 0 and "already exists" in out.lower():
            _run_git(ws, "remote", "set-url", "origin", auth_url)

        rc, out = _run_git(ws, "push", "-u", "origin", "main", timeout=60)
        if rc != 0:
            # Do not leak token in the log
            safe_out = out.replace(token, "[REDACTED]") if token else out
            result.log = f"git_push_failed: {safe_out[-500:]}"
            return result

        # capture commit hash
        rc, out = _run_git(ws, "rev-parse", "HEAD")
        result.commit_hash = out.strip().splitlines()[0] if rc == 0 else ""
        result.repo_url = f"https://github.com/{user}/{repo_name}"
        result.ok = True
        result.log = "push_ok"
        logger.info("github_push_ok", repo=result.repo_url,
                    commit=result.commit_hash[:8])
    except Exception as exc:
        # Never include token in error message
        result.log = f"push_exception: {type(exc).__name__}"
        logger.warning("github_push_exception", err=type(exc).__name__)
    finally:
        # Remove the temporary push workspace to save disk
        try:
            ws = output_root / "".join(
                c if c.isalnum() or c in "-_" else "_" for c in job_id
            )[:40] / ".git_push_workspace"
            if ws.exists():
                shutil.rmtree(ws, ignore_errors=True)
        except Exception:
            pass
    return result


# --- Main entry point ---------------------------------------------------
async def deploy(
    files: list[dict],
    job_id: str,
    project_id: str,
    user_id: str,
    trace_id: str,
    approval: bool,
    output_root: Optional[Path] = None,
) -> DeployResult:
    """
    Deploy the generated app.

    - approval MUST be True (Section 23: AI cannot self-approve).
    - Binds to 127.0.0.1 only (Section 37).
    - Records a version row (Section 21).
    - Optionally pushes to GitHub if credentials are present in .env.

    Raises NotApprovedError if approval is False or missing.
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)

    if not approval:
        log.warning("deploy_refused_no_approval")
        raise NotApprovedError(
            "Human approval required before deploy (Section 23)."
        )

    if not files:
        return DeployResult(ok=False, error="no_files",
                            job_id=job_id, project_id=project_id)

    root = Path(output_root) if output_root else Path(
        __file__).resolve().parent.parent / "output"
    root.mkdir(parents=True, exist_ok=True)

    result = DeployResult(job_id=job_id, project_id=project_id)
    t0 = time.perf_counter()

    log.info("deploy_started", files=len(files), project_id=project_id)

    try:
        # 1. Allocate a port (SQLite + asyncio.Lock, F16/F24/F27)
        try:
            port = await port_manager.allocate_port(job_id)
            result.port = port
            result.host_url = f"http://127.0.0.1:{port}"
        except port_manager.NoFreePortError as exc:
            result.error = "no_free_port"
            log.warning("deploy_no_free_port", err=str(exc))
            return result
        except Exception as exc:
            result.error = f"port_alloc_failed:{type(exc).__name__}"
            log.error("deploy_port_alloc_failed", err=str(exc))
            return result

        # 2. Build the generated app image
        build = build_app_image(files, job_id, root)
        if not build.ok:
            result.error = "app_build_failed"
            log.warning("deploy_build_failed", rc=build.returncode)
            # release the port; we will not run
            port_manager.release_port(port, job_id)
            result.port = None
            result.host_url = None
            return result

        # 3. Run container with the reserved port (127.0.0.1 only)
        run = run_app_container(job_id, host_port=port)
        result.container_name = run.container_name
        result.service_url = run.service_url
        if not run.ok:
            result.error = "app_run_failed"
            log.warning("deploy_run_failed", rc=run.returncode)
            port_manager.release_port(port, job_id)
            result.port = None
            result.host_url = None
            return result

        # 4. Wait for readiness
        if not wait_until_ready(run.container_name, timeout_s=60):
            result.error = "app_not_ready"
            log.warning("deploy_app_not_ready")
            cleanup_app(job_id, root)
            port_manager.release_port(port, job_id)
            result.port = None
            result.host_url = None
            return result

        # 5. Save version (Section 21)
        try:
            vrow = await save_version(
                project_id=project_id,
                job_id=job_id,
                artifact_path=str(root / job_id),
                status="working",
            )
            result.version = int(vrow["version"])
            result.artifact_path = str(vrow.get("artifact_path") or "")
        except Exception as exc:
            log.error("deploy_version_save_failed", err=str(exc))
            # Non-fatal: the app is running. We continue and record version 0.

        # 6. Optional GitHub push
        result.github = _push_to_github(
            files=files,
            job_id=job_id,
            project_id=project_id,
            version=result.version,
            output_root=root,
        )

        result.ok = True
        log.info(
            "deploy_done",
            port=result.port,
            host_url=result.host_url,
            version=result.version,
            github_ok=result.github.ok,
            github_attempted=result.github.attempted,
        )
        return result

    except Exception as exc:
        log.error("deploy_unexpected", err=str(exc))
        result.error = f"unexpected:{type(exc).__name__}"
        return result
    finally:
        result.duration_ms = int((time.perf_counter() - t0) * 1000)


# --- Teardown ------------------------------------------------------------
def stop_deployed(job_id: str, project_id: str, output_root: Optional[Path] = None) -> None:
    """
    Stop the running app container for a job and release its port.
    Does NOT delete the version row (Section 21 keeps history).
    """
    root = Path(output_root) if output_root else Path(
        __file__).resolve().parent.parent / "output"
    try:
        cleanup_app(job_id, root)
    except Exception:
        pass
    try:
        port_manager.release_ports_for_job(job_id)
    except Exception:
        pass
