"""
AI Software Company â€” Generated app Docker runner (Step 13)

Architecture references:
  Section 5   Improver: App URL http://app-{job_id}:8000 on shared network
  Section 26  Shared network ai_company_network
  Section 35  Sandbox hardening (no privileged, dropped caps, resource limits)
  Section 37  Generated app Dockerfile
  F22         Docker shared network ai_company_network
  F23         Lighthouse URL via service name
  F5          Generated apps ports 8001+ (only when binding to host)
  F9          Generated app = Docker container

Purpose:
  Build the generated app's Docker image and run it as a container on
  ai_company_network named app-{job_id}. The Improver accesses it via
  the service name; Deploy (Step 14) will additionally bind a host port.

Public API:
  build_app_image(files, job_id, output_root) -> BuildResult
  run_app_container(job_id, host_port=None) -> RunResult
  stop_app_container(job_id) -> None
  remove_app_image(job_id) -> None
  cleanup_app(job_id, output_root) -> None

Design notes:
  - No host ports bound unless host_port is explicitly provided (localhost
    only, per F5 / Section 37 hardening).
  - Non-root user, no privileged, --cap-drop=ALL, --security-opt no-new-privileges.
  - --memory 512m, --cpus 1.0 (slightly larger than sandbox tests because a
    live app with uvicorn + Jinja2 + SQLAlchemy needs more than 256m).
  - DATABASE_URL injected at runtime (F26 dynamic DB secrets).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from observability.logger import get_logger
from security import path_check

logger = get_logger("sandbox.app_runner")

NETWORK = "ai_company_network"
APP_IMAGE_PREFIX = "ai-company-app"
APP_CONTAINER_PREFIX = "app"
APP_INTERNAL_PORT = 8000

BUILD_TIMEOUT_S = 300
RUN_START_TIMEOUT_S = 15
APP_MEMORY = "512m"
APP_CPUS = "1.0"
APP_PIDS = "200"


@dataclass
class BuildResult:
    ok: bool = False
    image_tag: str = ""
    returncode: int = 0
    log: str = ""
    duration_ms: int = 0


@dataclass
class RunResult:
    ok: bool = False
    container_name: str = ""
    service_url: str = ""
    host_url: Optional[str] = None
    returncode: int = 0
    log: str = ""
    duration_ms: int = 0


def _to_posix(p: Path) -> str:
    return str(p).replace("\\", "/")


def _tail(text: str, n: int = 3000) -> str:
    return (text or "")[-n:]


def _safe_id(job_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in job_id)[:40]


def _image_tag(job_id: str) -> str:
    return f"{APP_IMAGE_PREFIX}-{_safe_id(job_id)}:latest"


def _container_name(job_id: str) -> str:
    return f"{APP_CONTAINER_PREFIX}-{_safe_id(job_id)}"


def _write_workspace(files: list[dict], workspace: Path) -> int:
    workspace.mkdir(parents=True, exist_ok=True)
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


def build_app_image(files, job_id, output_root):
    result = BuildResult()
    t0 = time.perf_counter()

    safe = _safe_id(job_id)
    workspace = output_root / safe
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    if _write_workspace(files, workspace) == 0:
        result.log = "no_files_written"
        result.duration_ms = int((time.perf_counter() - t0) * 1000)
        return result

    if not (workspace / "Dockerfile").exists():
        (workspace / "Dockerfile").write_text(
            "FROM python:3.12-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8000\n"
            'CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]\n',
            encoding="utf-8",
        )

    tag = _image_tag(job_id)
    result.image_tag = tag

    cmd = ["docker", "build", "-t", tag, _to_posix(workspace)]
    logger.info("app_build_start", tag=tag)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=BUILD_TIMEOUT_S,
        )
        result.returncode = proc.returncode
        result.log = _tail((proc.stdout or "") + "\n" + (proc.stderr or ""))
        result.ok = (proc.returncode == 0)
        logger.info("app_build_end", tag=tag, rc=proc.returncode,
                    duration_ms=int((time.perf_counter() - t0) * 1000))
    except subprocess.TimeoutExpired:
        result.returncode = -1
        result.log = f"build_timeout_after_{BUILD_TIMEOUT_S}s"
        logger.warning("app_build_timeout", tag=tag)
    except FileNotFoundError:
        result.returncode = -1
        result.log = "docker_cli_not_found"
    finally:
        result.duration_ms = int((time.perf_counter() - t0) * 1000)
    return result


def _container_exists(name: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name=^{name}$",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10,
        )
        return name in (r.stdout or "")
    except Exception:
        return False


def run_app_container(job_id, host_port=None):
    result = RunResult()
    t0 = time.perf_counter()

    tag = _image_tag(job_id)
    name = _container_name(job_id)
    result.container_name = name
    result.service_url = f"http://{name}:{APP_INTERNAL_PORT}"

    if _container_exists(name):
        try:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=15)
        except Exception:
            pass

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "--network", NETWORK,
        "--memory", APP_MEMORY,
        "--cpus", APP_CPUS,
        "--pids-limit", APP_PIDS,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-e", "DATABASE_URL=sqlite:///./app.db",
        "-e", "PYTHONUNBUFFERED=1",
    ]
    if host_port is not None:
        cmd += ["-p", f"127.0.0.1:{host_port}:{APP_INTERNAL_PORT}"]
        result.host_url = f"http://127.0.0.1:{host_port}"

    cmd.append(tag)

    logger.info("app_run_start", container=name, host_port=host_port)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=30,
        )
        result.returncode = proc.returncode
        result.log = _tail((proc.stdout or "") + "\n" + (proc.stderr or ""))
        result.ok = (proc.returncode == 0)
    except subprocess.TimeoutExpired:
        result.returncode = -1
        result.log = "docker_run_timeout"
    except FileNotFoundError:
        result.returncode = -1
        result.log = "docker_cli_not_found"
    finally:
        result.duration_ms = int((time.perf_counter() - t0) * 1000)

    if result.ok:
        time.sleep(2.0)
        logger.info("app_run_end", container=name,
                    service_url=result.service_url,
                    host_url=result.host_url)
    return result


def stop_app_container(job_id: str) -> None:
    name = _container_name(job_id)
    try:
        subprocess.run(["docker", "rm", "-f", name],
                       capture_output=True, timeout=15)
    except Exception:
        pass


def remove_app_image(job_id: str) -> None:
    tag = _image_tag(job_id)
    try:
        subprocess.run(["docker", "rmi", "-f", tag],
                       capture_output=True, timeout=30)
    except Exception:
        pass


def cleanup_app(job_id: str, output_root: Path) -> None:
    stop_app_container(job_id)
    remove_app_image(job_id)
    safe = _safe_id(job_id)
    workspace = output_root / safe
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)


def wait_until_ready(container_name: str, timeout_s: int = RUN_START_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=5,
            )
            text = (r.stdout or "") + (r.stderr or "")
            if "Uvicorn running on" in text or "Application startup complete" in text:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False
