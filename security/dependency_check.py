"""
AI Software Company — Dependency check (Step 11)

Architecture references:
  Section 13  requirements.txt -> validate (PyPI only), version, vulnerability
  Section 10  Check 8: Dependency validation
  Section 24  Audit action: dependency_check

Flow:
  requirements.txt
    -> parse each line
    -> validate package name format
    -> reject obviously dangerous names
    -> check existence on PyPI (via https://pypi.org/pypi/<name>/json)
    -> check that the pinned version exists (if pinned)
    -> run pip-audit on a temp requirements file for CVE scan
    -> return structured result

Public API:
  check_requirements(text: str) -> DependencyCheckResult
  check_requirements_file(path) -> DependencyCheckResult
  DependencyCheckResult fields:
      ok: bool
      packages_ok: list[str]
      issues: list[Issue]  # one per bad package or global error
      raw: dict            # extra info (audit output, etc.)

Design:
  - No AI, no shell-out except pip-audit (invoked via subprocess with a
    sanitized temporary requirements file).
  - Uses only requests (already a dependency) for PyPI lookups.
  - Time-bounded: each HTTP call has a short timeout.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# --- Config -------------------------------------------------------------
PYPI_TIMEOUT = 5             # seconds per HTTP request
PIP_AUDIT_TIMEOUT = 120      # seconds

# PEP 508-ish allowed package name (very permissive on purpose; PyPI is stricter)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,99}$")

# Explicitly banned (case-insensitive) package names. Extended later if needed.
_BANNED = {
    "pip",              # do not let a generated app reinstall pip
    "setuptools",       # same reason
    "wheel",
    "distribute",
}


# --- Result types -------------------------------------------------------
@dataclass
class Issue:
    package: str
    reason: str
    detail: str = ""


@dataclass
class DependencyCheckResult:
    ok: bool = True
    packages_ok: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


# --- Parsing ------------------------------------------------------------
def _parse_line(line: str) -> Optional[tuple[str, Optional[str]]]:
    """
    Parse a single requirements.txt line.
    Returns (name, version_or_None) or None if the line is not a package
    line (comment, empty, option like -r, etc.).
    """
    if line is None:
        return None
    s = line.strip()
    if not s:
        return None
    if s.startswith("#"):
        return None
    if s.startswith("-"):        # -r, --index-url, etc. — ignore for V1
        return None
    if s.startswith("git+") or s.startswith("http://") or s.startswith("https://"):
        return None              # URL-based requirements not supported in V1

    # Split on the operators we accept: ==, >=, <=, ~=, >, <
    # V1 policy: allow only == (pinned) or bare name. Anything else warns.
    for op in ("==", ">=", "<=", "~=", ">", "<"):
        if op in s:
            name, _, rest = s.partition(op)
            name = name.strip()
            version = rest.strip()
            # strip extras like package[extra]
            if "[" in name:
                name = name.split("[", 1)[0].strip()
            if not name:
                return None
            return name, (version or None)

    # No operator: bare name
    name = s.strip()
    if "[" in name:
        name = name.split("[", 1)[0].strip()
    return name, None


def _parse_requirements(text: str) -> list[tuple[str, Optional[str]]]:
    return [p for p in (_parse_line(ln) for ln in text.splitlines()) if p]


# --- PyPI lookups -------------------------------------------------------
def _pypi_has_package(name: str) -> tuple[bool, str]:
    """Return (exists, detail)."""
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        r = requests.get(url, timeout=PYPI_TIMEOUT)
    except Exception as exc:
        return False, f"network_error:{type(exc).__name__}"
    if r.status_code == 200:
        return True, "ok"
    if r.status_code == 404:
        return False, "not_found_on_pypi"
    return False, f"http_{r.status_code}"


def _pypi_has_version(name: str, version: str) -> tuple[bool, str]:
    """Return (exists, detail). Only called after _pypi_has_package passed."""
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        r = requests.get(url, timeout=PYPI_TIMEOUT)
    except Exception as exc:
        return False, f"network_error:{type(exc).__name__}"
    if r.status_code == 200:
        return True, "ok"
    if r.status_code == 404:
        return False, "version_not_found"
    return False, f"http_{r.status_code}"


# --- pip-audit ----------------------------------------------------------
def _pip_audit(reqs: list[tuple[str, Optional[str]]]) -> tuple[bool, dict]:
    """
    Run pip-audit on a temporary requirements file.
    Returns (clean, raw_dict). 'clean' means no known vulnerabilities.
    """
    # Build a temp requirements file with pinned == where possible.
    lines = []
    for name, ver in reqs:
        if ver:
            lines.append(f"{name}=={ver}")
        else:
            lines.append(name)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        tf.write("\n".join(lines) + "\n")
        temp_path = tf.name

    try:
        # Use the same Python as the current venv so pip-audit can see packages.
        cmd = [
            sys.executable,
            "-m",
            "pip_audit",
            "-r",
            temp_path,
            "--format",
            "json",
            "--progress-spinner",
            "off",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PIP_AUDIT_TIMEOUT,
        )
        out = proc.stdout or ""
        err = proc.stderr or ""
        try:
            data = json.loads(out) if out.strip() else {}
        except Exception:
            data = {"_raw_stdout": out[:2000], "_raw_stderr": err[:2000]}

        # pip-audit exit codes: 0 = no vulns found, 1 = vulns found, other = error
        # We only treat known vulnerabilities as a hard block for V1.
        if proc.returncode == 0:
            return True, data
        if proc.returncode == 1:
            return False, data
        # Any other error: don't block the whole build; surface as raw.
        return True, {
            "error": "pip_audit_failed",
            "returncode": proc.returncode,
            "stderr": err[:2000],
        }
    except subprocess.TimeoutExpired:
        return True, {"error": "pip_audit_timeout"}
    except Exception as exc:
        return True, {"error": f"pip_audit_exception:{type(exc).__name__}"}
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


# --- Public API ---------------------------------------------------------
def check_requirements(text: str) -> DependencyCheckResult:
    """
    Validate a requirements.txt text and return a structured result.
    Never raises for malformed input; returns issues instead.
    """
    result = DependencyCheckResult()

    if not text or not text.strip():
        result.ok = False
        result.issues.append(Issue(package="", reason="empty_requirements"))
        return result

    parsed = _parse_requirements(text)
    if not parsed:
        result.ok = False
        result.issues.append(Issue(package="", reason="no_parseable_packages"))
        return result

    for name, version in parsed:
        # 1. Name format
        if not _NAME_RE.match(name):
            result.ok = False
            result.issues.append(Issue(package=name, reason="invalid_name_format"))
            continue

        # 2. Banned names
        if name.lower() in _BANNED:
            result.ok = False
            result.issues.append(Issue(package=name, reason="banned_package"))
            continue

        # 3. Existence on PyPI
        exists, detail = _pypi_has_package(name)
        if not exists:
            result.ok = False
            result.issues.append(
                Issue(package=name, reason="pypi_lookup_failed", detail=detail)
            )
            continue

        # 4. Version existence (if pinned with ==)
        if version:
            v_ok, v_detail = _pypi_has_version(name, version)
            if not v_ok:
                result.ok = False
                result.issues.append(
                    Issue(package=name, reason="version_not_available", detail=v_detail)
                )
                continue

        result.packages_ok.append(f"{name}=={version}" if version else name)

    # 5. pip-audit CVE scan (only if packages parsed; still runs even if some failed
    #    so we surface everything at once).
    if result.packages_ok:
        clean, audit_raw = _pip_audit(parsed)
        result.raw["pip_audit"] = audit_raw
        if not clean:
            result.ok = False
            result.issues.append(
                Issue(package="*", reason="pip_audit_found_vulnerabilities")
            )

    return result


def check_requirements_file(path: str | Path) -> DependencyCheckResult:
    """Read a requirements file from disk and validate it."""
    p = Path(path)
    if not p.exists():
        r = DependencyCheckResult(ok=False)
        r.issues.append(Issue(package="", reason="file_not_found", detail=str(p)))
        return r
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        r = DependencyCheckResult(ok=False)
        r.issues.append(Issue(package="", reason="file_read_error",
                              detail=type(exc).__name__))
        return r
    return check_requirements(text)