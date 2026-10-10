"""
AI Software Company — File path security (Step 5)

Architecture references:
  Section 11  File path security (block traversal, absolute paths, .env)
  Section 10  Code validation: file path validation
  F28         Secret leak prevention (no access to .env or parent files)

Purpose:
  - Given an "allowed root" directory (e.g. output/<job_id>/), decide
    whether a proposed relative file path may be written/read inside it.
  - Reject:
      * path traversal:  ../, ..\\, any ".." component
      * absolute host paths:  C:\\..., /etc/..., \\\\server\\share
      * Windows drive-relative:  C:foo
      * null bytes, empty path
      * reserved system directories (.git at root is also blocked unless
        explicitly allowed — generated apps never need it)
      * direct access to .env or anything named *.env inside the root
  - Never touches the filesystem in a way that follows symlinks outside
    the root: we resolve the final path and re-check that it is still
    inside the root.

Design:
  - Pure Python, no AI.
  - Two entry points:
      validate_relative(rel_path)          -> Path  (raises on block)
      is_safe_join(root, rel_path) -> bool (no raise)
  - Returns POSIX-style relative Path on success so it is portable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


# --- Constants -----------------------------------------------------------
# Windows drive letter (absolute) e.g. C:\  or  C:foo
_DRIVE_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")
_DRIVE_REL_RE = re.compile(r"^[A-Za-z]:(?![\\/])")

# UNC paths: \\server\share
_UNC_RE = re.compile(r"^\\\\[^\\]+\\[^\\]+")

# Names that are always forbidden inside the allowed root
_FORBIDDEN_BASENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.test",
}

# Any file ending in .env is also forbidden (e.g. foo.env)
_FORBIDDEN_SUFFIXES = (".env",)

# Directory names that must never appear as a component inside generated apps
_FORBIDDEN_COMPONENTS = {
    ".git",
    ".ssh",
    ".aws",
    "__pycache__",  # allowed to exist, but we never accept it as a target
}


class PathSecurityError(ValueError):
    """Raised when a proposed path violates Section 11 rules."""

    def __init__(self, reason: str, path: str):
        super().__init__(f"blocked path ({reason}): {path!r}")
        self.reason = reason
        self.path = path


# --- Core check ----------------------------------------------------------
def _reject_if_bad_relative(rel_path: str) -> None:
    """Raise PathSecurityError if rel_path fails syntactic checks."""
    if not isinstance(rel_path, str) or rel_path == "":
        raise PathSecurityError("empty", rel_path or "")

    if "\x00" in rel_path:
        raise PathSecurityError("null byte", rel_path)

    # Absolute Unix
    if rel_path.startswith("/"):
        raise PathSecurityError("absolute unix path", rel_path)

    # Absolute Windows (C:\ ...)
    if _DRIVE_ABS_RE.match(rel_path):
        raise PathSecurityError("absolute windows path", rel_path)

    # Windows drive-relative (C:foo)
    if _DRIVE_REL_RE.match(rel_path):
        raise PathSecurityError("windows drive-relative path", rel_path)

    # UNC
    if _UNC_RE.match(rel_path):
        raise PathSecurityError("unc path", rel_path)

    # Normalize separators and inspect components
    normalized = rel_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p not in ("", ".")]

    if not parts:
        raise PathSecurityError("empty after normalization", rel_path)

    for part in parts:
        if part == "..":
            raise PathSecurityError("parent traversal", rel_path)
        if part in _FORBIDDEN_COMPONENTS:
            raise PathSecurityError(f"forbidden component {part!r}", rel_path)

    # Filename rules
    last = parts[-1]
    if last in _FORBIDDEN_BASENAMES:
        raise PathSecurityError("forbidden filename", rel_path)
    if last.endswith(_FORBIDDEN_SUFFIXES):
        raise PathSecurityError("forbidden filename suffix", rel_path)


def validate_relative(rel_path: str) -> Path:
    """
    Validate a proposed *relative* path (POSIX or Windows separators).
    Returns a normalized relative pathlib.Path on success.
    Raises PathSecurityError otherwise.
    """
    _reject_if_bad_relative(rel_path)
    normalized = rel_path.replace("\\", "/")
    return Path(*[p for p in normalized.split("/") if p not in ("", ".")])


def is_safe_join(root: os.PathLike | str, rel_path: str) -> bool:
    """
    Return True if `rel_path` may be joined under `root`.
    Never raises. Does NOT touch the filesystem, so it is safe to call
    before any write happens.
    """
    try:
        rel = validate_relative(rel_path)
    except PathSecurityError:
        return False

    try:
        root_path = Path(root).resolve(strict=False)
    except Exception:
        return False

    try:
        target = (root_path / rel).resolve(strict=False)
    except Exception:
        return False

    # target must be inside root (or equal to root for directory ops).
    try:
        target.relative_to(root_path)
    except ValueError:
        return False

    return True


def safe_join(root: os.PathLike | str, rel_path: str) -> Path:
    """
    Validate and return an absolute Path under `root`.
    Raises PathSecurityError on any violation.

    NOTE: This does NOT create directories or files. Caller is responsible
    for any subsequent mkdir / write, and should ensure the process has
    permission to write only inside `root`.
    """
    rel = validate_relative(rel_path)
    root_path = Path(root).resolve(strict=False)
    target = (root_path / rel).resolve(strict=False)
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise PathSecurityError("escapes allowed root", rel_path) from exc
    return target


# --- Convenience for callers --------------------------------------------
def describe_violation(rel_path: str) -> Optional[str]:
    """
    Return a short reason string if rel_path is blocked, else None.
    Useful for audit logs without raising.
    """
    try:
        _reject_if_bad_relative(rel_path)
        return None
    except PathSecurityError as exc:
        return exc.reason