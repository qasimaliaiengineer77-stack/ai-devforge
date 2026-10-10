"""
AI Software Company — Structured logger (Step 6)

Architecture references:
  Section 2   structlog (JSON)
  Section 24  Audit log fields: timestamp, trace_id, job_id, user_id,
              action, status, message, duration_ms
  F28         Secret leak prevention (logs redacted)

Features:
  - JSON output (one object per line) — easy to parse later.
  - Shared redaction: any string value is passed through security.secret_scan.redact
    before being emitted, so gsk_/ghp_/sk-/postgres URLs/session cookies never
    reach log files.
  - configure_logging() should be called once at process startup (api/main.py).
  - get_logger(name) returns a bound structlog logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from security.secret_scan import redact


# --- Redaction processor -------------------------------------------------
def _redact_processor(logger, method_name, event_dict: dict) -> dict:
    """
    structlog processor: redact secrets from all string values.
    Keeps non-string values untouched (int, bool, None, lists, dicts).
    Recurses into lists and dicts.
    """
    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, list):
            return [_walk(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_walk(v) for v in value)
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        return value

    return {k: _walk(v) for k, v in event_dict.items()}


# --- Configuration -------------------------------------------------------
_configured = False


def configure_logging(level: str = "INFO") -> None:
    """
    Configure structlog + stdlib logging once.
    Safe to call multiple times (idempotent).
    """
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # stdlib root logger (so third-party libs also go through our config)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Silence noisy libs (keep WARNING+)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_processor,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(sort_keys=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str = "ai_company") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Auto-configures on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


# --- Convenience helpers -------------------------------------------------
def bind_job(logger, *, job_id: str | None = None,
             user_id: str | int | None = None,
             trace_id: str | None = None) -> Any:
    """
    Bind job context to a logger. Use once at the start of a job so every
    subsequent log line carries job_id/user_id/trace_id automatically.
    """
    ctx = {}
    if job_id is not None:
        ctx["job_id"] = job_id
    if user_id is not None:
        ctx["user_id"] = str(user_id)
    if trace_id is not None:
        ctx["trace_id"] = trace_id
    return logger.bind(**ctx)