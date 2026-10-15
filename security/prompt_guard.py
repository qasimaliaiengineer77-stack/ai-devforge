"""
AI Software Company — Prompt injection defense (Step 5)

Architecture references:
  Section 31  Prompt injection defense (3 levels)
  F28         Secret leak prevention

Levels:
  L1 - Input sanitization
       * hard length cap
       * strip control characters (except \n and \t)
       * block common injection phrases ("ignore previous instructions", ...)
       * normalize whitespace (no huge runs)

  L2 - Instruction separation
       * build_messages() puts the system prompt in a distinct system role
         and wraps user input in clear <user_input>...</user_input> markers
         so the model can distinguish trusted instructions from untrusted text.
       * RAG context gets its own role/marker so it can never be confused
         with the system prompt.

  L3 - Output validation
       * validate_json_output() ensures the model returned a JSON object
         with required keys, at a size cap, and containing no secrets.
       * validate_text_output() ensures a plain string is within caps and
         contains no secrets or forbidden substrings.

Design:
  - Pure Python, no AI, no external calls.
  - Never raises for user input; returns structured results so the caller
    can log an audit action and return a safe error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from security.secret_scan import contains_secret

# --- Caps ----------------------------------------------------------------
MAX_INPUT_CHARS = 20000
MAX_OUTPUT_CHARS = 40000
MAX_SYSTEM_CHARS = 4000

# --- L1: Injection patterns (case-insensitive) ---------------------------
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)", re.IGNORECASE),
    re.compile(r"forget\s+(?:all\s+)?(?:previous|prior|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\b", re.IGNORECASE),
    re.compile(r"system\s*[:>]\s*", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(system|assistant)\s*>", re.IGNORECASE),
    re.compile(r"\[/?(system|assistant|inst)\]", re.IGNORECASE),
    re.compile(r"###\s*(instruction|system)", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?(?:system\s+prompt|api\s*key|secret)", re.IGNORECASE),
    re.compile(r"print\s+(?:the\s+)?(?:system\s+prompt|api\s*key|secret)", re.IGNORECASE),
    re.compile(r"output\s+(?:the\s+)?(?:system\s+prompt|api\s*key|secret)", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
]

# Characters we strip (keep \n, \t)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# --- Result types --------------------------------------------------------
@dataclass
class SanitizeResult:
    ok: bool
    cleaned: str
    reason: Optional[str] = None
    matched_pattern: Optional[str] = None


@dataclass
class OutputCheckResult:
    ok: bool
    data: Any = None
    reason: Optional[str] = None


# --- L1: Input sanitization ---------------------------------------------
def sanitize_user_input(text: str) -> SanitizeResult:
    """
    Clean and validate a raw user requirement string.
    Never raises; returns SanitizeResult.
    """
    if not isinstance(text, str):
        return SanitizeResult(ok=False, cleaned="", reason="not_a_string")

    if text == "" or not text.strip():
        return SanitizeResult(ok=False, cleaned="", reason="empty")

    if len(text) > MAX_INPUT_CHARS:
        return SanitizeResult(ok=False, cleaned="", reason="too_long")

    # Strip control chars, keep \n \t
    cleaned = _CONTROL_RE.sub("", text)

    # Collapse runs of spaces/tabs (not newlines)
    cleaned = re.sub(r"[ \t]{3,}", "  ", cleaned)

    # Collapse >2 consecutive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Injection pattern check
    for pat in _INJECTION_PATTERNS:
        if pat.search(cleaned):
            return SanitizeResult(
                ok=False,
                cleaned=cleaned,
                reason="injection_pattern",
                matched_pattern=pat.pattern,
            )

    return SanitizeResult(ok=True, cleaned=cleaned.strip())


# --- L2: Instruction separation -----------------------------------------
USER_INPUT_OPEN = "<user_input>"
USER_INPUT_CLOSE = "</user_input>"
RAG_OPEN = "<rag_context>"
RAG_CLOSE = "</rag_context>"


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def build_messages(
    system_prompt: str,
    user_input: str,
    rag_context: Optional[str] = None,
) -> list[dict]:
    """
    Build a chat-style message list with strict separation:
      - system role  : trusted instructions (from us)
      - user role    : untrusted content, wrapped in <user_input> markers
      - user role    : optional RAG context, wrapped in <rag_context> markers

    The caller MUST pass the sanitized user input (see sanitize_user_input).
    """
    sys_clean = _truncate(system_prompt or "", MAX_SYSTEM_CHARS)

    user_body = f"{USER_INPUT_OPEN}\n{user_input}\n{USER_INPUT_CLOSE}"

    messages: list[dict] = [
        {"role": "system", "content": sys_clean},
        {"role": "user", "content": user_body},
    ]

    if rag_context:
        rag_body = f"{RAG_OPEN}\n{_truncate(rag_context, MAX_SYSTEM_CHARS)}\n{RAG_CLOSE}"
        messages.append({"role": "user", "content": rag_body})

    return messages


# --- L3: Output validation ----------------------------------------------
_FORBIDDEN_OUTPUT_SUBSTRINGS = (
    "system prompt",
    "api key",
    "secret key",
    "-----BEGIN",
)


def validate_text_output(text: str, *, allow_empty: bool = False) -> OutputCheckResult:
    """Validate a plain-text model output."""
    if text is None:
        return OutputCheckResult(ok=False, reason="none")
    if len(text) > MAX_OUTPUT_CHARS:
        return OutputCheckResult(ok=False, reason="too_long")
    if not allow_empty and not text.strip():
        return OutputCheckResult(ok=False, reason="empty")
    if contains_secret(text):
        return OutputCheckResult(ok=False, reason="secret_detected")
    lowered = text.lower()
    for bad in _FORBIDDEN_OUTPUT_SUBSTRINGS:
        if bad.lower() in lowered:
            return OutputCheckResult(ok=False, reason=f"forbidden_substring:{bad}")
    return OutputCheckResult(ok=True, data=text)


def validate_json_output(
    text: str,
    required_keys: Iterable[str] = (),
) -> OutputCheckResult:
    """
    Validate that a model output is a JSON object with the required keys.
    Returns the parsed dict on success.
    """
    base = validate_text_output(text)
    if not base.ok:
        return base

    try:
        data = json.loads(text)
    except Exception:
        return OutputCheckResult(ok=False, reason="not_json")

    if not isinstance(data, dict):
        return OutputCheckResult(ok=False, reason="not_object")

    for key in required_keys:
        if key not in data:
            return OutputCheckResult(ok=False, reason=f"missing_key:{key}")

    return OutputCheckResult(ok=True, data=data)


# --- Convenience --------------------------------------------------------
def short_reason(reason: Optional[str], limit: int = 80) -> str:
    """Trim a reason string for safe logging/display."""
    if not reason:
        return ""
    return reason[:limit]