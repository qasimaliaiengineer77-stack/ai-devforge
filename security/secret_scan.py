"""
AI Software Company — Secret scanning + redaction (Step 5)

Architecture references:
  Section 12  Secret protection (generated apps, logs, errors)
  F28         Secret leak prevention (logs/errors/apps)
  Section 24  Audit log actions: secret_blocked

Purpose:
  - Detect secrets in text (generated code, logs, error messages).
  - Redact them before they reach logs / user-facing output.
  - Provide scan_text() used by the Coder agent (Step 10) to reject any
    generated file that contains a hardcoded secret.

Design:
  - Pure Python, no AI.
  - All patterns are conservative (avoid false negatives on real secret
    prefixes) but avoid matching innocuous words.
  - redact() is safe to call on any string; it never raises.
"""

import re
from typing import Iterable, NamedTuple

# --- Patterns ------------------------------------------------------------
# Each entry: (name, compiled_regex). Keep names stable for audit logs.
_PATTERN_LIST: list[tuple[str, re.Pattern]] = [
    # Groq API keys
    ("groq_api_key", re.compile(r"gsk_[A-Za-z0-9_\-]{20,}")),
    # GitHub tokens (classic + fine-grained)
    ("github_token_classic", re.compile(r"ghp_[A-Za-z0-9_\-]{20,}")),
    ("github_token_fine", re.compile(r"github_pat_[A-Za-z0-9_\-]{20,}")),
    # OpenAI-style keys (used for reference; we don't use OpenAI)
    ("openai_style_key", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    # Database URLs with credentials
    ("postgres_url", re.compile(r"postgresql(?:\+\w+)?://[^\s\"'<>]+:[^\s\"'<>]+@[^\s\"'<>]+")),
    ("mysql_url", re.compile(r"mysql(?:\+\w+)?://[^\s\"'<>]+:[^\s\"'<>]+@[^\s\"'<>]+")),
    # Session cookies in logs
    ("session_cookie", re.compile(r"ai_company_session=[A-Za-z0-9_\-]+")),
    # Our own session cookie name with different value shapes
    ("set_cookie_session", re.compile(r"set-cookie:\s*ai_company_session=[^\s;]+", re.IGNORECASE)),
    # AWS keys (long shot, but cheap to check)
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    # Private key blocks
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

_REDACTION = "[REDACTED]"


class SecretHit(NamedTuple):
    kind: str
    match: str  # the matched substring (caller may redact or reject)


def find_secrets(text: str) -> list[SecretHit]:
    """
    Return list of secret matches found in `text`.
    Never raises. Empty list means no secrets found.
    """
    if not text:
        return []
    hits: list[SecretHit] = []
    for kind, pat in _PATTERN_LIST:
        for m in pat.finditer(text):
            hits.append(SecretHit(kind=kind, match=m.group(0)))
    return hits


def contains_secret(text: str) -> bool:
    """True if any secret pattern matches. Fast path."""
    if not text:
        return False
    return any(pat.search(text) for _, pat in _PATTERN_LIST)


def redact(text: str) -> str:
    """
    Replace every secret match with [REDACTED]. Never raises.
    If text is None or empty, returns it unchanged.
    """
    if not text:
        return text
    out = text
    for _, pat in _PATTERN_LIST:
        out = pat.sub(_REDACTION, out)
    return out


def redact_many(texts: Iterable[str]) -> list[str]:
    """Apply redact() to each item in an iterable."""
    return [redact(t) for t in texts]


def reject_if_secret(text: str, *, context: str = "") -> tuple[bool, list[SecretHit]]:
    """
    Helper for the Coder agent (Step 10).

    Returns (is_clean, hits).
      - is_clean = True  -> no secrets found, safe to store.
      - is_clean = False -> caller should reject the file and log a
                            `secret_blocked` audit action.

    `context` is a short label (e.g. file path) used by the caller when
    writing the audit log. This function does NOT log anything itself.
    """
    hits = find_secrets(text)
    return (len(hits) == 0, hits)