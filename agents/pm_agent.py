"""
AI Software Company — PM Agent (Step 7)

Architecture references:
  Section 3   Agent 1: PM
  Section 5   PM: Model=openai/gpt-oss-120b, Temp=0.2, JSON mode
              Output: {app_name, features, pages}
              Rule: 3-5 features, English
              Validate: JSON check + repair (max 2)
  Section 16  AppState keys: app_name, features, pages
  Section 24  Audit actions used: agent_started, agent_completed, build_failed
  Section 31  Prompt injection defense (sanitize + validate)

Public API:
  async def run_pm_agent(requirement, job_id, user_id, trace_id) -> dict
      returns {"app_name": str, "features": [str], "pages": [str]}
      raises PMAgentError on unrecoverable failure.

Design:
  - Pure orchestration; the AI client and validators are separate modules.
  - Cost and latency recorded via observability.cost_tracker.record_call.
  - All log lines are redacted by the structlog processor.
"""

from __future__ import annotations

import json
from typing import Any

from ai import groq_client
from config import groq_config as cfg
from observability.cost_tracker import record_call
from observability.logger import bind_job, get_logger
from security import prompt_guard as pg

logger = get_logger("agents.pm")

# --- PM system prompt (English per architecture) -------------------------
PM_SYSTEM_PROMPT = """You are the PM agent in an AI software company.

Your job: given a user's requirement (Urdu, Roman Urdu, or English),
produce a SHORT product brief as a single JSON object with exactly these keys:

{
  "app_name": "Short product name in English (<=40 chars, no spaces at ends)",
  "features": ["3 to 5 short feature sentences in English"],
  "pages": ["1 to 5 short page names in English (e.g., 'Login', 'Dashboard')"]
}

Rules:
- "features" MUST contain between 3 and 5 items (inclusive).
- "pages" MUST contain between 1 and 5 items (inclusive).
- All strings in English. No emojis. No markdown.
- The user's requirement may be in Urdu or Roman Urdu. Translate internally,
  but output only the JSON object in English.
- Output MUST be valid JSON: double-quoted keys and strings, no trailing
  commas, no comments, no code fences.
- If the requirement is vague, choose sensible defaults. Do not ask questions.
- Return ONLY the JSON object. No preamble, no explanation, no fences.
"""

# --- PM repair prompt (used if first JSON is invalid) --------------------
PM_REPAIR_PROMPT = """Your previous response was not valid according to the schema.

Return ONLY a single valid JSON object with exactly these keys:
{
  "app_name": "string",
  "features": ["3 to 5 English feature strings"],
  "pages": ["1 to 5 English page names"]
}

Do not include explanations, markdown, or code fences.
"""


class PMAgentError(RuntimeError):
    """Raised when the PM agent cannot produce a valid brief."""


# --- Validation ----------------------------------------------------------
def _validate_brief(data: Any) -> tuple[bool, str]:
    """Return (is_valid, reason)."""
    if not isinstance(data, dict):
        return False, "not_object"

    app_name = data.get("app_name")
    features = data.get("features")
    pages = data.get("pages")

    if not isinstance(app_name, str) or not app_name.strip():
        return False, "bad_app_name"
    if len(app_name) > 80:
        return False, "app_name_too_long"

    if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        return False, "bad_features"
    if not (3 <= len(features) <= 5):
        return False, f"features_count={len(features)} (need 3-5)"

    if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
        return False, "bad_pages"
    if not (1 <= len(pages) <= 5):
        return False, f"pages_count={len(pages)} (need 1-5)"

    return True, "ok"


def _normalize_brief(data: dict) -> dict:
    """Trim strings and drop empties. Keeps only the three required keys."""
    app_name = str(data.get("app_name", "")).strip()
    features = [str(f).strip() for f in data.get("features", []) if str(f).strip()]
    pages = [str(p).strip() for p in data.get("pages", []) if str(p).strip()]
    return {"app_name": app_name, "features": features, "pages": pages}


# --- Main entry point ---------------------------------------------------
async def run_pm_agent(
    requirement: str,
    job_id: str,
    user_id: str,
    trace_id: str,
) -> dict:
    """
    Run the PM agent: requirement -> {app_name, features, pages}.

    Raises PMAgentError if no valid JSON is produced after repair attempts.
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)
    log.info("agent_started", agent="pm", requirement_len=len(requirement or ""))

    # --- L1: sanitize user input ---
    san = pg.sanitize_user_input(requirement)
    if not san.ok:
        log.warning("pm_input_rejected", reason=san.reason)
        raise PMAgentError(f"invalid_requirement:{san.reason}")

    # --- Build messages (L2: instruction separation) ---
    messages = pg.build_messages(PM_SYSTEM_PROMPT, san.cleaned)

    # --- First attempt ---
    try:
        result = await groq_client.chat(
            messages,
            temperature=0.2,
            max_tokens=cfg.PRIMARY_MAX_TOKENS,
        )
    except Exception as exc:
        log.error("pm_groq_failed", reason=str(exc))
        raise PMAgentError("groq_failed") from exc

    # --- Record cost/tokens ---
    try:
        record_call(
            job_id=job_id,
            agent="pm",
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            latency_ms=result.latency_ms,
            rag_tokens=0,
            used_fallback=result.used_fallback,
            attempts=result.attempts,
        )
    except Exception:
        # Never let cost tracking break the agent
        pass

    # --- L3: validate JSON ---
    checked = pg.validate_json_output(result.content, required_keys=[])
    if checked.ok:
        ok, reason = _validate_brief(checked.data)
        if ok:
            brief = _normalize_brief(checked.data)
            log.info("agent_completed", agent="pm", features=len(brief["features"]),
                     pages=len(brief["pages"]))
            return brief
        else:
            log.warning("pm_schema_invalid", reason=reason)
            last_reason = reason
    else:
        log.warning("pm_json_invalid", reason=checked.reason)
        last_reason = checked.reason

    # --- Repair attempts (max 2) ---
    for repair_attempt in range(1, 3):
        repair_messages = [
            {"role": "system", "content": PM_SYSTEM_PROMPT},
            {"role": "user", "content": f"<user_input>\n{san.cleaned}\n</user_input>"},
            {"role": "assistant", "content": result.content[:2000]},
            {"role": "user", "content": PM_REPAIR_PROMPT},
        ]
        try:
            result = await groq_client.chat(
                repair_messages,
                temperature=0.1,
                max_tokens=cfg.PRIMARY_MAX_TOKENS,
            )
        except Exception as exc:
            log.error("pm_repair_groq_failed", reason=str(exc), attempt=repair_attempt)
            raise PMAgentError("groq_failed_on_repair") from exc

        try:
            record_call(
                job_id=job_id,
                agent="pm",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                latency_ms=result.latency_ms,
                rag_tokens=0,
                used_fallback=result.used_fallback,
                attempts=result.attempts,
            )
        except Exception:
            pass

        checked = pg.validate_json_output(result.content, required_keys=[])
        if checked.ok:
            ok, reason = _validate_brief(checked.data)
            if ok:
                brief = _normalize_brief(checked.data)
                log.info("agent_completed", agent="pm",
                         repair_attempts=repair_attempt,
                         features=len(brief["features"]),
                         pages=len(brief["pages"]))
                return brief
            last_reason = reason
        else:
            last_reason = checked.reason

    log.error("pm_failed", reason=last_reason)
    raise PMAgentError(f"pm_validation_failed:{last_reason}")


# --- Convenience ---------------------------------------------------------
def pretty(brief: dict) -> str:
    """Return a readable string of the brief (for CLI tests)."""
    return json.dumps(brief, indent=2, ensure_ascii=False)