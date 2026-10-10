"""
AI Software Company — Architect Agent (Step 8)

Architecture references:
  Section 3   Agent 2: Architect
  Section 5   Architect: Temp=0.2, JSON mode
              Output: {files, database, routes}
              Rule: max 10 files, max 5 tables,
                    har table: id + created_at
              Validate: id check, path check (max 2)
  Section 9   Every table must have id + created_at
  Section 11  File path security (we validate every path here)
  Section 16  AppState keys: files, database, routes
  Section 24  Audit actions: agent_started, agent_completed

Public API:
  async def run_architect_agent(pm_brief, job_id, user_id, trace_id) -> dict
      returns {"files": [...], "database": {...}, "routes": [...]}
      raises ArchitectAgentError on unrecoverable failure.

Design:
  - Pure orchestration; AI client + validators are separate modules.
  - Every proposed file path is validated by security.path_check.
  - Cost and latency recorded via observability.cost_tracker.record_call.
"""

from __future__ import annotations

import json
from typing import Any

from ai import groq_client
from config import groq_config as cfg
from observability.cost_tracker import record_call
from observability.logger import bind_job, get_logger
from security import path_check
from security import prompt_guard as pg

logger = get_logger("agents.architect")

MAX_FILES = 10
MAX_TABLES = 5

# --- System prompt -------------------------------------------------------
ARCHITECT_SYSTEM_PROMPT = """You are the Architect agent in an AI software company.

You receive a product brief and must produce a SHORT technical plan as a
single JSON object with exactly these keys:

{
  "files": ["relative/path/to/file.ext", "..."],
  "database": {
    "tables": [
      {
        "name": "table_name",
        "columns": [
          {"name": "id", "type": "integer", "primary_key": true},
          {"name": "created_at", "type": "datetime"},
          {"name": "...", "type": "..."}
        ]
      }
    ]
  },
  "routes": [
    {"method": "GET", "path": "/", "handler": "index", "page": "Dashboard"}
  ]
}

Rules (must follow exactly):
- Files list: at most 10 entries. Use POSIX-style relative paths only.
  Forbidden: absolute paths, "../", backslashes, ".env", ".git".
  Prefer this typical layout for FastAPI + Jinja2 apps:
    main.py, database.py, models.py, schemas.py, auth.py,
    templates/base.html, templates/login.html, templates/dashboard.html,
    static/style.css, requirements.txt
  (adapt as needed, but stay <= 10 files).
- Database: at most 5 tables. Every table MUST have BOTH of these columns:
    {"name": "id", "type": "integer", "primary_key": true}
    {"name": "created_at", "type": "datetime"}
  Table and column names: lowercase snake_case, English.
- Routes: 1 to 12 entries. Each route is {"method", "path", "handler", "page"}.
  method is one of GET, POST, PUT, DELETE. path starts with "/".
- Output MUST be valid JSON: double-quoted keys and strings, no trailing
  commas, no comments, no code fences.
- Return ONLY the JSON object. No preamble, no explanation, no fences.
"""

ARCHITECT_REPAIR_PROMPT = """Your previous response was not valid according to the schema.

Return ONLY a single valid JSON object with exactly these keys:

{
  "files": ["relative/path.ext", "..."],
  "database": {"tables": [{"name": "...", "columns": [{"name": "id", "type": "integer", "primary_key": true}, {"name": "created_at", "type": "datetime"}]}]},
  "routes": [{"method": "GET", "path": "/", "handler": "...", "page": "..."}]
}

Constraints:
- files: <= 10 entries, relative POSIX paths, no "..", no backslashes, no ".env".
- database.tables: <= 5 entries, each table MUST have "id" and "created_at" columns.
- routes: 1 to 12 entries, method in {GET, POST, PUT, DELETE}, path starts with "/".
- Return ONLY valid JSON. No markdown, no explanations, no code fences.
"""


class ArchitectAgentError(RuntimeError):
    """Raised when the Architect agent cannot produce a valid plan."""


# --- Validation helpers --------------------------------------------------
_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE"}


def _validate_files(files: Any) -> tuple[bool, str, list[str]]:
    if not isinstance(files, list):
        return False, "files_not_list", []
    if len(files) == 0:
        return False, "files_empty", []
    if len(files) > MAX_FILES:
        return False, f"files_count={len(files)} (max {MAX_FILES})", []

    clean: list[str] = []
    for raw in files:
        if not isinstance(raw, str) or not raw.strip():
            return False, "file_entry_bad", []
        rel = raw.strip()
        try:
            path_check.validate_relative(rel)
        except path_check.PathSecurityError as exc:
            return False, f"file_path_blocked:{exc.reason}", []
        clean.append(rel)
    return True, "ok", clean


def _validate_database(db: Any) -> tuple[bool, str, dict]:
    if not isinstance(db, dict):
        return False, "database_not_object", {}
    tables = db.get("tables")
    if not isinstance(tables, list):
        return False, "tables_not_list", {}
    if len(tables) == 0:
        return False, "tables_empty", {}
    if len(tables) > MAX_TABLES:
        return False, f"tables_count={len(tables)} (max {MAX_TABLES})", {}

    clean_tables: list[dict] = []
    for t in tables:
        if not isinstance(t, dict):
            return False, "table_not_object", {}
        name = t.get("name")
        cols = t.get("columns")
        if not isinstance(name, str) or not name.strip():
            return False, "table_name_bad", {}
        if not isinstance(cols, list) or not cols:
            return False, f"columns_bad_for:{name}", {}

        col_names: set[str] = set()
        clean_cols: list[dict] = []
        for c in cols:
            if not isinstance(c, dict):
                return False, f"column_not_object:{name}", {}
            cname = c.get("name")
            if not isinstance(cname, str) or not cname.strip():
                return False, f"column_name_bad:{name}", {}
            col_names.add(cname.strip())
            clean_cols.append({
                "name": cname.strip(),
                "type": str(c.get("type", "string")).strip() or "string",
                **({"primary_key": True} if c.get("primary_key") else {}),
            })

        if "id" not in col_names:
            return False, f"table_missing_id:{name}", {}
        if "created_at" not in col_names:
            return False, f"table_missing_created_at:{name}", {}

        clean_tables.append({"name": name.strip(), "columns": clean_cols})

    return True, "ok", {"tables": clean_tables}


def _validate_routes(routes: Any) -> tuple[bool, str, list[dict]]:
    if not isinstance(routes, list):
        return False, "routes_not_list", []
    if len(routes) == 0:
        return False, "routes_empty", []
    if len(routes) > 12:
        return False, f"routes_count={len(routes)} (max 12)", []

    clean: list[dict] = []
    for r in routes:
        if not isinstance(r, dict):
            return False, "route_not_object", []
        method = str(r.get("method", "")).strip().upper()
        path = str(r.get("path", "")).strip()
        handler = str(r.get("handler", "")).strip()
        page = str(r.get("page", "")).strip()
        if method not in _ALLOWED_METHODS:
            return False, f"route_method_bad:{method}", []
        if not path.startswith("/"):
            return False, f"route_path_bad:{path}", []
        if not handler:
            return False, "route_handler_empty", []
        clean.append({"method": method, "path": path, "handler": handler, "page": page})
    return True, "ok", clean


def _validate_plan(data: Any) -> tuple[bool, str, dict]:
    if not isinstance(data, dict):
        return False, "not_object", {}

    ok, reason, files = _validate_files(data.get("files"))
    if not ok:
        return False, reason, {}

    ok, reason, database = _validate_database(data.get("database"))
    if not ok:
        return False, reason, {}

    ok, reason, routes = _validate_routes(data.get("routes"))
    if not ok:
        return False, reason, {}

    return True, "ok", {"files": files, "database": database, "routes": routes}


# --- Main entry point ---------------------------------------------------
async def run_architect_agent(
    pm_brief: dict,
    job_id: str,
    user_id: str,
    trace_id: str,
) -> dict:
    """
    Run the Architect agent: PM brief -> {files, database, routes}.

    Raises ArchitectAgentError if no valid JSON is produced after repairs.
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)
    log.info("agent_started", agent="architect")

    # Format PM brief as the user_input body
    brief_text = json.dumps(pm_brief, ensure_ascii=False, indent=2)
    san = pg.sanitize_user_input(brief_text)
    if not san.ok:
        log.warning("architect_input_rejected", reason=san.reason)
        raise ArchitectAgentError(f"invalid_brief:{san.reason}")

    messages = pg.build_messages(ARCHITECT_SYSTEM_PROMPT, san.cleaned)

    # --- First attempt ---
    try:
        result = await groq_client.chat(
            messages,
            temperature=0.2,
            max_tokens=cfg.PRIMARY_MAX_TOKENS,
        )
    except Exception as exc:
        log.error("architect_groq_failed", reason=str(exc))
        raise ArchitectAgentError("groq_failed") from exc

    try:
        record_call(
            job_id=job_id,
            agent="architect",
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

    # --- L3: validate JSON ---
    checked = pg.validate_json_output(result.content, required_keys=[])
    if checked.ok:
        ok, reason, plan = _validate_plan(checked.data)
        if ok:
            log.info(
                "agent_completed",
                agent="architect",
                files=len(plan["files"]),
                tables=len(plan["database"]["tables"]),
                routes=len(plan["routes"]),
            )
            return plan
        else:
            log.warning("architect_schema_invalid", reason=reason)
            last_reason = reason
    else:
        log.warning("architect_json_invalid", reason=checked.reason)
        last_reason = checked.reason

    # --- Repair attempts (max 2) ---
    for repair_attempt in range(1, 3):
        repair_messages = [
            {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"<user_input>\n{san.cleaned}\n</user_input>"},
            {"role": "assistant", "content": result.content[:2000]},
            {"role": "user", "content": ARCHITECT_REPAIR_PROMPT},
        ]
        try:
            result = await groq_client.chat(
                repair_messages,
                temperature=0.1,
                max_tokens=cfg.PRIMARY_MAX_TOKENS,
            )
        except Exception as exc:
            log.error("architect_repair_groq_failed",
                      reason=str(exc), attempt=repair_attempt)
            raise ArchitectAgentError("groq_failed_on_repair") from exc

        try:
            record_call(
                job_id=job_id,
                agent="architect",
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
            ok, reason, plan = _validate_plan(checked.data)
            if ok:
                log.info(
                    "agent_completed",
                    agent="architect",
                    repair_attempts=repair_attempt,
                    files=len(plan["files"]),
                    tables=len(plan["database"]["tables"]),
                    routes=len(plan["routes"]),
                )
                return plan
            last_reason = reason
        else:
            last_reason = checked.reason

    log.error("architect_failed", reason=last_reason)
    raise ArchitectAgentError(f"architect_validation_failed:{last_reason}")


# --- Convenience ---------------------------------------------------------
def pretty(plan: dict) -> str:
    """Return a readable string of the plan (for CLI tests)."""
    return json.dumps(plan, indent=2, ensure_ascii=False)