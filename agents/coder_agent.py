"""
AI Software Company — Coder Agent (Step 10)

Architecture references:
  Section 5   Coder: Temp 0.1, FILE-BY-FILE, RAG cap 1500, repair max 3
  Section 9   10 security checks the generated app must satisfy
  Section 10  10 code validation checks applied to every generated file
  Section 15  File generation order
  Section 14  RAG (200-token chunks, cap 1500)
  Section 16  AppState key: code_files, rag_tokens_used
  F20/F21     RAG chunk cap + Coder RAG cap
  F28         Secret leak prevention (reject any file containing a secret)

Public API:
  async def run_coder_agent(pm_brief, plan, design, job_id, user_id, trace_id)
      -> {"files": [{"path": str, "content": str}, ...],
          "rag_tokens_used": int,
          "failed_files": [...]}
      raises CoderAgentError if the entire run cannot proceed.

Checks applied per file (Section 10, checks 1-8; 9-10 run in Tester Step 12):
  1. JSON structure        -> the AI must return {"path","content"}
  2. File path validation  -> security.path_check.validate_relative
  3. Forbidden path        -> included in path_check
  4. Syntax check          -> ast.parse for .py files
  5. Import check          -> extract imports; flag unknown top-level packages
  6. Secret detection      -> security.secret_scan.contains_secret
  7. Dangerous command     -> block os.system, subprocess, eval, exec, __import__
  8. Dependency validation -> deferred to Step 11 (Dependency Check)
"""

from __future__ import annotations

import ast
import json
import re
from typing import Optional

from ai import groq_client
from config import groq_config as cfg
from observability.cost_tracker import record_call
from observability.logger import bind_job, get_logger
from rag.retriever import format_for_prompt, retrieve_for_topics
from security import path_check, prompt_guard as pg
from security.secret_scan import contains_secret

logger = get_logger("agents.coder")

MAX_REPAIR_ATTEMPTS = 1
MAX_FILE_CONTENT_CHARS = 30000

# --- Section 15 generation order ----------------------------------------
SECTION_15_ORDER = [
    "database.py",
    "models.py",
    "schemas.py",
    "auth.py",
    "main.py",
    "templates/base.html",
    "templates/login.html",
    "templates/dashboard.html",
    "static/style.css",
    "requirements.txt",
    "Dockerfile",
]


def _order_key(path: str) -> tuple[int, str]:
    try:
        return (SECTION_15_ORDER.index(path), path)
    except ValueError:
        return (len(SECTION_15_ORDER), path)


# --- File type classification -------------------------------------------
def _file_kind(path: str) -> str:
    p = path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith(".html") or p.endswith(".htm"):
        return "html"
    if p.endswith(".css"):
        return "css"
    if p.endswith("requirements.txt"):
        return "requirements"
    if p.endswith("dockerfile") or p == "dockerfile":
        return "dockerfile"
    if p.endswith(".md"):
        return "markdown"
    return "other"


# --- RAG topics per file -------------------------------------------------
def _rag_topics_for(path: str) -> list[str]:
    p = path.lower()
    topics: list[str] = []
    if p.endswith("database.py") or p.endswith("models.py"):
        topics += ["sqlalchemy", "model", "orm"]
    if p.endswith("schemas.py"):
        topics += ["pydantic", "schema", "validation"]
    if p.endswith("auth.py"):
        topics += ["fastapi", "route", "pydantic"]
    if p.endswith("main.py"):
        topics += ["fastapi", "route", "endpoint"]
    if p.startswith("templates/") or p.endswith(".html"):
        topics += ["jinja2", "template", "html"]
    if p.endswith(".css") or "style" in p:
        topics += ["tailwind", "css", "style"]
    if p.endswith("requirements.txt"):
        topics += ["fastapi", "sqlalchemy", "pydantic"]
    if not topics:
        topics = ["fastapi"]
    return topics


# --- System prompt -------------------------------------------------------
CODER_SYSTEM_PROMPT = """You are the Coder agent in an AI software company.

MINIMUM LENGTH (HARD requirement, reject if smaller):
- database.py     : at least 80 lines of code
- models.py       : at least 60 lines of code (1 model class per table)
- schemas.py      : at least 40 lines of code
- auth.py         : at least 60 lines of code (hashing + login + logout + CSRF)
- main.py         : at least 100 lines of code (all routes + startup + templates)
- templates/*.html: at least 60 lines of HTML each
- static/style.css: at least 40 lines of CSS
- requirements.txt: at least 8 packages, one per line, no prose

If any file is shorter than the minimum above, the build WILL be rejected.

PER-FILE REQUIREMENTS (these are HARD requirements; a missing item = FAIL):
- database.py   : MUST import sqlalchemy. MUST define a Base class and a
                  `backup_db()` function. MUST read DATABASE_URL from
                  os.getenv("DATABASE_URL", "sqlite:///./app.db").
- models.py     : MUST import sqlalchemy. MUST define at least one ORM
                  model class. EVERY model MUST have `id` and `created_at`
                  columns.
- schemas.py    : MUST import pydantic (BaseModel). MUST define at least
                  one Pydantic model.
- auth.py       : MUST import passlib AND bcrypt. MUST use
                  CryptContext(schemes=["bcrypt"]). MUST define
                  hash_password() and verify_password().
- main.py       : MUST import fastapi (FastAPI). MUST include at least one
                  @app.get or @app.post route. MUST include CSRF token
                  handling on POST forms.
- templates/*.html : Use Jinja2 syntax; extend templates/base.html;
                  include Tailwind CDN script.
- requirements.txt : Use ONLY one package per line, no prose. Must include
                  at minimum: fastapi, uvicorn, sqlalchemy, pydantic,
                  python-multipart, passlib, bcrypt, jinja2.

If any of these is missing, the Tester agent will reject the build.

You write ONE FILE at a time. Given:
  - a product brief (PM agent)
  - a technical plan (Architect agent)
  - a design brief (Designer agent)
  - a single target file path to generate
  - a small RAG context (docs excerpts) that you should follow

return a single JSON object with exactly two keys:

{
  "path": "<the same relative path you were asked to generate>",
  "content": "<the full file contents as a single string>"
}

STRICT RULES:
- Output MUST be valid JSON. Double-quoted keys/strings. No trailing commas.
- Do NOT wrap the JSON in markdown fences or add any prose before/after.
- "content" must be the FULL file, not a diff or snippet.
- Python files: complete, runnable FastAPI project code. Use SQLAlchemy 2.x
  with declarative Base, Pydantic v2, and passlib+bcrypt for passwords.
- HTML files: use Jinja2 template syntax. Extend templates/base.html.
  Use Tailwind CSS via the CDN: <script src="https://cdn.tailwindcss.com"></script>.
- The generated app MUST follow these 10 security rules (Section 9):
  1) passlib + bcrypt for password hashing
  2) HttpOnly cookie for sessions
  3) secrets via os.getenv (never hardcoded)
  4) protected routes check login
  5) Pydantic models for request bodies
  6) SQLAlchemy ORM (no raw SQL string formatting)
  7) a backup_db() helper in database.py
  8) startup event creates tables (Base.metadata.create_all)
  9) every table has id + created_at columns
 10) CSRF token on POST forms
- DO NOT use: os.system, subprocess, eval, exec, __import__, open() on
  arbitrary paths, shutil.rmtree outside of a backup helper.
- DO NOT hardcode any API key, password, or DATABASE_URL. Read from
  os.getenv("DATABASE_URL", "sqlite:///./app.db").
- Keep the file focused and complete. Prefer clarity over cleverness.
"""


CODER_REPAIR_PROMPT = """Your previous response was invalid OR failed validation.

Return ONLY a single valid JSON object:
{"path": "<same path as before>", "content": "<full file contents>"}

Rules:
- Valid JSON, no markdown fences, no prose.
- Complete file contents (not a diff).
- If the target is Python: parseable by ast.parse, no os.system/subprocess/eval/exec.
- If the target is Python: do not hardcode secrets or the DATABASE_URL.
- If the target is database.py: MUST include `import sqlalchemy`, a Base
  class, and a `backup_db()` function.
- If the target is models.py: MUST include `id` AND `created_at` on every
  table.
- If the target is schemas.py: MUST include pydantic / BaseModel.
- If the target is auth.py: MUST include `from passlib.context import
  CryptContext` and use bcrypt.
- If the target is main.py: MUST include `from fastapi import FastAPI`
  and at least one route.
- If the target is requirements.txt: ONLY one package per line, no prose,
  and MUST include fastapi, uvicorn, sqlalchemy, pydantic, python-multipart,
  passlib, bcrypt, jinja2.
- If the previous error mentioned a missing package or symbol, FIX IT.
"""




# Fixed requirements.txt content - we do NOT let AI generate this.
FIXED_REQUIREMENTS = """fastapi==0.142.2
starlette==1.6.0
uvicorn[standard]==0.34.0
sqlalchemy==2.0.36
pydantic==2.10.4
python-multipart==0.0.32
passlib==1.7.4
bcrypt==4.0.1
jinja2==3.1.6
httpx
"""


class CoderAgentError(RuntimeError):
    """Raised when the Coder agent cannot produce a valid project."""


# --- Output parsing ------------------------------------------------------
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+\-]*\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(s: str) -> str:
    """Remove leading/trailing ```lang ... ``` fences if present."""
    if not s:
        return s
    t = s.strip()
    if t.startswith("```"):
        # drop first fence line
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1:]
        # drop trailing fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    return t


def _parse_coder_output(raw: str) -> tuple[bool, str, dict]:
    """
    Return (ok, reason, data) where data = {"path": str, "content": str}.

    Strategy:
      1. Strip markdown fences.
      2. Try raw JSON.
      3. Try to extract the outermost { ... } substring and parse that.
         (GPT-OSS reasoning models sometimes prepend prose like
          "Here is the file:" before the JSON, or append notes after.)
      4. Validate required fields.
    """
    text = _strip_code_fences(raw or "").strip()

    def _validate(data: dict):
        if not isinstance(data, dict):
            return False, "not_object", {}
        path = data.get("path")
        content = data.get("content")
        if not isinstance(path, str) or not path.strip():
            return False, "bad_path", {}
        if not isinstance(content, str) or not content.strip():
            return False, "bad_content", {}
        if len(content) > MAX_FILE_CONTENT_CHARS:
            return False, f"content_too_large:{len(content)}", {}
        return True, "ok", {"path": path.strip(), "content": content}

    # Attempt 1: direct JSON
    try:
        data = json.loads(text)
        return _validate(data)
    except Exception:
        pass

    # Attempt 2: find outermost { ... } and parse that
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            data = json.loads(candidate)
            ok, reason, result = _validate(data)
            if ok:
                return ok, reason, result
        except Exception:
            pass

        # Attempt 3: try line-by-line to find a JSON object that parses
        # (in case there are multiple { } blocks and one is valid)
        for line in text.splitlines():
            ls = line.strip()
            if not ls.startswith("{"):
                continue
            try:
                data = json.loads(ls)
                ok, reason, result = _validate(data)
                if ok:
                    return ok, reason, result
            except Exception:
                continue

    return False, "not_json:no_valid_json_found", {}

# --- Check 4: syntax -----------------------------------------------------
def _check_python_syntax(content: str) -> tuple[bool, str]:
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return False, f"syntax_error:line{exc.lineno}:{exc.msg}"
    except Exception as exc:
        return False, f"syntax_error:{type(exc).__name__}"
    return True, "ok"


# --- Check 5: imports (light) --------------------------------------------
_ALLOWED_TOP_LEVEL = {
    # stdlib (common subset)
    "os", "sys", "json", "re", "time", "datetime", "pathlib", "typing",
    "collections", "functools", "itertools", "contextlib", "secrets",
    "hashlib", "hmac", "uuid", "logging", "asyncio", "random", "string",
    "sqlite3", "shutil", "tempfile", "urllib", "http", "html", "base64",
    "dataclasses", "enum", "abc", "io", "csv", "math", "decimal",
    "warnings", "traceback", "inspect", "threading", "queue", "socket",
    "email", "smtplib", "ssl", "struct", "binascii",
    # web/ORM libs the generated app is allowed to use
    "fastapi", "starlette", "uvicorn", "pydantic", "pydantic_settings",
    "sqlalchemy", "jinja2", "passlib", "bcrypt", "itsdangerous",
    "dotenv", "multipart", "python_multipart",
    # internal modules of the generated app
    "database", "models", "schemas", "auth", "main",
}


def _extract_imports(content: str) -> list[str]:
    """Return the set of top-level module names imported by `content`."""
    names: set[str] = set()
    try:
        tree = ast.parse(content)
    except Exception:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return sorted(names)


def _check_imports(content: str) -> tuple[bool, str, list[str]]:
    """Light import check. Unknown top-levels are warned but not failed."""
    names = _extract_imports(content)
    unknown = [n for n in names if n not in _ALLOWED_TOP_LEVEL]
    # We do not fail; Step 11 (dependency check) will verify the requirements.
    if unknown:
        return True, f"imports_unverified:{','.join(unknown)}", unknown
    return True, "ok", []


# --- Check 6: secrets ----------------------------------------------------
def _check_secrets(content: str) -> tuple[bool, str]:
    if contains_secret(content):
        return False, "secret_detected"
    return True, "ok"


# --- Check 7: dangerous patterns ----------------------------------------
_DANGER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("os_system", re.compile(r"\bos\.system\s*\(")),
    ("subprocess", re.compile(r"\bsubprocess\b")),
    ("eval", re.compile(r"(?<![\w.])eval\s*\(")),
    ("exec", re.compile(r"(?<![\w.])exec\s*\(")),
    ("dunder_import", re.compile(r"__import__\s*\(")),
    ("shell_true", re.compile(r"shell\s*=\s*True")),
    ("rm_rf", re.compile(r"\brm\s+-rf\b")),
    ("fork_bomb", re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;")),
    ("shutil_rmtree", re.compile(r"\bshutil\.rmtree\s*\(")),
]


def _check_dangerous(content: str) -> tuple[bool, str]:
    for name, pat in _DANGER_PATTERNS:
        if pat.search(content):
            return False, f"dangerous:{name}"
    return True, "ok"


# --- Full 8-check validation ---------------------------------------------
def _validate_file(path: str, content: str) -> tuple[bool, str, dict]:
    """
    Apply checks 1-7 (check 8, dependency validation, runs in Step 11).
    Returns (ok, reason, meta).
    """
    meta = {"warnings": []}

    # 2 & 3: path safety
    try:
        rel = path_check.validate_relative(path)
    except path_check.PathSecurityError as exc:
        return False, f"path_blocked:{exc.reason}", meta
    norm_path = str(rel).replace("\\", "/")

    kind = _file_kind(norm_path)

    # 4: syntax (Python only)
    if kind == "python":
        ok, reason = _check_python_syntax(content)
        if not ok:
            return False, reason, meta

        # 5: imports (warn only)
        _, warn, unknown = _check_imports(content)
        if unknown:
            meta["warnings"].append(warn)

    # 5.5: minimum file size (real code, not a stub)
    if kind == "python":
        min_bytes = 2000
    elif kind == "html":
        min_bytes = 1500
    elif kind == "css":
        min_bytes = 1000
    elif kind == "requirements":
        min_bytes = 60   # at least a few packages
    else:
        min_bytes = 200
    if len(content) < min_bytes:
        return False, f"file_too_small:{len(content)}<{min_bytes}", meta

    # 6: secrets (any file type)
    ok, reason = _check_secrets(content)
    if not ok:
        return False, reason, meta

    # 7: dangerous patterns
    ok, reason = _check_dangerous(content)
    if not ok:
        return False, reason, meta

    meta["kind"] = kind
    meta["path"] = norm_path
    return True, "ok", meta


# --- Build the prompt ----------------------------------------------------
def _build_user_body(
    pm_brief: dict,
    plan: dict,
    design: dict,
    target_path: str,
    rag_context: str,
) -> str:
    payload = {
        "pm_brief": pm_brief,
        "technical_plan": plan,
        "design_brief": design,
        "target_path": target_path,
        "existing_project": [f for f in plan.get("files", [])],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    if rag_context:
        body += "\n\n<rag_context>\n" + rag_context + "\n</rag_context>"
    return body


# --- Generate one file ---------------------------------------------------
FILE_HINTS = {
    "database.py": "MUST include: import sqlalchemy, Base class, backup_db() function",
    "models.py": "MUST include: SQLAlchemy Base, every table needs id + created_at",
    "schemas.py": "MUST include: from pydantic import BaseModel",
    "auth.py": "MUST include: from passlib.context import CryptContext, bcrypt, hash_password(), verify_password()",
    "main.py": "MUST include: from fastapi import FastAPI, at least one @app.get route",
    "requirements.txt": "One package per line, no prose. Include: fastapi, uvicorn, sqlalchemy, pydantic, python-multipart, passlib, bcrypt, jinja2",
    "templates/base.html": "Jinja2 syntax, extend base.html, Tailwind CDN",
    "templates/login.html": "Jinja2 syntax, form with csrf_token",
    "templates/dashboard.html": "Jinja2 syntax, extends base",
    "static/style.css": "40+ lines of CSS, mobile-first",
}


def _hint_for(path: str) -> str:
    """Return a short hint string for a file type."""
    for key, hint in FILE_HINTS.items():
        if path.endswith(key):
            return hint
    return ""


async def _generate_one(
    *,
    pm_brief: dict,
    plan: dict,
    design: dict,
    target_path: str,
    job_id: str,
    user_id: str,
    trace_id: str,
    log,
    extra_hints: Optional[list[str]] = None,
) -> tuple[Optional[str], int, list[str], str, str]:
    # Hard-coded requirements.txt - skip AI to prevent junk
    if target_path == "requirements.txt":
        log.info("coder_fixed_file", path=target_path)
        return FIXED_REQUIREMENTS, 0, [], "", ""
    """
    Generate one file. Returns (content_or_None, rag_tokens_used, warnings).
    """
    topics = _rag_topics_for(target_path)
    try:
        rag = retrieve_for_topics(
            topics, top_k=5, min_score=0.0, token_cap=cfg.CODER_RAG_TOKEN_CAP
        )
        rag_context = format_for_prompt(rag.chunks)
        rag_tokens = rag.rag_tokens_used
    except Exception as exc:
        log.warning("coder_rag_failed", path=target_path, err=str(exc))
        rag_context = ""
        rag_tokens = 0

    # If tester gave specific issues, append them as repair hints
    hint_text = ""
    if extra_hints:
        hint_text = "\n\nREPAIR HINTS (previous tester errors):\n- " + \
                    "\n- ".join(str(h)[:200] for h in extra_hints[:6])

    user_body = _build_user_body(pm_brief, plan, design, target_path, rag_context) + hint_text
    san = pg.sanitize_user_input(user_body)
    if not san.ok:
        # Very unlikely: the body is our own JSON + RAG. Truncate and retry once.
        san = pg.sanitize_user_input(user_body[: pg.MAX_INPUT_CHARS])

    messages = pg.build_messages(CODER_SYSTEM_PROMPT, san.cleaned)

    warnings: list[str] = []
    last_reason = "unknown"
    last_attempt_content: Optional[str] = None

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
        try:
            result = await groq_client.chat(
                messages,
                temperature=0.1,
                max_tokens=cfg.PRIMARY_MAX_TOKENS,
            )
        except Exception as exc:
            log.error("coder_groq_failed", path=target_path, err=str(exc))
            raise CoderAgentError("groq_failed") from exc

        try:
            record_call(
                job_id=job_id, agent="coder", model=result.model,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                total_tokens=result.total_tokens, latency_ms=result.latency_ms,
                rag_tokens=rag_tokens, used_fallback=result.used_fallback,
                attempts=result.attempts,
            )
        except Exception:
            pass

        ok, reason, data = _parse_coder_output(result.content)
        if not ok:
            last_attempt_content = result.content  # preserve raw
            last_reason = reason
            log.warning("coder_parse_failed", path=target_path, reason=reason,
                        attempt=attempt)
        else:
            # Force the path to be the one we asked for.
            content = data["content"]
            last_attempt_content = content  # preserve even if validation fails
            vok, vreason, vmeta = _validate_file(target_path, content)
            if vok:
                warnings.extend(vmeta.get("warnings", []))
                return content, rag_tokens, warnings, "", ""
            last_reason = vreason
            log.warning("coder_validate_failed", path=target_path,
                        reason=vreason, attempt=attempt)

        # Repair loop
        messages = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": f"<user_input>\n{san.cleaned}\n</user_input>"},
            {"role": "assistant", "content": (result.content or "")[:2000]},
            {"role": "user", "content": CODER_REPAIR_PROMPT},
        ]

    log.error("coder_file_failed", path=target_path, reason=last_reason)
    # 5-tuple: (None, tokens, warnings, reason, failed_content)
    return None, rag_tokens, warnings, last_reason, (last_attempt_content or "")


# --- Main entry point ---------------------------------------------------
async def run_coder_agent(
    pm_brief: dict,
    plan: dict,
    design: dict,
    job_id: str,
    user_id: str,
    trace_id: str,
    repair_paths: Optional[list[str]] = None,
    extra_hints: Optional[list[str]] = None,
) -> dict:
    """
    Run the Coder agent: (PM + Architect + Designer) -> generated project files.

    Returns:
      {
        "files": [{"path": str, "content": str}, ...],
        "rag_tokens_used": int,
        "failed_files": [{"path": str, "reason": str}, ...],
      }
    Raises CoderAgentError if NO file could be generated.
    """
    log = bind_job(logger, job_id=job_id, user_id=user_id, trace_id=trace_id)
    log.info("agent_started", agent="coder")

    wanted = list(plan.get("files") or [])
    if not wanted:
        raise CoderAgentError("plan_has_no_files")

    wanted.sort(key=_order_key)

    files_out: list[dict] = []
    failed: list[dict] = []
    total_rag_tokens = 0

    for path in wanted:
        try:
            content, rag_tokens, warnings, fail_reason, failed_content = await _generate_one(
                pm_brief=pm_brief, plan=plan, design=design,
                target_path=path,
                job_id=job_id, user_id=user_id, trace_id=trace_id,
                log=log,
                extra_hints=extra_hints,
            )
            total_rag_tokens += rag_tokens
            if content is None:
                failed.append({
                    "path": path,
                    "reason": fail_reason or "repair_exhausted",
                    "content": failed_content or "",
                    "hint": _hint_for(path),
                })
                log.error("coder_file_unrecoverable", path=path, reason=fail_reason)
            else:
                files_out.append({"path": path, "content": content})
                log.info("coder_file_ok", path=path,
                         bytes=len(content), warnings=warnings)
        except CoderAgentError:
            raise
        except Exception as exc:
            failed.append({"path": path, "reason": f"unexpected:{type(exc).__name__}"})
            log.error("coder_file_crashed", path=path, err=str(exc))

    if not files_out:
        raise CoderAgentError("no_files_generated")

    log.info(
        "agent_completed",
        agent="coder",
        files_ok=len(files_out),
        files_failed=len(failed),
        rag_tokens=total_rag_tokens,
    )

    return {
        "files": files_out,
        "rag_tokens_used": total_rag_tokens,
        "failed_files": failed,
    }