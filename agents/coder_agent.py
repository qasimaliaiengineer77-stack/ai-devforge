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

from pathlib import Path

import ast
import json
import re
from typing import Optional

from ai import groq_client
from config import groq_config as cfg
from pydantic_fix import fix_pydantic_v1_kwargs as _fix_pyd
from cfgfix import fix_config_class as _fix_cfg
from preflight import preflight_fix as _preflight_fix
from autoimport import fix_missing_imports as _fix_missing_imports
from clashfix import fix_field_type_clash as _fix_clash
from goodreq import ensure_requirements as _ensure_req
from agents.template_exports import format_for_prompt as _tpl_frozen
from observability.cost_tracker import record_call
from observability.logger import bind_job, get_logger
from rag.retriever import format_for_prompt, retrieve_for_topics
from security import path_check, prompt_guard as pg
from security.secret_scan import contains_secret

logger = get_logger("agents.coder")

MAX_REPAIR_ATTEMPTS = 2
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



COMMON MISTAKES — AVOID THESE (they will fail validation):

1. Python syntax errors:
   - WRONG: def login(username, password, csrf_token):     # all required, no default OK
   - WRONG: def login(username, password="", csrf_token):  # required after default = syntax error
   - RIGHT: def login(username, password="", csrf_token="") # all with defaults after
   - RIGHT: def login(username, password, csrf_token):      # all without defaults
   - Rule: parameters WITHOUT defaults must come BEFORE parameters WITH defaults.

2. Missing imports:
   - Every SQLAlchemy model needs: `from sqlalchemy import Integer, String, DateTime, func`
   - Every Pydantic model needs: `from pydantic import BaseModel`
   - Every FastAPI route file needs: `from fastapi import FastAPI, Form, Depends`
   - Password hashing needs: `from passlib.context import CryptContext`
   - CSRF/tokens needs: `import secrets`

3. Model requirements (Section 9, rule #9):
   - EVERY ORM model MUST have `id` (Integer, primary_key=True)
   - EVERY ORM model MUST have `created_at` (DateTime, server_default=func.now())
   - Missing either -> auto-fail

4. requirements.txt rules:
   - One package per line, no prose
   - Must include: fastapi, uvicorn, sqlalchemy, pydantic, python-multipart, passlib, bcrypt, jinja2

5. Jinja2 templates:
   - First line MUST be: {% extends "base.html" %} OR <!DOCTYPE html>
   - Every {% block X %} must be closed with {% endblock %}
   - Use {{ csrf_token }} inside <input type="hidden" name="csrf_token" value="...">

6. Password hashing:
   - MUST use: passlib CryptContext OR bcrypt directly
   - MUST have: hash_password() or similar function
   - NEVER store plaintext passwords

SELF-CHECK BEFORE RESPONDING:
   - Is your code valid Python? (mentally run ast.parse)
   - Does every table have id + created_at?
   - Does requirements.txt contain only package lines?

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
- Remember the COMMON MISTAKES list. Most common: parameter without default after parameter with default; missing id/created_at on models; missing passlib/bcrypt imports.
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
typing_extensions>=4.13.0
email-validator>=2.0.0
pyflakes>=3.2.0
"""


# Load frozen templates
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Queued files to append after main.py
_ROUTES_APPENDED: list = []
_AUTH_TEMPLATE_LOADED = (_TEMPLATES_DIR / "auth.py").read_text(encoding="utf-8") if (_TEMPLATES_DIR / "auth.py").exists() else ""
_MAIN_TEMPLATE_LOADED = (_TEMPLATES_DIR / "main.py").read_text(encoding="utf-8") if (_TEMPLATES_DIR / "main.py").exists() else ""
_DB_TEMPLATE_LOADED = (_TEMPLATES_DIR / "database.py").read_text(encoding="utf-8") if (_TEMPLATES_DIR / "database.py").exists() else ""
_BASE_TEMPLATE_LOADED = (_TEMPLATES_DIR / "base.html").read_text(encoding="utf-8") if (_TEMPLATES_DIR / "base.html").exists() else ""
_LOGIN_TEMPLATE_LOADED = (_TEMPLATES_DIR / "login.html").read_text(encoding="utf-8") if (_TEMPLATES_DIR / "login.html").exists() else ""


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


WRONG_IMPORTS = {
    "from fastapi.middleware.sessions import SessionMiddleware":
        "from starlette.middleware.sessions import SessionMiddleware",
    "from fastapi.middleware.cors import CORSMiddleware":
        "from fastapi.middleware.cors import CORSMiddleware",  # actually valid, skip
    "from fastapi.responses import HTMLResponse":  # valid
        "from fastapi.responses import HTMLResponse",
    "from fastapi.templating import Jinja2Templates":  # valid
        "from fastapi.templating import Jinja2Templates",
    "from fastapi.security import OAuth2PasswordBearer":  # valid
        "from fastapi.security import OAuth2PasswordBearer",
    "import fastapi.middleware.sessions":
        "import starlette.middleware.sessions",
    "from fastapi.middleware.trustedhost import TrustedHostMiddleware":
        "from starlette.middleware.trustedhost import TrustedHostMiddleware",
    "from fastapi.middleware.gzip import GZipMiddleware":
        "from starlette.middleware.gzip import GZipMiddleware",
    "from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware":
        "from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware",
}

# Only these need fixing (the wrong ones)
_WRONG_IMPORT_MAP = {
    "from fastapi.middleware.sessions import":
        "from starlette.middleware.sessions import",
    "import fastapi.middleware.sessions":
        "import starlette.middleware.sessions",
    "from fastapi.middleware.trustedhost import":
        "from starlette.middleware.trustedhost import",
    "from fastapi.middleware.gzip import":
        "from starlette.middleware.gzip import",
    "from fastapi.middleware.httpsredirect import":
        "from starlette.middleware.httpsredirect import",
}


def _autocorrect_imports(content: str) -> tuple[str, list[str]]:
    """Fix known wrong imports. Returns (fixed_content, fixes_applied)."""
    fixes = []
    for wrong, right in _WRONG_IMPORT_MAP.items():
        if wrong in content:
            content = content.replace(wrong, right)
            fixes.append(wrong + " -> " + right)
    return content, fixes


def _try_wrap_html(raw: str) -> dict | None:
    """If raw looks like HTML, wrap it as a coder JSON response."""
    t = (raw or "").strip()
    if not t:
        return None
    low = t.lower()
    if low.startswith("<!doctype") or low.startswith("<html") or low.startswith("{%"):
        return {"path": "", "content": t}  # path will be filled by caller
    return None


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

    # Last resort: if content looks like HTML, wrap it
    html_wrap = _try_wrap_html(raw)
    if html_wrap is not None:
        return True, "wrapped_html", html_wrap

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

    # Auto-correct known wrong imports (Python files only)
    if path.endswith(".py"):
        content, fixes = _autocorrect_imports(content)
        for fx in fixes:
            meta["warnings"].append("autocorrect:" + fx)

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
        # Per-file threshold: schemas.py can be small (Pydantic models)
        if target_path.endswith("schemas.py"):
            min_bytes = 400
        elif target_path.endswith("database.py") or target_path.endswith("auth.py"):
            min_bytes = 400  # frozen templates
        else:
            min_bytes = 1500
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
        "file_requirements": _hint_for(target_path),
        "existing_project": [f for f in plan.get("files", [])],
        "frozen_modules": _tpl_frozen(),
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


def _extract_syntax_error_line(reason: str):
    """Parse 'syntax_error:line92:msg' -> (92, msg) or (None, None)."""
    import re as _re
    if not reason or not reason.startswith("syntax_error"):
        return None, None
    m = _re.search(r"line(\d+)", reason)
    if not m:
        return None, None
    line_no = int(m.group(1))
    msg = reason.split(":", 2)[-1] if ":" in reason else reason
    return line_no, msg


async def _repair_lines_ai(
    target_path: str,
    content: str,
    error_line: int,
    error_msg: str,
    log,
) -> str | None:
    """Send only the broken lines to AI, splice back."""
    lines = content.splitlines()
    if not (1 <= error_line <= len(lines)):
        return None
    ctx_start = max(0, error_line - 11)
    ctx_end = min(len(lines), error_line + 10)
    block = "\n".join(lines[ctx_start:ctx_end])

    prompt = (
        f"You are fixing a Python syntax error.\n"
        f"File: {target_path}\n"
        f"Error at line {error_line}: {error_msg}\n\n"
        f"Here is the block (lines {ctx_start+1} to {ctx_end}):\n"
        f"---BEGIN---\n{block}\n---END---\n\n"
        f"Return ONLY the corrected block, wrapped in ---BEGIN--- / ---END---. "
        f"Do NOT add explanations. Keep the same code style."
    )

    try:
        r = await groq_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        txt = r.content or ""
        b1 = txt.find("---BEGIN---")
        b2 = txt.find("---END---")
        if b1 == -1 or b2 == -1 or b2 <= b1:
            return None
        new_block = txt[b1 + 11:b2].strip("\n")
        new_lines = new_block.splitlines()
        merged = lines[:ctx_start] + new_lines + lines[ctx_end:]
        return "\n".join(merged)
    except Exception as e:
        log.warning("line_repair_ai_failed", err=type(e).__name__)
        return None


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

    if target_path == "database.py" and _DB_TEMPLATE_LOADED:
        log.info("coder_frozen_file", path=target_path)
        return _DB_TEMPLATE_LOADED, 0, [], "", ""

    if target_path.replace("\\", "/") == "templates/base.html" and _BASE_TEMPLATE_LOADED:
        log.info("coder_frozen_file", path=target_path)
        return _BASE_TEMPLATE_LOADED, 0, [], "", ""

    if target_path.replace("\\", "/") == "templates/login.html" and _LOGIN_TEMPLATE_LOADED:
        log.info("coder_frozen_file", path=target_path)
        return _LOGIN_TEMPLATE_LOADED, 0, [], "", ""

    if target_path == "auth.py" and _AUTH_TEMPLATE_LOADED:
        log.info("coder_frozen_file", path=target_path)
        return _AUTH_TEMPLATE_LOADED, 0, [], "", ""

    if target_path == "main.py" and _MAIN_TEMPLATE_LOADED:
        log.info("coder_frozen_file_routes", path=target_path)
        routes_code = await _generate_app_routes(pm_brief, plan, design, target_path, log)
        if routes_code:
            _ROUTES_APPENDED.append({"path": "routes/main_routes.py", "content": routes_code})
            _ROUTES_APPENDED.append({"path": "routes/__init__.py", "content": ""})
            log.info("coder_routes_file_queued", chars=len(routes_code))
        return _MAIN_TEMPLATE_LOADED, 0, [], "", ""
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
        log.info("coder_raw_response", path=target_path, attempt=attempt,
                 raw_len=len(result.content or ""),
                 model=getattr(result, "model", "?"),
                 first80=(result.content or "")[:80])
        ok, reason, data = _parse_coder_output(result.content)
        if not ok:
            last_attempt_content = result.content  # preserve raw
            last_reason = reason
            log.warning("coder_parse_failed", path=target_path, reason=reason,
                        attempt=attempt)
        else:
            # Bug E fix: reject responses where AI returned a different file.
            _ai_p = (data.get("path") or "").strip().replace("\\", "/")
            _want_p = target_path.strip().replace("\\", "/")
            _path_bad = bool(_ai_p) and (
                _ai_p.rsplit("/", 1)[-1] != _want_p.rsplit("/", 1)[-1]
            )
            # Force the path to be the one we asked for.
            content = data["content"]
            last_attempt_content = content  # preserve even if validation fails
            if _path_bad:
                vok, vreason, vmeta = (
                    False,
                    f"path_mismatch:got={_ai_p}:want={target_path}",
                    {},
                )
                log.warning("coder_path_mismatch", path=target_path,
                            got=_ai_p, attempt=attempt)
            else:
                vok, vreason, vmeta = _validate_file(target_path, content)
            if vok:
                warnings.extend(vmeta.get("warnings", []))
                return content, rag_tokens, warnings, "", ""
            last_reason = vreason

            # Line-level repair for syntax errors (cheap, safe)
            if vreason.startswith("syntax_error") and attempt == 1:
                err_line, err_msg = _extract_syntax_error_line(vreason)
                if err_line:
                    log.info("line_repair_start", path=target_path, line=err_line)
                    fixed = await _repair_lines_ai(
                        target_path, content, err_line, err_msg, log
                    )
                    if fixed:
                        vok2, vreason2, vmeta2 = _validate_file(target_path, fixed)
                        if vok2:
                            log.info("line_repair_ok", path=target_path)
                            warnings.extend(vmeta2.get("warnings", []))
                            return fixed, rag_tokens, warnings, "", ""
                        else:
                            log.warning("line_repair_still_bad", reason=vreason2)

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




# --- Cross-file import repair -------------------------------------------
_ALIAS_SUFFIX_MAP = {
    "read": ("out", "response", "schema"),
    "out": ("read", "response"),
    "response": ("out", "read"),
    "in": ("create", "input", "request"),
    "input": ("create", "in", "request"),
    "create": ("in", "input"),
    "update": ("edit", "patch"),
}


def _defined_names(content: str) -> set:
    names = set()
    try:
        tree = ast.parse(content)
    except Exception:
        return names
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def _find_similar(name: str, available: set) -> str:
    n = name.lower()
    for k, vs in _ALIAS_SUFFIX_MAP.items():
        if n.endswith(k):
            base = n[: -len(k)]
            for v in vs:
                for c in available:
                    cl = c.lower()
                    if cl.startswith(base) and cl.endswith(v):
                        return c
    if len(n) >= 4:
        for c in available:
            if c.lower().startswith(n[:4]) and c != name:
                return c
    return ""


def _cross_file_alias_repair(files: list) -> list:
    """Add alias lines when a file imports a name that target module lacks."""
    py = {f["path"]: f["content"] for f in files if f["path"].endswith(".py")}
    defined = {p: _defined_names(c) for p, c in py.items()}

    aliases_to_add = {}
    for path, content in py.items():
        try:
            tree = ast.parse(content)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level or not node.module:
                continue
            target_file = node.module + ".py"
            if target_file not in py:
                # Try as package path
                target_file = node.module.replace(".", "/") + ".py"
            if target_file not in py:
                continue
            avail = defined.get(target_file, set())
            for al in node.names:
                if al.name == "*" or al.name in avail:
                    continue
                similar = _find_similar(al.name, avail)
                if similar:
                    aliases_to_add.setdefault(target_file, {})[al.name] = similar

    if not aliases_to_add:
        return files

    new_contents = {}
    for target, amap in aliases_to_add.items():
        lines = ["", "# Auto-generated aliases (cross-file import repair)"]
        for missing, similar in amap.items():
            lines.append(f"{missing} = {similar}  # noqa: F401")
        new_contents[target] = py[target].rstrip() + "\n" + "\n".join(lines) + "\n"

    out = []
    for f_ in files:
        if f_["path"] in new_contents:
            out.append({"path": f_["path"], "content": new_contents[f_["path"]]})
        else:
            out.append(f_)
    return out




# --- Strip relative imports ---------------------------------------------
def _strip_relative_imports(files: list) -> list:
    """Remove leading dots from relative imports in .py files."""
    out = []
    for f in files:
        if not f["path"].endswith(".py"):
            out.append(f)
            continue
        content = f["content"]
        new_lines = []
        changed = False
        for line in content.splitlines(keepends=True):
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if stripped.startswith("from . import "):
                # "from . import X" -> "import X"
                line = indent + "import " + stripped[len("from . import "):]
                changed = True
            elif stripped.startswith("from ."):
                # "from .mod import X" -> "from mod import X"
                line = indent + "from " + stripped[len("from ."):].lstrip()
                changed = True
            elif stripped.startswith("import ."):
                line = indent + "import " + stripped[len("import ."):].lstrip(".")
                changed = True
            new_lines.append(line)
        if changed:
            f = {"path": f["path"], "content": "".join(new_lines)}
        out.append(f)
    return out




# --- TemplateResponse signature repair ----------------------------------
import re as _re_tresp

_TRESP_RE = _re_tresp.compile(
    r"TemplateResponse\s*\(\s*"           # TemplateResponse(
    r"([\"\'])([^\"\']+)\1\s*,\s*"     # "filename.html",
    r"\{\s*([\"\'])request\3\s*:\s*"   # {"request":
    r"([A-Za-z_][A-Za-z0-9_]*)\s*,?"       # varname,
)


def _repair_template_response(files: list) -> list:
    """Rewrite old TemplateResponse("x.html", {"request": r, ...})
    to new  TemplateResponse(r, "x.html", {...})."""
    out = []
    for f in files:
        if not f["path"].endswith(".py"):
            out.append(f)
            continue
        content = f["content"]
        new_content = _TRESP_RE.sub(
            lambda m: (
                'TemplateResponse(' + m.group(4) + ', '
                + m.group(1) + m.group(2) + m.group(1) + ', {'
            ),
            content,
        )
        if new_content != content:
            out.append({"path": f["path"], "content": new_content})
        else:
            out.append(f)
    return out




# --- Ensure static/ folder exists if main.py mounts it ------------------
def _ensure_static_dir(files: list) -> list:
    """If any .py file mounts StaticFiles(directory='static') but no
    static/ file exists, add a placeholder static/style.css."""
    has_static_mount = False
    has_static_file = False
    for f in files:
        p = f["path"]
        if p.startswith("static/"):
            has_static_file = True
        if p.endswith(".py"):
            c = f["content"]
            if "StaticFiles" in c and '"static"' in c:
                has_static_mount = True
            if "StaticFiles" in c and "'static'" in c:
                has_static_mount = True
    if has_static_mount and not has_static_file:
        files = list(files) + [{
            "path": "static/style.css",
            "content": "/* placeholder so StaticFiles mount succeeds */\n",
        }]
    return files




async def _generate_app_routes(pm_brief, plan, design, target_path, log):
    """Ask AI for app-specific routes only, to insert into main.py template."""
    pages = pm_brief.get("pages", []) or []
    features = pm_brief.get("features", []) or []
    routes = plan.get("routes", []) or []
    frozen = {("GET", "/login"), ("POST", "/login"), ("POST", "/logout"), ("GET", "/health")}
    app_routes = []
    for r in routes:
        m = (r.get("method") or "GET").upper()
        p = r.get("path") or ""
        if (m, p) not in frozen:
            app_routes.append(r)
    if not app_routes:
        app_routes = [{"method": "GET", "path": "/", "handler": "dashboard"}]

    system = (
        "You are writing ONLY the app-specific route functions for a FastAPI app. "
        "main.py already has in scope: app (FastAPI), templates (Jinja2Templates), "
        "get_db, get_current_user (returns Optional[models.User]), "
        "get_session_id, generate_csrf_token, validate_csrf_token, "
        "models, schemas, auth (modules), Request, Form, Depends, HTTPException, "
        "status, RedirectResponse, HTMLResponse, select, Session. "
        "DO NOT re-import any of these. Output ONLY route functions as raw Python code. "
        "Use @app.get(...) and @app.post(...). Use templates.TemplateResponse(request, \"file.html\", {...}). "
        "AVAILABLE TEMPLATES (use only these in TemplateResponse): "
        + ", ".join(plan.get("files", [])) + ". "
        "For login-required routes, use user: Optional[models.User] = Depends(get_current_user). "
        "For POST forms, accept csrf_token: str = Form(...) and validate. "
        "No markdown, no JSON, no imports. "
        + ("\\n\\n" + _tpl_frozen() if _tpl_frozen() else "")
    )
    import json as _json
    body = _json.dumps({
        "app_name": pm_brief.get("app_name", "App"),
        "features": features, "pages": pages,
        "routes_to_implement": app_routes,
    }, ensure_ascii=False, indent=2)
    try:
        from security import prompt_guard as pg
        from ai.groq_client import chat as _gc
        san = pg.sanitize_user_input(body)
        if not san.ok:
            san = pg.sanitize_user_input(body[: pg.MAX_INPUT_CHARS])
        msgs = pg.build_messages(system, san.cleaned)
        r = await _gc(msgs, temperature=0.1, max_tokens=4000)
        code = (r.content or "").strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1] if "\n" in code else code
            code = code.rsplit("```", 1)[0]
        ok, why = _validate_routes_fragment(code)
        if not ok:
            log.warning("coder_routes_invalid", reason=why, chars=len(code), preview=code[:120])
            # Retry up to 2 more times with stricter prompt
            for _retry in range(2):
                strict_prompt = (
                    system + "\n\nCRITICAL: Your previous response was INVALID ("
                    + why + "). Return ONLY the FULL function definitions. "
                    "Each function MUST have its own @app.get(...) or @app.post(...) decorator "
                    "and complete `async def name(request: Request, ...):` signature. "
                    "Do NOT return bare dictionaries. Do NOT return fragments. "
                    "Return complete, runnable Python functions only."
                )
                msgs2 = [
                    {"role": "system", "content": strict_prompt},
                    {"role": "user", "content": san.cleaned},
                ]
                try:
                    r2 = await _gc(msgs2, temperature=0.1, max_tokens=6000)
                    code2 = (getattr(r2, "content", "") or "").strip()
                    if code2.startswith("```"):
                        code2 = code2.split("\n", 1)[1] if "\n" in code2 else code2
                        code2 = code2.rsplit("```", 1)[0]
                    ok2, why2 = _validate_routes_fragment(code2)
                    if ok2:
                        log.info("coder_routes_generated_retry_ok", chars=len(code2), retry=_retry + 1)
                        return code2
                    log.warning("coder_routes_retry_failed", retry=_retry + 1, reason=why2)
                except Exception as _e:
                    log.warning("coder_routes_retry_exc", err=str(_e)[:200])

            # Fallback: minimal valid dashboard route
            log.warning("coder_routes_using_fallback")
            return (
                "from fastapi import APIRouter, Request, Depends\n"
                "from fastapi.responses import HTMLResponse, RedirectResponse\n"
                "from typing import Optional\n"
                "from main import templates, get_current_user, render_first_available\n"
                "import models\n\n"
                "router = APIRouter()\n\n"
                "@router.get(\"/\", response_class=HTMLResponse)\n"
                "async def dashboard(request: Request, user: Optional[models.User] = Depends(get_current_user)):\n"
                "    if not user:\n"
                "        return RedirectResponse(url=\"/login\", status_code=302)\n"
                "    return render_first_available(request, [\"dashboard.html\", \"home.html\", \"index.html\", \"login.html\"], {\"user\": user})\n"
            )
        log.info("coder_routes_generated", chars=len(code))
        return code
    except Exception as exc:
        log.warning("coder_routes_failed", err=str(exc))
        return (
            '@app.get("/", response_class=HTMLResponse)\n'
            'async def dashboard(request: Request, user: Optional[models.User] = Depends(get_current_user)):\n'
            '    if not user:\n'
        '        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)\n'
            '    return templates.TemplateResponse(request, "dashboard.html", {"user": user})\n'
        )




# --- Fix base.html self-extend bug -------------------------------------
def _fix_base_template(files: list) -> list:
    """If base.html starts with {% extends "base.html" %}, strip that line."""
    out = []
    for f in files:
        p = f["path"].replace("\\", "/")
        if p.endswith("base.html"):
            c = f["content"]
            stripped = c.lstrip()
            if stripped.startswith('{% extends "base.html" %}'):
                # Remove the first line (extends) and any leading blank lines
                parts = c.split("\n", 1)
                c = parts[1] if len(parts) > 1 else ""
                c = c.lstrip("\n")
                f = {"path": f["path"], "content": c}
        out.append(f)
    return out




# --- SQLAlchemy 2.0 Mapped[...] fix ------------------------------------
import re as _re_sqla

_REL_PATTERNS = [
    # tasks: List["Task"] = relationship(  ->  Mapped[list["Task"]]
    (_re_sqla.compile(r'(\w+)\s*:\s*List\["(\w+)"\]\s*=\s*relationship\s*\('),
     lambda m: f'{m.group(1)}: Mapped[list["{m.group(2)}"]] = relationship('),
    # tasks: List[Task] = relationship(  (unquoted)  ->  Mapped[list["Task"]]
    (_re_sqla.compile(r'(\w+)\s*:\s*List\[(\w+)\]\s*=\s*relationship\s*\('),
     lambda m: f'{m.group(1)}: Mapped[list["{m.group(2)}"]] = relationship('),
    # tasks: list["Task"] = relationship(
    (_re_sqla.compile(r'(\w+)\s*:\s*list\["(\w+)"\]\s*=\s*relationship\s*\('),
     lambda m: f'{m.group(1)}: Mapped[list["{m.group(2)}"]] = relationship('),
    # tasks: list[Task] = relationship(  (unquoted)
    (_re_sqla.compile(r'(\w+)\s*:\s*list\[(\w+)\]\s*=\s*relationship\s*\('),
     lambda m: f'{m.group(1)}: Mapped[list["{m.group(2)}"]] = relationship('),
    # owner: "User" = relationship(
    (_re_sqla.compile(r'(\w+)\s*:\s*"(\w+)"\s*=\s*relationship\s*\('),
     lambda m: f'{m.group(1)}: Mapped["{m.group(2)}"] = relationship('),
    # owner: User = relationship(  (bare capital)
    (_re_sqla.compile(r'(\w+)\s*:\s*([A-Z]\w*)\s*=\s*relationship\s*\('),
     lambda m: f'{m.group(1)}: Mapped["{m.group(2)}"] = relationship('),
]


def _fix_sqlalchemy_relationships(files: list) -> list:
    """Add Mapped[...] type hints to SQLAlchemy relationship declarations."""
    out = []
    for f in files:
        p = f["path"]
        if not p.endswith(".py") or "models" not in p:
            out.append(f)
            continue
        content = f["content"]
        changed = False

        # First apply the substitutions
        for regex, repl in _REL_PATTERNS:
            new_content = regex.sub(repl, content)
            if new_content != content:
                content = new_content
                changed = True

        # Then ensure Mapped import is present IF any Mapped[...] got added
        if "Mapped[" in content and "import Mapped" not in content:
            if "from sqlalchemy.orm import relationship" in content:
                content = content.replace(
                    "from sqlalchemy.orm import relationship",
                    "from sqlalchemy.orm import Mapped, relationship",
                    1,
                )
                changed = True
            elif "from sqlalchemy.orm import" in content:
                # Generic: add Mapped to any orm import line
                content = content.replace(
                    "from sqlalchemy.orm import ",
                    "from sqlalchemy.orm import Mapped, ",
                    1,
                )
                changed = True
            else:
                # No orm import at all — add one right after sqlalchemy import
                content = content.replace(
                    "from sqlalchemy import ",
                    "from sqlalchemy.orm import Mapped\nfrom sqlalchemy import ",
                    1,
                )
                changed = True

        if changed:
            out.append({"path": p, "content": content})
        else:
            out.append(f)
    return out




# --- Auto-prefix auth-only functions with auth. -------------------------
import re as _re_auth

_AUTH_ONLY = ["hash_password", "verify_password", "bcrypt_hash_password"]


def _fix_auth_calls(files: list) -> list:
    """In main.py (or any .py), prefix bare calls to auth-only funcs with auth."""
    out = []
    for f in files:
        p = f["path"]
        if not p.endswith(".py") or p.endswith("auth.py"):
            out.append(f)
            continue
        content = f["content"]
        orig = content

        # Skip if 'def <name>' exists in the same file (locally defined)
        for name in _AUTH_ONLY:
            if _re_auth.search(rf"\bdef\s+{name}\b", content):
                continue
            # Replace bare `name(` with `auth.name(`
            # Avoid matching `auth.name(` or `something.name(`
            pattern = _re_auth.compile(rf"(?<![\w.]){name}\s*\(")
            content = pattern.sub(f"auth.{name}(", content)

        # Ensure `auth` is imported when we made a change
        if content != orig and "import auth" not in content:
            # Insert after the last "import models" style line or at top
            if "from database import" in content:
                content = content.replace(
                    "from database import",
                    "import auth\nfrom database import",
                    1,
                )
            else:
                content = "import auth\n" + content

        out.append({"path": p, "content": content} if content != orig else f)
    return out




# --- Strip {{ config.* }} from AI templates (undefined in template) -----
import re as _re_cfg

_CFG_RE = _re_cfg.compile(r"\{\{\s*config(\.[\w.]+|\[[^\]]+\])*\s*\}\}")


def _fix_template_config(files: list) -> list:
    """Remove {{ config.X }} expressions from .html files — they're undefined."""
    out = []
    for f in files:
        p = f["path"].replace("\\", "/")
        if not p.endswith(".html"):
            out.append(f)
            continue
        c = f["content"]
        new = _CFG_RE.sub("", c)
        if new != c:
            out.append({"path": f["path"], "content": new})
        else:
            out.append(f)
    return out




# --- Pydantic v1 -> v2 keyword fix --------------------------------------
def _fix_pydantic_kwargs_files(files: list) -> list:
    """Run pydantic_fix on every .py file content."""
    out = []
    for f in files:
        p = f["path"]
        if not p.endswith(".py"):
            out.append(f)
            continue
        try:
            new_content, ch = _fix_pyd(f["content"])
            new_content, ch2 = _fix_cfg([{"path": p, "content": new_content}])
            new_content = ch2[0]["content"] if ch2 else new_content
            ch = ch + [f"cfgfix: {p}"] if ch2 else ch
        except Exception:
            out.append(f)
            continue
        if ch:
            out.append({"path": p, "content": new_content})
        else:
            out.append(f)
    return out




# --- Cross-file static repair loop (max 3 attempts) --------------------
async def _cross_file_repair_loop(files: list, log, max_attempts: int = 3) -> list:
    from security.static_checks import run_all as _run_static
    try:
        from agents.template_exports import all_template_exports as _all_exports
    except Exception:
        _all_exports = None

    FROZEN = {"main.py", "auth.py", "database.py", "templates/base.html"}

    for attempt in range(1, max_attempts + 1):
        issues = _run_static(files)
        cross = [i for i in issues if ":missing:" in i or "undefined name" in i]
        if not cross:
            log.info("cross_file_check_passed", attempt=attempt)
            return files

        by_file = {}
        for i in cross:
            parts = i.split(":", 3)
            fname = parts[0] if parts else ""
            by_file.setdefault(fname, []).append(i)

        log.warning("cross_file_repair_start", attempt=attempt,
                    files=list(by_file.keys()), count=len(cross))

        exports_text = ""
        if _all_exports:
            ex = _all_exports()
            chunks = []
            for mod, info in ex.items():
                names = info.get("names", [])
                if names:
                    chunks.append(f"{mod}.py exports: {', '.join(names)}")
            exports_text = "\n".join(chunks)

        project_names = []
        for f_ in files:
            if not f_["path"].endswith(".py"):
                continue
            try:
                tree = ast.parse(f_["content"])
                for n in tree.body:
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        project_names.append(f"{f_['path']}: {n.name}")
            except Exception:
                continue
        project_names_text = "\n".join(project_names[:80])

        new_files = []
        for f_ in files:
            if f_["path"] not in by_file or f_["path"] in FROZEN:
                new_files.append(f_)
                continue

            errs = "\n".join(by_file[f_["path"]])
            prompt = (
                f"This file has static errors. Rewrite it COMPLETELY.\n"
                f"File: {f_['path']}\n\n"
                f"ERRORS:\n{errs}\n\n"
                f"AVAILABLE NAMES IN FROZEN MODULES:\n{exports_text}\n\n"
                f"OTHER PROJECT FILES' EXPORTS:\n{project_names_text}\n\n"
                f"RULES:\n"
                f"- Only import names that actually exist above.\n"
                f"- If a name is missing, use the closest existing one (e.g. NoteOut not NoteRead).\n"
                f"- Do NOT redefine templates, get_db, get_current_user if they come from a frozen module - import them.\n"
                f"- Return ONLY the full file content, no markdown, no JSON."
            )
            try:
                from security import prompt_guard as _pg
                from ai.groq_client import chat as _gc
                san = _pg.sanitize_user_input(prompt)
                if not san.ok:
                    san = _pg.sanitize_user_input(prompt[: _pg.MAX_INPUT_CHARS])
                msgs = [
                    {"role": "system", "content": "You are fixing static errors in one Python file."},
                    {"role": "user", "content": san.cleaned},
                ]
                r = None
                nc = ""
                for _try in range(3):
                    try:
                        r = await _gc(msgs, temperature=0.1, max_tokens=6000)
                    except Exception as _e:
                        log.warning("ensure_files_call_failed", path=target, err=str(_e))
                        r = None
                    if r is not None and (r.content or "").strip():
                        nc = (r.content or "").strip()
                        break
                    await asyncio.sleep(2)
                if nc.startswith("```"):
                    nc = nc.split("\n", 1)[1] if "\n" in nc else nc
                    nc = nc.rsplit("```", 1)[0]
                if nc:
                    new_files.append({"path": f_["path"], "content": nc})
                    log.info("cross_file_rewrite", path=f_["path"], chars=len(nc))
                else:
                    new_files.append(f_)
            except Exception as exc:
                log.warning("cross_file_rewrite_failed", path=f_["path"], err=str(exc))
                new_files.append(f_)

        files = new_files

    log.warning("cross_file_repair_exhausted", attempts=max_attempts)
    return files




# --- Ensure every planned file exists ----------------------------------
async def _ensure_planned_files(files: list, plan: dict, log) -> list:
    planned = [f for f in (plan.get("files") or []) if isinstance(f, str)]
    if not planned:
        return files
    present = {f["path"].replace("\\", "/") for f in files}
    FROZEN = {"database.py", "auth.py", "templates/base.html"}
    missing = [p for p in planned if p.replace("\\", "/") not in present and p.replace("\\", "/") not in FROZEN]
    if not missing:
        log.info("ensure_files_ok", total=len(planned))
        return files
    log.warning("ensure_files_missing", missing=missing)
    for target in missing[:6]:
        for attempt in (1, 2):
            try:
                _hint = ""
                _tables = (plan.get("database") or {}).get("tables", []) or []
                if target.endswith("models.py") and _tables:
                    _l = ["MODEL CLASSES REQUIRED (define EXACTLY these):"]
                    for _t in _tables:
                        _nm = (_t.get("name") or "").rstrip("s").capitalize() or "Item"
                        _cols = [c.get("name") for c in (_t.get("columns") or []) if c.get("name")]
                        _l.append(f"  - class {_nm}(Base): columns = {', '.join(_cols)}")
                    _hint = "\n".join(_l) + "\n\n"
                elif target.endswith("schemas.py") and _tables:
                    _l = ["PYDANTIC MODELS REQUIRED (define EXACTLY these):"]
                    for _t in _tables:
                        _nm = (_t.get("name") or "").rstrip("s").capitalize() or "Item"
                        _l.append(f"  - class {_nm}Base, {_nm}Create, {_nm}Update, {_nm}Out (BaseModel)")
                    _hint = "\n".join(_l) + "\n\n"

                prompt = (
                    f"Generate ONLY this file: {target}\n\n"
                    + _hint +
                    f"Rules:\n"
                    f"- Return ONLY the full file content, no markdown, no JSON.\n"
                    f"- For .py: valid Python, use imports from sqlalchemy, fastapi, pydantic.\n"
                    f"- If schemas.py: Pydantic v2 models (BaseModel), use Field(pattern=...) not regex=.\n"
                    f"- If models.py: SQLAlchemy 2.0, every column Mapped[...], every relationship Mapped[list[\"X\"]].\n"
                    f"- For .html: Jinja2 template extending base.html, use {{% block content %}}.\n"
                )
                from security import prompt_guard as _pg
                from ai.groq_client import chat as _gc
                san = _pg.sanitize_user_input(prompt)
                if not san.ok:
                    san = _pg.sanitize_user_input(prompt[: _pg.MAX_INPUT_CHARS])
                msgs = [
                    {"role": "system", "content": "You write one complete file. Return raw content only."},
                    {"role": "user", "content": san.cleaned},
                ]
                r = await _gc(msgs, temperature=0.1, max_tokens=6000)
                nc = (r.content or "").strip()
                if nc.startswith("```"):
                    nc = nc.split("\n", 1)[1] if "\n" in nc else nc
                    nc = nc.rsplit("```", 1)[0]
                if nc and len(nc) > 20:
                    files.append({"path": target, "content": nc})
                    log.info("ensure_files_created", path=target, chars=len(nc), attempt=attempt)
                    break
            except Exception as exc:
                log.warning("ensure_files_failed", path=target, attempt=attempt, err=str(exc))
    return files




# --- Validate routes fragment before merging into main.py ---------------
def _validate_routes_fragment(code: str) -> tuple[bool, str]:
    """Check that the AI's routes fragment parses as valid Python at module level."""
    if not code or not code.strip():
        return False, "empty"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax:{e.lineno}:{e.msg}"
    # Must contain at least one @app.get / @app.post / @app.put / @app.delete
    has_router = False
    has_route_fn = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "router":
                    has_router = True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                src = ast.unparse(dec)
                if src.startswith("router."):
                    has_route_fn = True
                    break
    if not has_router:
        return False, "no_router_object"
    if not has_route_fn:
        return False, "no_route_functions"
    return True, "ok"




# --- Ensure frozen files exist (even if Architect didn't list them) -----
def _add_missing_frozen_files(files: list) -> list:
    """Append any frozen template that is not already in `files`."""
    present = {f["path"].replace("\\", "/") for f in files}
    frozen = [
        ("database.py", _DB_TEMPLATE_LOADED),
        ("auth.py", _AUTH_TEMPLATE_LOADED),
        ("templates/base.html", _BASE_TEMPLATE_LOADED),
        ("templates/login.html", _LOGIN_TEMPLATE_LOADED),
    ]
    for path, tpl in frozen:
        if path in present or not tpl:
            continue
        files.append({"path": path, "content": tpl})
    return files




def _ensure_routes_init(files: list) -> list:
    """Add routes/__init__.py if routes folder is used but init is missing."""
    paths = {f["path"] for f in files}
    if any(p.startswith("routes/") for p in paths) and "routes/__init__.py" not in paths:
        files.append({"path": "routes/__init__.py", "content": ""})
    return files




# --- Guarantee User class exists in models.py ---------------------------
def _ensure_user_model(files: list) -> list:
    """If models.py exists but lacks a User class, append a minimal one."""
    out = []
    for f in files:
        if f["path"] != "models.py":
            out.append(f)
            continue
        content = f["content"]
        try:
            tree = ast.parse(content)
            has_user = any(isinstance(n, ast.ClassDef) and n.name == "User" for n in tree.body)
        except Exception:
            has_user = "class User" in content
        if has_user:
            out.append(f)
            continue
        MIN_USER = "\n\n# Auto-added User model (auth requires it)\n" \
                   "from sqlalchemy import Column, Integer, String, DateTime, func\n\n\n" \
                   "class User(Base):\n" \
                   "    __tablename__ = \"users\"\n" \
                   "    id = Column(Integer, primary_key=True, index=True)\n" \
                   "    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)\n" \
                   "    email = Column(String(255), unique=True, nullable=False, index=True)\n" \
                   "    hashed_password = Column(String(255), nullable=False)\n"
        content = content.rstrip() + MIN_USER
        out.append({"path": f["path"], "content": content})
    return out




# --- Strip "templates/" prefix in Jinja extends -----------------------
import re as _re_tplext
_TE = _re_tplext.compile(r'\{%\s*extends\s+"templates/([^"]+)"\s*%\}')


def _strip_template_prefix(files: list) -> list:
    """Rewrite {% extends "templates/base.html" %} to {% extends "base.html" %}."""
    out = []
    for f in files:
        if not f["path"].endswith(".html"):
            out.append(f)
            continue
        new_c = _TE.sub(lambda m: '{% extends "' + m.group(1) + '" %}', f["content"])
        if new_c != f["content"]:
            out.append({"path": f["path"], "content": new_c})
        else:
            out.append(f)
    return out




# --- Fix models.py: remove self-defined Base, import from database -------
import re as _re_base


def _fix_models_base(files: list) -> list:
    """If models.py defines its own Base class, replace with import from database."""
    out = []
    for f in files:
        if f["path"] != "models.py":
            out.append(f)
            continue
        content = f["content"]
        orig = content

        # Pattern 1: class Base(DeclarativeBase): ...
        # Pattern 2: class Base(Base): ...
        # Pattern 3: Base = declarative_base()
        patterns = [
            (r"^[ \t]*class\s+Base\s*\(\s*DeclarativeBase\s*\)\s*:\s*\n(?:[ \t]+.*\n)*", ""),
            (r"^[ \t]*class\s+Base\s*\(\s*Base\s*\)\s*:\s*\n(?:[ \t]+.*\n)*", ""),
            (r"^[ \t]*Base\s*=\s*declarative_base\s*\(\s*\)\s*\n", ""),
        ]
        removed = False
        for pat, repl in patterns:
            new_c = _re_base.sub(pat, repl, content, flags=_re_base.MULTILINE)
            if new_c != content:
                content = new_c
                removed = True

        if not removed:
            out.append(f)
            continue

        # Ensure import from database is present
        if "from database import" in content and "Base" in content.split("from database import")[1].split("\n")[0]:
            pass  # already importing
        else:
            # Insert import right after __future__ or docstring
            lines = content.splitlines(keepends=True)
            insert_at = 0
            for i, ln in enumerate(lines[:10]):
                if ln.startswith("from __future__"):
                    insert_at = i + 1
                    break
                if ln.strip().startswith('"""') and i == 0:
                    # Skip docstring
                    for j in range(i + 1, min(len(lines), 30)):
                        if '"""' in lines[j]:
                            insert_at = j + 1
                            break
                    break
            content = "".join(lines[:insert_at]) + "from database import Base\n" + "".join(lines[insert_at:])

        out.append({"path": f["path"], "content": content})
    return out


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

    # Drain queued routes files
    for _rf in _ROUTES_APPENDED:
        if _rf["path"] not in {f["path"] for f in files_out}:
            files_out.append(_rf)
    _ROUTES_APPENDED.clear()

    files_out = _ensure_routes_init(files_out)
    files_out = _strip_template_prefix(files_out)
    files_out = _strip_relative_imports(files_out)
    files_out = _fix_base_template(files_out)
    files_out = _fix_sqlalchemy_relationships(files_out)
    files_out = _fix_models_base(files_out)
    files_out = _fix_auth_calls(files_out)
    files_out = _fix_template_config(files_out)
    files_out = _fix_pydantic_kwargs_files(files_out)
    files_out, _gr_ch = _ensure_req(files_out)
    if _gr_ch:
        log.info("goodreq_changes", changes=_gr_ch)
    files_out, _pf_ch = _preflight_fix(files_out)
    if _pf_ch:
        log.info("preflight_changes", changes=_pf_ch)
    files_out = _add_missing_frozen_files(files_out)
    files_out = _ensure_user_model(files_out)
    files_out = await _ensure_planned_files(files_out, plan, log)
    files_out, _cl_ch = _fix_clash(files_out)
    if _cl_ch:
        log.info("clashfix_changes", changes=_cl_ch)
    files_out, _imp_ch = _fix_missing_imports(files_out)
    if _imp_ch:
        log.info("autoimport_changes", changes=_imp_ch)
    files_out = await _cross_file_repair_loop(files_out, log)
    files_out = _ensure_static_dir(files_out)
    files_out = _repair_template_response(files_out)
    files_out = _cross_file_alias_repair(files_out)
    return {
        "files": files_out,
        "rag_tokens_used": total_rag_tokens,
        "failed_files": failed,
    }