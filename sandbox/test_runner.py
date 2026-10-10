"""
AI Software Company — Sandbox test runner (Step 12)

Runs INSIDE the Docker sandbox (no network) against the generated app.
Performs 3 levels of checks:

  L1: Static security checks (Section 9, 10 checks)
  L2: ast.parse on every .py file (syntax)
  L3: FastAPI TestClient HTTP tests (5 checks)

Output: prints JSON as the LAST line of stdout so the host can parse it:
  {"l1_ok": bool, "l1_issues": [...],
   "l2_ok": bool, "l2_issues": [...],
   "l3_ok": bool, "l3_issues": [...],
   "tests_run": int, "tests_passed": int}

Exit code: 0 if all levels pass, 1 otherwise.
"""

import ast
import importlib.util
import json
import os
import re
import sys
from pathlib import Path


APP_DIR = Path("/app")
ISSUES_L1 = []
ISSUES_L2 = []
ISSUES_L3 = []


# ---------- L1: static security checks (Section 9) -----------------------
def _read_all_text() -> dict:
    """Read every text file in the app dir. Returns {relpath: content}."""
    out = {}
    for p in APP_DIR.rglob("*"):
        if not p.is_file():
            continue
        if p.name in {"test_runner.py"}:
            continue
        try:
            out[str(p.relative_to(APP_DIR)).replace("\\", "/")] = p.read_text(
                encoding="utf-8", errors="replace"
            )
        except Exception:
            continue
    return out


def l1_static_checks(files: dict) -> bool:
    joined = "\n".join(files.values())

    # 1. passlib + bcrypt
    if "passlib" not in joined or "bcrypt" not in joined:
        ISSUES_L1.append("check1_missing_passlib_bcrypt")

    # 2. HttpOnly cookie (set_cookie / httponly)
    if not re.search(r"httponly\s*=\s*True", joined, re.IGNORECASE):
        ISSUES_L1.append("check2_missing_httponly")

    # 3. os.getenv for secrets (at least one getenv usage)
    if "os.getenv" not in joined and "os.environ" not in joined:
        ISSUES_L1.append("check3_missing_os_getenv")

    # 4. Protected routes (at least one Depends requiring a user)
    if "Depends" not in joined:
        ISSUES_L1.append("check4_missing_depends")

    # 5. Pydantic validation (BaseModel)
    if "BaseModel" not in joined and "pydantic" not in joined:
        ISSUES_L1.append("check5_missing_pydantic")

    # 6. SQLAlchemy ORM
    if "sqlalchemy" not in joined.lower():
        ISSUES_L1.append("check6_missing_sqlalchemy")

    # 7. backup_db function
    if "backup_db" not in joined:
        ISSUES_L1.append("check7_missing_backup_db")

    # 8. Startup event / lifespan (create_all)
    if "create_all" not in joined:
        ISSUES_L1.append("check8_missing_create_all")

    # 9. every table has id + created_at — grep for created_at column
    if "created_at" not in joined:
        ISSUES_L1.append("check9_missing_created_at")

    # 10. CSRF token presence
    if "csrf" not in joined.lower():
        ISSUES_L1.append("check10_missing_csrf")

    return len(ISSUES_L1) == 0


# ---------- L2: syntax (ast.parse) ---------------------------------------
def l2_syntax_checks(files: dict) -> bool:
    any_py = False
    for rel, content in files.items():
        if not rel.endswith(".py"):
            continue
        any_py = True
        try:
            ast.parse(content)
        except SyntaxError as exc:
            ISSUES_L2.append(f"{rel}:syntax:{exc.lineno}:{exc.msg}")
        except Exception as exc:
            ISSUES_L2.append(f"{rel}:ast_error:{type(exc).__name__}")
    if not any_py:
        ISSUES_L2.append("no_python_files")
    return len(ISSUES_L2) == 0


# ---------- L3: HTTP tests via TestClient --------------------------------
def l3_http_tests() -> bool:
    """
    Import the app from main.py and hit 5 endpoints.
    Runs from /app so `import main` works.
    """
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))

    # Ensure a safe DATABASE_URL for the sandbox
    os.environ.setdefault("DATABASE_URL", "sqlite:///./sandbox_test.db")

    # Locate main.py
    main_path = APP_DIR / "main.py"
    if not main_path.exists():
        ISSUES_L3.append("main.py_not_found")
        return False

    try:
        spec = importlib.util.spec_from_file_location("main", main_path)
        main_mod = importlib.util.module_from_spec(spec)
        sys.modules["main"] = main_mod
        spec.loader.exec_module(main_mod)
    except Exception as exc:
        ISSUES_L3.append(f"import_main_failed:{type(exc).__name__}:{str(exc)[:120]}")
        return False

    app = getattr(main_mod, "app", None)
    if app is None:
        ISSUES_L3.append("no_app_object_in_main")
        return False

    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        ISSUES_L3.append(f"testclient_import_failed:{type(exc).__name__}")
        return False

    client = TestClient(app)

    def _hit(method, path, expect=None, data=None):
        try:
            if method == "GET":
                r = client.get(path)
            else:
                r = client.post(path, data=data or {})
        except Exception as exc:
            ISSUES_L3.append(f"{method} {path} raised {type(exc).__name__}: {str(exc)[:120]}")
            return None
        if expect is not None and r.status_code not in expect:
            ISSUES_L3.append(f"{method} {path} -> {r.status_code} (expected {expect})")
        return r

    # 5 tests
    _hit("GET", "/", expect=(200, 404))                # 1
    _hit("GET", "/health", expect=(200, 404))          # 2
    _hit("GET", "/login", expect=(200, 404, 405))      # 3
    _hit("GET", "/signup", expect=(200, 404, 405))     # 4
    _hit("POST", "/login", expect=(200, 400, 401, 403, 404, 422),
         data={"username": "x", "password": "y"})      # 5

    return len(ISSUES_L3) == 0


# ---------- main ---------------------------------------------------------
def main() -> int:
    files = _read_all_text()
    if not files:
        print(json.dumps({
            "l1_ok": False, "l1_issues": ["no_files"],
            "l2_ok": False, "l2_issues": [],
            "l3_ok": False, "l3_issues": [],
            "tests_run": 0, "tests_passed": 0,
        }))
        return 1

    l1_ok = l1_static_checks(files)
    l2_ok = l2_syntax_checks(files)
    l3_ok = l3_http_tests()

    tests_run = 5
    tests_passed = tests_run - len(ISSUES_L3)

    report = {
        "l1_ok": l1_ok,
        "l1_issues": ISSUES_L1,
        "l2_ok": l2_ok,
        "l2_issues": ISSUES_L2,
        "l3_ok": l3_ok,
        "l3_issues": ISSUES_L3,
        "tests_run": tests_run,
        "tests_passed": tests_passed,
    }
    print(json.dumps(report))
    return 0 if (l1_ok and l2_ok and l3_ok) else 1


if __name__ == "__main__":
    sys.exit(main())