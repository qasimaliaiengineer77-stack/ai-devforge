"""
autoimport.py (v2) - jo naam use hua magar import nahi hua, uska import khud jodta hai.
Frozen files (main.py) par bhi chalta hai. Ab dict mein naya naam jodne ki zaroorat nahi:

  1. COMMON dict (known naam, sahi module)
  2. Project ki doosri files (jaise Recipe -> from models import Recipe)
  3. Libraries mein khud dhoondta hai (sqlalchemy, sqlalchemy.orm, fastapi, ...)
"""
import ast
import builtins
import importlib
import inspect
import re

LIBS = ["sqlalchemy", "sqlalchemy.orm", "fastapi", "fastapi.responses", "typing", "datetime", "pydantic"]

COMMON = {
    "func": "from sqlalchemy import func", "select": "from sqlalchemy import select",
    "update": "from sqlalchemy import update", "delete": "from sqlalchemy import delete",
    "insert": "from sqlalchemy import insert", "and_": "from sqlalchemy import and_",
    "or_": "from sqlalchemy import or_", "text": "from sqlalchemy import text",
    "desc": "from sqlalchemy import desc", "asc": "from sqlalchemy import asc",
    "Column": "from sqlalchemy import Column", "Integer": "from sqlalchemy import Integer",
    "String": "from sqlalchemy import String", "Text": "from sqlalchemy import Text",
    "Float": "from sqlalchemy import Float", "Numeric": "from sqlalchemy import Numeric",
    "Boolean": "from sqlalchemy import Boolean", "DateTime": "from sqlalchemy import DateTime",
    "Date": "from sqlalchemy import Date", "ForeignKey": "from sqlalchemy import ForeignKey",
    "Session": "from sqlalchemy.orm import Session", "Mapped": "from sqlalchemy.orm import Mapped",
    "mapped_column": "from sqlalchemy.orm import mapped_column",
    "relationship": "from sqlalchemy.orm import relationship",
    "Depends": "from fastapi import Depends", "HTTPException": "from fastapi import HTTPException",
    "Request": "from fastapi import Request", "Form": "from fastapi import Form",
    "Query": "from fastapi import Query", "status": "from fastapi import status",
    "APIRouter": "from fastapi import APIRouter",
    "RedirectResponse": "from fastapi.responses import RedirectResponse",
    "HTMLResponse": "from fastapi.responses import HTMLResponse",
    "JSONResponse": "from fastapi.responses import JSONResponse",
    "timedelta": "from datetime import timedelta", "timezone": "from datetime import timezone",
    "date": "from datetime import date", "Optional": "from typing import Optional",
    "List": "from typing import List", "Dict": "from typing import Dict", "Any": "from typing import Any",
    "Decimal": "from decimal import Decimal",
    "secrets": "import secrets", "os": "import os", "uuid": "import uuid",
    "json": "import json", "re": "import re", "math": "import math",
}

_NEVER_SOURCE = {"main"}
_mod_cache = {}


def _bound_names(tree) -> set:
    b = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b.add(n.name)
        elif isinstance(n, ast.arg):
            b.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            b.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                b.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            b.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            b.update(n.names)
    return b


def _datetime_import(src: str) -> str:
    return "import datetime" if re.search(r"\bdatetime\.(datetime|date|timedelta|timezone)\b", src) \
        else "from datetime import datetime"


def _project_owners(files: list) -> dict:
    owners = {}
    for f in files:
        if not f["path"].endswith(".py"):
            continue
        try:
            tree = ast.parse(f["content"])
        except SyntaxError:
            continue
        mod = f["path"][:-3].replace("/", ".")
        for n in tree.body:
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                owners.setdefault(n.name, set()).add(mod)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        owners.setdefault(t.id, set()).add(mod)
    return owners


def _lib_import(name: str, libs: list):
    for m in libs:
        mod = _mod_cache.get(m, False)
        if mod is False:
            try:
                mod = importlib.import_module(m)
            except Exception:
                mod = None
            _mod_cache[m] = mod
        if mod is not None and hasattr(mod, name) and not inspect.ismodule(getattr(mod, name)):
            return f"from {m} import {name}"
    return None


def fix_missing_imports(files: list, libs: list = None):
    libs = LIBS if libs is None else libs
    owners = _project_owners(files)
    out, changes = [], []
    for f in files:
        if not f["path"].endswith(".py"):
            out.append(f)
            continue
        src = f["content"]
        try:
            tree = ast.parse(src)
        except SyntaxError:
            out.append(f)
            continue
        me = f["path"][:-3].replace("/", ".")
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        undefined = used - _bound_names(tree)

        lines = set()
        for name in sorted(undefined):
            if name == "datetime":
                lines.add(_datetime_import(src))
            elif name in COMMON:
                lines.add(COMMON[name])
            elif name.startswith("_"):
                continue
            else:
                srcs = owners.get(name, set()) - {me} - _NEVER_SOURCE
                if len(srcs) == 1:
                    lines.add(f"from {next(iter(srcs))} import {name}")
                else:
                    li = _lib_import(name, libs)
                    if li:
                        lines.add(li)
        if not lines:
            out.append(f)
            continue

        text = src.splitlines(keepends=True)
        at = 0
        for i, ln in enumerate(text[:10]):
            if ln.startswith("from __future__"):
                at = i + 1
        new_src = "".join(text[:at]) + "\n".join(sorted(lines)) + "\n" + "".join(text[at:])
        out.append({**f, "content": new_src})
        changes.append(f"{f['path']}: +{', '.join(sorted(lines))}")
    return out, changes
