"""
Static checks that run BEFORE the Docker sandbox.
Catches ~80% of AI-generated errors without needing Docker.

Checks:
  1. ast.parse on every .py  (syntax)
  2. Local import validation  (from X import Y  ->  Y must exist in X)
  3. pyflakes                 (undefined names)

Returns: list of issue strings (empty = all pass)
"""

from __future__ import annotations

import ast
import io


def _all_py(files):
    return [f for f in files if f["path"].endswith(".py")]


def _parse(content):
    try:
        return ast.parse(content), None
    except SyntaxError as e:
        return None, f"syntax:{e.lineno}:{e.msg}"


def _check_syntax(files):
    issues = []
    for f in _all_py(files):
        _, err = _parse(f["content"])
        if err:
            issues.append(f"{f['path']}:{err}")
    return issues


def _top_names(content):
    tree, _ = _parse(content)
    if tree is None:
        return set()
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
    return out


def _module_exports(files):
    exports = {}
    for f in _all_py(files):
        p = f["path"]
        if not p.endswith(".py"):
            continue
        mod = p[:-3]
        names = _top_names(f["content"])
        exports[mod] = names
        exports.setdefault(mod.split("/")[-1], names)
    return exports


def _check_local_imports(files):
    exports = _module_exports(files)
    issues = []
    for f in _all_py(files):
        tree, _ = _parse(f["content"])
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level or not node.module:
                continue
            mod = node.module
            if mod not in exports:
                continue
            avail = exports[mod]
            for al in node.names:
                if al.name == "*":
                    continue
                if al.name not in avail:
                    issues.append(f"{f['path']}:missing:{mod}.{al.name}")
    return issues


def _check_pyflakes(files):
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError:
        return []
    issues = []
    for f in _all_py(files):
        buf = io.StringIO()
        try:
            check(f["content"], filename=f["path"], reporter=Reporter(buf, buf))
        except Exception:
            continue
        for line in buf.getvalue().splitlines():
            low = line.lower()
            if "undefined name" in low:
                issues.append(f"{f['path']}:{line.strip()}")
    return issues


def run_all(files):
    """Run every static check. Returns list of issue strings."""
    if not files:
        return ["no_files"]
    issues = []
    issues.extend(_check_syntax(files))
    issues.extend(_check_local_imports(files))
    issues.extend(_check_pyflakes(files))
    return issues
