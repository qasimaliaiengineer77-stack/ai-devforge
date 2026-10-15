"""Extract exported names AND function signatures from frozen templates."""

from __future__ import annotations

import ast
from pathlib import Path
from functools import lru_cache

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _fmt_arg(a: ast.arg, default=None) -> str:
    ann = ""
    if a.annotation is not None:
        try:
            ann = ": " + ast.unparse(a.annotation)
        except Exception:
            ann = ""
    d = ""
    if default is not None:
        try:
            d = " = " + ast.unparse(default)
        except Exception:
            d = " = ..."
    return f"{a.arg}{ann}{d}"


def _fmt_returns(fn) -> str:
    if fn.returns is None:
        return ""
    try:
        return " -> " + ast.unparse(fn.returns)
    except Exception:
        return ""


def _fmt_function(fn) -> str:
    args = []
    total = len(fn.args.args)
    defaults = fn.args.defaults
    offset = total - len(defaults)
    for i, a in enumerate(fn.args.args):
        default = defaults[i - offset] if i >= offset else None
        args.append(_fmt_arg(a, default))
    return f"{fn.name}({', '.join(args)}){_fmt_returns(fn)}"


def _extract(content: str) -> dict:
    try:
        tree = ast.parse(content)
    except Exception:
        return {"names": [], "functions": [], "classes": [], "assigns": []}
    names, funcs, classes, assigns = [], [], [], []
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef):
            funcs.append(_fmt_function(node))
            names.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            funcs.append(_fmt_function(node))
            names.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                    assigns.append(t.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)
                assigns.append(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for al in node.names:
                names.append(al.asname or al.name)
    return {
        "names": sorted(set(names)),
        "functions": funcs,
        "classes": sorted(set(classes)),
        "assigns": sorted(set(assigns)),
    }


@lru_cache(maxsize=1)
def all_template_exports() -> dict:
    result = {}
    if not _TEMPLATES_DIR.exists():
        return result
    for f in _TEMPLATES_DIR.glob("*.py"):
        if f.name.startswith("_"):
            continue
        result[f.stem] = _extract(f.read_text(encoding="utf-8"))
    return result


def format_for_prompt() -> str:
    """Return frozen modules with SIGNATURES so the AI can call them correctly."""
    ex = all_template_exports()
    if not ex:
        return ""
    lines = ["FROZEN MODULES (already provided; import ONLY these names; use EXACT signatures):"]
    for mod, info in ex.items():
        lines.append(f"\n[{mod}.py]")
        lines.append(f"  Module-level names (variables/classes/imports):")
        lines.append(f"    {', '.join(info['names']) or '(none)'}")
        if info["functions"]:
            lines.append(f"  Functions (call with these exact args):")
            for fn in info["functions"]:
                lines.append(f"    {fn}")
        if info["classes"]:
            lines.append(f"  Classes: {', '.join(info['classes'])}")
    lines.append("")
    lines.append("RULE: If you call a frozen function, you MUST pass all required positional args.")
    lines.append("RULE: Do NOT redefine any name listed above.")
    return "\n".join(lines)
