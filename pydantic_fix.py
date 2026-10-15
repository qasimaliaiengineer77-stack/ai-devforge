"""
pydantic_fix.py - Pydantic v1 ke removed keyword arguments ko v2 naam mein badalta hai.
Coder ke 6 auto-fixes ke saath, schemas.py (aur baaki .py files) generate hone ke baad chalao.

    from pydantic_fix import fix_pydantic_v1_kwargs
    new_source, changes = fix_pydantic_v1_kwargs(source)
"""
import ast

RENAMES = {"regex": "pattern", "min_items": "min_length", "max_items": "max_length"}
TARGET_CALLS = {"Field", "Query", "Path", "Body", "Form", "Header", "Cookie", "constr", "conlist"}


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")


def fix_pydantic_v1_kwargs(source: str):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []          # syntax toota hua ho to chhedo mat

    edits = []                     # (lineno, byte_col, old_name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in TARGET_CALLS:
            for kw in node.keywords:
                if kw.arg in RENAMES:
                    edits.append((kw.lineno, kw.col_offset, kw.arg))
    if not edits:
        return source, []

    lines = source.splitlines(keepends=True)
    changes = []
    # peeche se aage, taake ek hi line ke multiple edits ke offsets na bigdein
    for lineno, col, old in sorted(edits, reverse=True):
        b = lines[lineno - 1].encode("utf-8")     # col_offset bytes mein hota hai
        if b[col:col + len(old)].decode("utf-8", "ignore") == old:
            new = RENAMES[old]
            lines[lineno - 1] = (b[:col] + new.encode() + b[col + len(old):]).decode("utf-8")
            changes.append(f"line {lineno}: {old}= -> {new}=")
    return "".join(lines), changes[::-1]
