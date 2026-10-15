"""
clashfix.py - field ka naam aur uski type ka naam same ho (date: Optional[date] = None)
to Pydantic/SQLAlchemy error deta hai. Field ka naam BADALTE NAHI (routes/templates/DB toot jayenge),
balki class ke andar type ko alias deta hai: date -> _date.

    from clashfix import fix_field_type_clash
    files_out, changes = fix_field_type_clash(files_out)
"""
import ast


def _top_imports(tree):
    m = {}
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            for a in n.names:
                if a.asname is None and a.name != "*":
                    m[a.name] = f"from {n.module} import {a.name} as _{a.name}"
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.asname is None and "." not in a.name:
                    m[a.name] = f"import {a.name} as _{a.name}"
    return m


def fix_field_type_clash(files: list):
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
        imports = _top_imports(tree)
        edits, needed = [], set()

        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            fields = [s for s in cls.body if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
            field_names = {s.target.id for s in fields}
            for s in fields:
                for n in ast.walk(s.annotation):
                    if isinstance(n, ast.Name) and n.id in field_names and n.id in imports:
                        edits.append((n.lineno, n.col_offset, n.end_col_offset, n.id))
                        needed.add(n.id)

        if not edits:
            out.append(f)
            continue

        lines = src.splitlines(keepends=True)
        for lineno, col, end, name in sorted(set(edits), reverse=True):
            b = lines[lineno - 1].encode("utf-8")
            if b[col:end].decode("utf-8", "ignore") == name:
                lines[lineno - 1] = (b[:col] + f"_{name}".encode() + b[end:]).decode("utf-8")

        last = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
        add = "".join(imports[n] + "\n" for n in sorted(needed))
        lines.insert(last, add)
        out.append({**f, "content": "".join(lines)})
        changes.append(f"{f['path']}: type alias for {', '.join(sorted(needed))}")
    return out, changes
