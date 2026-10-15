"""
preflight.py - import-time RuntimeError ki aam wajahein pehle hi theek kar deta hai.
files = [{"path": ..., "content": ...}, ...]   (tumhare Coder ka format)

    from preflight import preflight_fix
    files_out, changes = preflight_fix(files_out)
"""
import re

BASE_REQ = {"fastapi", "uvicorn", "sqlalchemy", "httpx"}
TRIGGERS = [
    ("python-multipart", r"\bForm\(|\bUploadFile\b|\bFile\("),
    ("itsdangerous", r"SessionMiddleware"),
    ("email-validator", r"\bEmailStr\b"),
    ("jinja2", r"Jinja2Templates"),
]


def _norm(line: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)", line)
    return m.group(1).lower().replace("_", "-") if m else ""


def preflight_fix(files: list):
    changes = []
    out = list(files)
    py_text = "\n".join(f["content"] for f in out if f["path"].endswith(".py"))

    need = set(BASE_REQ)
    for pkg, pat in TRIGGERS:
        if re.search(pat, py_text):
            need.add(pkg)

    idx = next((i for i, f in enumerate(out) if f["path"] == "requirements.txt"), None)
    if idx is None:
        out.append({"path": "requirements.txt", "content": "\n".join(sorted(need)) + "\n"})
        changes.append("requirements.txt created")
    else:
        have = {_norm(l) for l in out[idx]["content"].splitlines()}
        missing = sorted(p for p in need if p not in have)
        if missing:
            content = out[idx]["content"].rstrip("\n") + "\n" + "\n".join(missing) + "\n"
            out[idx] = {**out[idx], "content": content}
            changes.append("requirements += " + ", ".join(missing))

    existing = {f["path"] for f in out}
    dirs = set(re.findall(r"StaticFiles\([^)]*directory\s*=\s*[\"']([^\"']+)[\"']", py_text))
    for d in dirs:
        d = d.strip("/\\")
        if d and not any(p.startswith(d + "/") for p in existing):
            out.append({"path": f"{d}/.gitkeep", "content": "# keep\n"})
            changes.append(f"{d}/.gitkeep added")
    return out, changes
