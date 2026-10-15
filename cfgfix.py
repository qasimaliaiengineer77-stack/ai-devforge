"""cfgfix.py - Pydantic v1 class Config -> v2 model_config = ConfigDict(...)"""
import re


def fix_config_class(files: list):
    out, changes = [], []
    for f in files:
        if not f["path"].endswith(".py"):
            out.append(f)
            continue
        src = f["content"]
        if "class Config" not in src or "ConfigDict" in src:
            out.append(f)
            continue
        pattern = re.compile(
            r"^([ \t]*)class\s+Config\s*:\s*\n((?:[ \t]+.*\n)*)",
            re.MULTILINE,
        )

        def _repl(m):
            indent = m.group(1)
            body = m.group(2)
            kwargs = []
            for line in body.splitlines():
                s = line.strip()
                if "=" in s and not s.startswith("#"):
                    k, _, v = s.partition("=")
                    kwargs.append(f"{k.strip()}={v.strip()}")
            if not kwargs:
                return ""
            return f"{indent}model_config = ConfigDict({', '.join(kwargs)})\n"

        new_src = pattern.sub(_repl, src)
        if new_src == src:
            out.append(f)
            continue
        head = new_src.split("class ", 1)[0]
        if "ConfigDict" not in head:
            new_src = re.sub(
                r"from pydantic import ([^\n]+)",
                lambda mm: mm.group(0) if "ConfigDict" in mm.group(1)
                else f"from pydantic import ConfigDict, {mm.group(1)}",
                new_src, count=1,
            )
        out.append({**f, "content": new_src})
        changes.append(f"{f['path']}: class Config -> model_config")
    return out, changes
