"""goodreq.py - agar requirements.txt plan mein nahi, known_good se seed karo."""
from pathlib import Path

_KG = Path(__file__).resolve().parent / "known_good" / "requirements.txt"


def ensure_requirements(files: list):
    if any(f["path"] == "requirements.txt" for f in files) or not _KG.exists():
        return files, []
    return files + [{"path": "requirements.txt", "content": _KG.read_text(encoding="utf-8")}], \
        ["requirements.txt seeded from known_good"]
