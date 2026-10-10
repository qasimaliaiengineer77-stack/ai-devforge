"""
AI Software Company — Seed first Admin (Step 3, F14 fix)

Usage:
  python -m scripts.seed_admin

Behavior:
  - If no admin exists, creates one.
  - Reads ADMIN_USERNAME and ADMIN_PASSWORD from .env (if set).
  - Otherwise, prompts interactively (password input hidden).

Security:
  - Password is hashed with bcrypt before storing.
  - Plaintext password is NEVER logged or stored.
  - If admin already exists, exits without changing anything.
"""

import getpass
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import select

from auth.passwords import hash_password
from db.audit import ACTION_SEED_RUN, ACTION_USER_CREATED, log_action
from db.database import SessionLocal, init_db
from db.models import User


def _create_admin(username: str, password: str) -> int:
    """Create admin user. Returns user_id. Raises if username exists."""
    init_db()
    db = SessionLocal()
    try:
        existing = db.execute(
            select(User).where(User.username == username)
        ).scalar_one_or_none()
        if existing is not None:
            print(f"[seed_admin] User '{username}' already exists. Nothing to do.")
            log_action(
                db,
                ACTION_SEED_RUN,
                status="skipped",
                message=f"user {username} already exists",
            )
            return existing.id

        admin = User(
            username=username,
            password_hash=hash_password(password),
            role="admin",
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)

        log_action(
            db,
            ACTION_USER_CREATED,
            status="ok",
            user_id=admin.id,
            message=f"admin '{username}' created via seed script",
        )
        log_action(
            db,
            ACTION_SEED_RUN,
            status="ok",
            user_id=admin.id,
            message="seed_admin created admin",
        )
        print(f"[seed_admin] Admin '{username}' created (user_id={admin.id}).")
        return admin.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> int:
    load_dotenv()

    username = os.getenv("ADMIN_USERNAME") or input("Admin username: ").strip()
    if not username:
        print("[seed_admin] ERROR: username cannot be empty.")
        return 1

    # Prefer env var (for scripted use), else prompt hidden.
    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Admin password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("[seed_admin] ERROR: passwords do not match.")
            return 1

    if len(password) < 8:
        print("[seed_admin] ERROR: password must be at least 8 characters.")
        return 1

    try:
        _create_admin(username, password)
    except Exception as exc:
        # Do not print password or hash. Just the error class.
        print(f"[seed_admin] FAILED: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())