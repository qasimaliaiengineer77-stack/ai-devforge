"""
Reset admin password (dev helper).
Usage: python -m scripts.reset_admin_password
"""
import getpass
import sys
from sqlalchemy import select

from auth.passwords import hash_password
from db.database import SessionLocal
from db.models import User


def main() -> int:
    username = input("Username to reset [admin]: ").strip() or "admin"
    pw = getpass.getpass("New password (min 8 chars): ")
    if len(pw) < 8:
        print("ERROR: password must be at least 8 characters.")
        return 1
    confirm = getpass.getpass("Confirm new password: ")
    if pw != confirm:
        print("ERROR: passwords do not match.")
        return 1

    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
        if user is None:
            print(f"ERROR: user '{username}' not found.")
            return 2
        user.password_hash = hash_password(pw)
        db.commit()
        print(f"OK: password reset for user '{username}'.")
        return 0
    except Exception as exc:
        db.rollback()
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())