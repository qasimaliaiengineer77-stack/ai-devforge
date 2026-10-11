"""
Reset or create the two test users used for Step 17 smoke tests.

- user1     (role=user)      — a normal user who submits builds
- approver1 (role=approver)  — the Approval Authority who decides

Passwords are entered interactively and never logged or printed.
"""
import getpass
import sys

from sqlalchemy import select

from auth.passwords import hash_password
from db.database import SessionLocal, init_db
from db.models import User


def _ensure(db, username, password, role):
    u = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if u is None:
        u = User(username=username, password_hash=hash_password(password), role=role)
        db.add(u)
        db.commit()
        db.refresh(u)
        print(f"  created {username} (id={u.id}, role={u.role})")
    else:
        u.password_hash = hash_password(password)
        u.role = role
        db.commit()
        print(f"  updated {username} (id={u.id}, role={u.role})")


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        print("Reset password for user1 (role=user):")
        p1 = getpass.getpass("  new password (min 8 chars): ")
        if len(p1) < 8:
            print("  ERROR: too short")
            return 1
        p1c = getpass.getpass("  confirm: ")
        if p1 != p1c:
            print("  ERROR: mismatch")
            return 1
        _ensure(db, "user1", p1, "user")

        print()
        print("Reset password for approver1 (role=approver):")
        p2 = getpass.getpass("  new password (min 8 chars): ")
        if len(p2) < 8:
            print("  ERROR: too short")
            return 1
        p2c = getpass.getpass("  confirm: ")
        if p2 != p2c:
            print("  ERROR: mismatch")
            return 1
        _ensure(db, "approver1", p2, "approver")
    finally:
        db.close()
    print()
    print("Done. Passwords are not stored anywhere except as bcrypt hashes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())