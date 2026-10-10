"""
AI Software Company — Password hashing (Step 3)

Architecture rules:
  - passlib + bcrypt (bcrypt==4.0.1 pinned per Section 4 / F7)
  - Never log or return plaintext passwords
  - Never commit passwords (secrets stay in .env or DB as hash only)
"""

from passlib.context import CryptContext

# bcrypt==4.0.1 is pinned because newer bcrypt breaks passlib 1.7.4.
# Do not upgrade bcrypt without testing passlib compatibility.
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
)


def hash_password(plain: str) -> str:
    """Hash a plaintext password. Returns a bcrypt hash string."""
    if not isinstance(plain, str) or not plain:
        raise ValueError("Password must be a non-empty string")
    # bcrypt max input is 72 bytes; truncate safely to avoid ValueError.
    if len(plain.encode("utf-8")) > 72:
        plain = plain.encode("utf-8")[:72].decode("utf-8", errors="ignore")
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored hash. Never raises."""
    if not plain or not hashed:
        return False
    try:
        if len(plain.encode("utf-8")) > 72:
            plain = plain.encode("utf-8")[:72].decode("utf-8", errors="ignore")
        return pwd_context.verify(plain, hashed)
    except Exception:
        # Do not leak details; treat any error as failed verification.
        return False