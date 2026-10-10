"""
AI Software Company — Database setup (Step 3)
SQLAlchemy engine, session, and Base for models.

Rule (architecture Section 22 hardening):
  - commit only on success, rollback on exception
  - never commit unconditionally in finally
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_company.db")

# SQLite needs check_same_thread=False for FastAPI's threaded handlers.
# (WAL mode + timeout helps with concurrent access in later steps.)
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False, "timeout": 30}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """
    FastAPI dependency: yields a DB session, always closes it.
    Commit is NOT done here — caller commits on success only.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_transaction():
    """
    Transaction context manager for scripts (seed_admin, tests).
    Commit on success, rollback on exception — never commit in finally.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Create all tables. Safe to call multiple times."""
    # Import models so they register with Base.metadata
    from db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)