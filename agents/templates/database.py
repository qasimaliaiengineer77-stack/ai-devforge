"""Frozen database module (template). Provided by platform, not AI."""

import os
import shutil
import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()
Base.__allow_unmapped__ = True  # allow non-Mapped annotations (SQLAlchemy 2.0)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def backup_db(backup_dir: str = "./backups") -> Path:
    if not DATABASE_URL.startswith("sqlite"):
        raise RuntimeError("backup_db is only implemented for SQLite databases")
    source_path = Path(DATABASE_URL.split("///")[-1])
    if not source_path.is_file():
        raise FileNotFoundError(f"SQLite database file not found at {source_path}")
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = backup_path / f"app_{timestamp}.db"
    shutil.copy2(source_path, backup_file)
    return backup_file
