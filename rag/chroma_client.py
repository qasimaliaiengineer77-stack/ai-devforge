"""
AI Software Company — ChromaDB HTTP client wrapper (Step 10)

Architecture references:
  Section 14  RAG architecture (ChromaDB, all-MiniLM-L6-v2, Top-K 5, min 0.5)
  F20         Chunk size: 200 tokens, no overlap
  F21         Coder RAG cap: max 1500 tokens per request

Design (minimal correction, documented in PROJECT_STATUS.md):
  - We run ChromaDB as a Docker container (chromadb/chroma:0.5.23) on the
    shared network ai_company_network, and connect via HTTP client from the
    host. This replaces embedding chromadb into the venv, which required
    C++ Build Tools on Windows + Python 3.12.
  - Embeddings are computed client-side by FastEmbed using the exact model
    from the architecture (all-MiniLM-L6-v2, ONNX runtime). The server
    stores the resulting vectors.

Public API:
  get_client() -> chromadb.HttpClient
  get_or_create_collection(name) -> Collection
  is_available() -> bool
  reset_client() -> None
  COLLECTIONS -> list of the 5 collection names (Section 14)
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv

from rag.embedding_fn import get_embedding_function

load_dotenv()

# --- Config --------------------------------------------------------------
# Default matches the docker run command documented in PROJECT_STATUS.md.
CHROMA_URL = os.getenv("CHROMA_URL", "http://127.0.0.1:8888")

# --- 5 collections (Section 14) ------------------------------------------
COLLECTIONS = [
    "fastapi_docs",
    "jinja2_docs",
    "tailwind_docs",
    "sqlalchemy_2_docs",
    "pydantic_2_docs",
]

# --- Lazy singleton ------------------------------------------------------
_client = None


def _parse_chroma_url(url: str) -> tuple[str, int, bool]:
    """Return (host, port, ssl) parsed from a URL like http://127.0.0.1:8888."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    ssl = parsed.scheme == "https"
    return host, port, ssl


def get_client():
    """
    Return a shared chromadb.HttpClient connected to the Chroma server.
    Raises RuntimeError if chromadb-client is not installed or server is down.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb  # provided by chromadb-client
    except Exception as exc:
        raise RuntimeError(
            "chromadb-client is not installed. Run: pip install -r requirements.txt"
        ) from exc

    host, port, ssl = _parse_chroma_url(CHROMA_URL)

    try:
        _client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
        # Trigger a real call so we fail fast if the server is unreachable.
        _client.heartbeat()
    except Exception as exc:
        _client = None
        raise RuntimeError(
            f"cannot connect to Chroma server at {CHROMA_URL}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    return _client


def is_available() -> bool:
    """Return True if the Chroma server responds to heartbeat."""
    try:
        get_client()
        return True
    except Exception:
        return False


def get_or_create_collection(name: str):
    """
    Return the named collection, creating it if it does not exist.
    The FastEmbed embedding function is attached so client-side embeddings
    are computed automatically when documents are added or queried.
    """
    if name not in COLLECTIONS:
        raise ValueError(f"unknown collection: {name!r} (allowed: {COLLECTIONS})")
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        embedding_function=get_embedding_function(),
    )


def reset_client() -> None:
    """Forget the cached client. Used in tests."""
    global _client
    _client = None