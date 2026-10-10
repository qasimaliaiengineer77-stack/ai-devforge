"""
AI Software Company — Token-based chunker (Step 10)

Architecture references:
  Section 14  RAG: chunk size 200 tokens, overlap 0
  F20         RAG chunk 200 tokens (no overlap)
  F21         Coder RAG cap 1500 tokens

Design:
  - Pure Python, uses tiktoken for accurate token counting.
  - Splits a long text into consecutive 200-token windows with NO overlap.
  - If the text is shorter than chunk_size, returns a single chunk.
  - Preserves word boundaries when possible (splits on whitespace between
    tokens, not inside words).
  - Never raises on empty input: returns [].
"""

from __future__ import annotations

from typing import Iterable

import tiktoken

# Default encoder for cl100k_base (used by modern OpenAI + Groq llama family
# for tokenization approximation). It counts tokens well enough for chunking.
_ENCODER = tiktoken.get_encoding("cl100k_base")

# --- Config (matches Section 14 + F20) -----------------------------------
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 0


def count_tokens(text: str) -> int:
    """Return token count for a string. Returns 0 for empty/None."""
    if not text:
        return 0
    return len(_ENCODER.encode(text, disallowed_special=()))


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """
    Split `text` into token-based chunks.

    - chunk_size: tokens per chunk (default 200)
    - overlap:    tokens shared between consecutive chunks (default 0, per F20)

    Returns a list of strings. Empty input -> [].

    Strategy: encode the whole text to token ids, walk in windows of
    `chunk_size` with step `chunk_size - overlap`, and decode each window.
    This is faster and more accurate than splitting on characters.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    tokens = _ENCODER.encode(text, disallowed_special=())
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        window = tokens[start:end]
        if not window:
            break
        chunk = _ENCODER.decode(window)
        # Skip whitespace-only chunks
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(tokens):
            break

    return chunks


def chunk_many(
    texts: Iterable[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Apply chunk_text to each item in an iterable and flatten."""
    out: list[str] = []
    for t in texts:
        out.extend(chunk_text(t, chunk_size=chunk_size, overlap=overlap))
    return out