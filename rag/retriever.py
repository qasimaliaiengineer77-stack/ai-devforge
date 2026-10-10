"""
AI Software Company — RAG retriever (Step 10)

Architecture references:
  Section 14  RAG: Top-K 5, relevance min 0.5
  F20         Chunk size 200 tokens (see rag/chunker.py)
  F21         Coder RAG cap: max 1500 tokens per request
  Section 16  AppState field: rag_tokens_used

Purpose:
  - Given a set of "topics" (e.g., file types the Coder is about to write),
    fetch the most relevant chunks from the 5 ChromaDB collections.
  - Respect the per-request token cap (1500) so the Coder stays inside
    Groq's TPM budget.

Public API:
  retrieve_for_topics(topics, *, top_k=5, min_score=0.5, token_cap=1500)
      -> RetrievalResult
  COLLECTION_TOPICS  -> mapping keyword -> collection name
  format_for_prompt(chunks) -> str  (ready to drop into a prompt as <rag_context>)

Notes:
  - Chroma returns distances (smaller = more similar) for the default
    "l2" space and cosine distance for "cosine". We convert distance to a
    similarity score so 0.5 means "at least 50% similar".
  - We treat any chunk whose similarity < min_score as irrelevant and drop it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

from rag import chroma_client as cc
from rag.chunker import count_tokens

logger = logging.getLogger("rag.retriever")

# --- Topic -> collection mapping (Section 14) ----------------------------
COLLECTION_TOPICS = {
    "fastapi": "fastapi_docs",
    "route": "fastapi_docs",
    "endpoint": "fastapi_docs",
    "jinja2": "jinja2_docs",
    "template": "jinja2_docs",
    "html": "jinja2_docs",
    "tailwind": "tailwind_docs",
    "css": "tailwind_docs",
    "style": "tailwind_docs",
    "sqlalchemy": "sqlalchemy_2_docs",
    "model": "sqlalchemy_2_docs",
    "orm": "sqlalchemy_2_docs",
    "pydantic": "pydantic_2_docs",
    "schema": "pydantic_2_docs",
    "validation": "pydantic_2_docs",
}


# --- Result types --------------------------------------------------------
@dataclass
class RetrievedChunk:
    collection: str
    text: str
    score: float         # similarity in [0, 1]; higher = more similar
    distance: float      # raw Chroma distance
    tokens: int


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    rag_tokens_used: int = 0
    truncated: bool = False  # True if token cap forced us to drop chunks


# --- Distance -> similarity ----------------------------------------------
def _distance_to_score(distance: float) -> float:
    """
    Convert a Chroma distance into a similarity score in [0, 1].

    For cosine distance in [0, 2], similarity = 1 - distance/2, clamped to [0,1].
    For l2 distance (>= 0), we use 1/(1+distance) which is in (0, 1].
    We cannot always tell which space a collection uses, so we take the
    more forgiving of the two: for typical distances around 0-2 the results
    match closely enough for a 0.5 threshold.
    """
    try:
        d = float(distance)
    except Exception:
        return 0.0
    if d < 0:
        d = 0.0
    # Cosine-style
    cosine_sim = max(0.0, min(1.0, 1.0 - d / 2.0))
    # L2-style
    l2_sim = 1.0 / (1.0 + d)
    return max(cosine_sim, l2_sim)


# --- Topic -> collections ------------------------------------------------
def collections_for_topics(topics: Iterable[str]) -> list[str]:
    """
    Map a set of free-form topic words to collection names.
    Always returns at least one collection; defaults to fastapi_docs.
    """
    picked: list[str] = []
    for t in topics:
        key = str(t).strip().lower()
        if not key:
            continue
        col = COLLECTION_TOPICS.get(key)
        if col and col not in picked:
            picked.append(col)
    if not picked:
        picked.append("fastapi_docs")
    return picked


# --- Main entry point ----------------------------------------------------
def retrieve_for_topics(
    topics: Iterable[str],
    *,
    top_k: int = 5,
    min_score: float = 0.5,
    token_cap: int = 1500,
    query_text: Optional[str] = None,
) -> RetrievalResult:
    """
    Retrieve the most relevant chunks across the collections matched by
    `topics`, capped to `token_cap` tokens total.

    - top_k: number of chunks to ask Chroma for per collection (Section 14)
    - min_score: minimum similarity to keep a chunk (Section 14)
    - token_cap: hard cap on total tokens returned (F21)
    - query_text: optional custom query; if None, we join topics as the query
    """
    result = RetrievalResult()
    collections = collections_for_topics(topics)
    query = query_text or " ".join(str(t) for t in topics) or "web app"

    raw_hits: list[RetrievedChunk] = []

    for col_name in collections:
        try:
            col = cc.get_or_create_collection(col_name)
        except Exception as exc:
            logger.warning("retriever_collection_failed col=%s err=%s", col_name, exc)
            continue

        try:
            res = col.query(query_texts=[query], n_results=top_k)
        except Exception as exc:
            logger.warning("retriever_query_failed col=%s err=%s", col_name, exc)
            continue

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        if not ids:
            continue

        for i in range(len(ids)):
            text = docs[i] if i < len(docs) else ""
            dist = dists[i] if i < len(dists) else 0.0
            if not text or not text.strip():
                continue
            score = _distance_to_score(dist)
            if score < min_score:
                continue
            raw_hits.append(
                RetrievedChunk(
                    collection=col_name,
                    text=text,
                    score=score,
                    distance=float(dist),
                    tokens=count_tokens(text),
                )
            )

    # Sort by score descending (best first), then apply the token cap.
    raw_hits.sort(key=lambda c: c.score, reverse=True)

    used = 0
    for hit in raw_hits:
        if used + hit.tokens > token_cap:
            result.truncated = True
            continue
        result.chunks.append(hit)
        used += hit.tokens

    result.rag_tokens_used = used
    return result


# --- Prompt formatting ---------------------------------------------------
def format_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Return a single string with each chunk separated by a blank line,
    suitable for embedding inside <rag_context>...</rag_context>.
    """
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        parts.append(f"[{c.collection} | score={c.score:.2f}]\n{c.text}")
    return "\n\n".join(parts)