"""
AI Software Company — FastEmbed embedding function for Chroma (Step 10)

Architecture references:
  Section 14  Embedding: all-MiniLM-L6-v2 (local, free)
  F20         Chunk size 200 tokens, no overlap

Why this file exists:
  - chromadb-client (thin client) has no default embedding function.
  - We attach a FastEmbedEmbeddingFunction to every collection so the
    client computes embeddings locally (ONNX runtime, ~200 MB total)
    using the exact model from the architecture: all-MiniLM-L6-v2.
  - The vector is then sent to the Chroma server; the server stores it.
  - This keeps the architecture's intent (local + free + same model)
    without needing PyTorch (~2 GB) or C++ Build Tools on Windows.

Important fix (2026-10-09):
  chromadb-client 0.5.23's convert_np_embeddings_to_list() calls
  .tolist() on every embedding. Plain Python lists do not have that
  method, so we must return numpy arrays from __call__. This is the
  standard contract for Chroma embedding functions (Chroma's own
  HuggingFaceEmbeddingFunction also returns numpy arrays).
"""

from __future__ import annotations

from typing import Iterable

# FastEmbed's TextEmbedding is loaded lazily so importing this module is cheap
# and tests that don't need embeddings stay fast.
_model = None


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Section 14


def _get_model():
    """Return the lazily-initialised FastEmbed TextEmbedding model."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        # First call downloads the ONNX model (~45-90 MB) to a local cache.
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def _to_list(vec) -> list[float]:
    """
    Convert a single embedding to a plain Python list of floats.
    Handles both numpy arrays (which have .tolist()) and plain lists/tuples.
    Used only by embed_documents/embed_query for external callers.
    """
    if hasattr(vec, "tolist"):
        return [float(x) for x in vec.tolist()]
    return [float(x) for x in vec]


def embed_documents(texts: Iterable[str]) -> list[list[float]]:
    """
    Embed a list of documents as plain Python lists (for external callers).
    NOTE: ChromaDB's __call__ does NOT use this; see FastEmbedEmbeddingFunction.
    """
    model = _get_model()
    out: list[list[float]] = []
    for vec in model.embed(list(texts)):
        out.append(_to_list(vec))
    return out


def embed_query(text: str) -> list[float]:
    """Embed a single query string as a plain Python list (for external callers)."""
    model = _get_model()
    for vec in model.query_embed([text]):
        return _to_list(vec)
    return []


class FastEmbedEmbeddingFunction:
    """
    Chroma-compatible embedding function.

    Chroma's EmbeddingFunction protocol expects:
      __call__(self, input: Documents) -> Embeddings

    We return a list of numpy arrays (NOT plain lists), because
    chromadb-client 0.5.23's convert_np_embeddings_to_list() calls
    .tolist() on each element. This matches the contract used by
    Chroma's own HuggingFaceEmbeddingFunction.

    We do NOT subclass chromadb's EmbeddingFunction base class, because
    the thin client and the server disagree on its import path in some
    versions. Chroma only calls `__call__` and optionally `name()`, so a
    plain duck-typed class is stable across versions.
    """

    def __call__(self, input):  # noqa: A002 (name matches Chroma's protocol)
        model = _get_model()
        # IMPORTANT: return list[np.ndarray], not list[list[float]].
        # Each element must expose .tolist() for chromadb-client.
        return list(model.embed(list(input)))

    def name(self) -> str:
        # Stable identifier Chroma uses to validate embedding function
        # compatibility when a collection is reopened.
        return "fastembed-all-MiniLM-L6-v2"


_embedding_fn_singleton: FastEmbedEmbeddingFunction | None = None


def get_embedding_function() -> FastEmbedEmbeddingFunction:
    """Return the process-wide singleton embedding function."""
    global _embedding_fn_singleton
    if _embedding_fn_singleton is None:
        _embedding_fn_singleton = FastEmbedEmbeddingFunction()
    return _embedding_fn_singleton