"""Embedding + cosine-similarity primitives for TrustRAG-Guard.

This is the "ruler" every vector-space layer measures with. Two jobs:
  1. turn text into normalized 384-dim vectors (embeddings), and
  2. measure similarity between those vectors (cosine).

The vector-space layers (layers/clustering.py, layers/over_alignment.py and
layers/fitted.py) use the same embedder and the same cosine measure, so we keep
these functions small and well-tested.

The model is loaded lazily and cached, so importing this module is cheap and
only the first embed() call pays the model-loading cost.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim sentence embeddings

# Module-level cache so we load the (heavy) model only once per process.
_MODEL_CACHE: dict[str, object] = {}


def _get_model(model_name: str = DEFAULT_MODEL):
    """Load the SentenceTransformer model once and reuse it afterwards.

    Imported lazily so that code which only needs the cosine math (e.g. unit
    tests on synthetic vectors) does not require sentence-transformers/torch.
    """
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a list of strings into L2-normalized float32 vectors.

    Returns an array of shape (len(texts), 384). Because the vectors are
    normalized to length 1, cosine similarity between any two of them is just
    their dot product (see cosine_matrix / query_similarities below).
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = _get_model(model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,   # L2-normalize -> cosine == dot product
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Full pairwise cosine-similarity matrix for a set of NORMALIZED vectors.

    Shape (N, N); entry [i, j] is cos(vec_i, vec_j). This single matrix is the
    raw material Stage 2's clustering operates on. Values lie in [-1, 1]; the
    diagonal is ~1 (every vector is identical to itself).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {embeddings.shape}")
    return embeddings @ embeddings.T


def query_similarities(
    query: str, documents: list[str], model_name: str = DEFAULT_MODEL
) -> np.ndarray:
    """Cosine similarity of each document to the query.

    Returns a 1-D array of length len(documents), aligned with the input order.
    This is the measurement signal A (Stage 2) will judge: honest chunks are
    relevant, but an optimized poison chunk is often *suspiciously* relevant.
    """
    if not documents:
        return np.empty((0,), dtype=np.float32)

    query_vec = embed([query], model_name)[0]        # shape (384,)
    doc_vecs = embed(documents, model_name)           # shape (N, 384)
    return (doc_vecs @ query_vec).astype(np.float32)  # shape (N,)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two single vectors (handles un-normalized input)."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
