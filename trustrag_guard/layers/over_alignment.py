"""Layer 1 - query-conditional over-alignment (signal A, our edge).

Clustering looks at how chunks relate to EACH OTHER. This looks at how each
chunk relates to the QUERY. Honest passages are relevant; poison is optimized
to win retrieval, so it can overshoot - be more aligned to the query than
genuine passages normally are.

The threshold must come from CLEAN data (a reference sample), not from the
retrieved pool itself: when poison is a large share of the pool, pool-relative
statistics (median, MAD) get dragged toward the poison and stop working.
"""
from dataclasses import dataclass

import numpy as np

# Flag anything above the clean 99th percentile (DESIGN.md starting guess;
# Stage 5 replaces this with a conformal threshold).
DEFAULT_QUANTILE = 0.99


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def query_similarities(query_emb, chunk_embs):
    """cosine(query, chunk) for every chunk."""
    q = _unit(np.asarray(query_emb, dtype=np.float64).reshape(-1))
    return _unit(chunk_embs) @ q


def clean_retrieval_sims(query_embs, kb_embs, top_k=5):
    """Simulate clean retrieval: for each clean query, the similarities of its
    top_k retrieved chunks from a clean knowledge base. Pool them all."""
    sims = _unit(query_embs) @ _unit(kb_embs).T          # (n_queries, n_chunks)
    top = -np.sort(-sims, axis=1)[:, :top_k]              # best top_k per query
    return top.ravel()


def fit_threshold(clean_sims, quantile=DEFAULT_QUANTILE):
    """Offline step: the clean similarity level that only (1 - quantile) of
    clean retrievals ever exceed."""
    clean_sims = np.asarray(clean_sims, dtype=np.float64).ravel()
    if clean_sims.size == 0:
        raise ValueError("clean_sims is empty - need a clean reference sample")
    return float(np.quantile(clean_sims, quantile))


@dataclass
class AlignmentResult:
    sims: np.ndarray       # cosine(query, chunk) per chunk
    threshold: float
    flagged: np.ndarray    # bool per chunk; True = suspiciously aligned


def over_alignment_test(query_emb, chunk_embs, threshold):
    """Online step: only scores against the pre-fit threshold."""
    sims = query_similarities(query_emb, chunk_embs)
    return AlignmentResult(sims, float(threshold), sims > threshold)
