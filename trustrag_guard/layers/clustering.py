"""Layer 1 - clustering baseline (the TrustRAG-style idea).

Question this answers: "do the retrieved chunks split into two real groups,
and if so, which group looks like the injected poison?"

It works on embeddings only (no text, no model), so it is fast and testable
without downloading anything.

Known limit (this is why signal A exists): clustering only sees how chunks
relate to EACH OTHER. Poison that is reworded so it does not clump can hide.

K-Means runs on one thread: for a few dozen points,
coordinating threads costs more than it saves. On Windows the limit applies to
the whole process while it is held, so overlapping calls can restore it early or
leave it at one. That can change speed, not results.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import ThreadpoolController

# Starting guess from DESIGN.md. Not a law of nature - Stage 5 calibrates it.
DEFAULT_TAU = 0.25

# KMeans(k=2) + silhouette need a few points to mean anything.
MIN_CHUNKS = 4

# scikit-learn's OpenMP library only, so torch keeps its own thread count. One
# controller is reused because building one scans every loaded library.
_KMEANS_OPENMP = ThreadpoolController().select(prefix=["vcomp", "libgomp", "libomp"])


@dataclass
class SplitResult:
    silhouette: float       # s for the k=2 split (nan if it could not be computed)
    split: bool             # True if s >= tau, i.e. a real second group exists
    labels: np.ndarray      # cluster id (0/1) per chunk; all 0 when split is False
    suspect: np.ndarray     # bool per chunk; True = candidate poison cluster


def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _mean_intra_cosine(x):
    """Average pairwise cosine inside one group (1.0 for a single point)."""
    n = len(x)
    if n < 2:
        return 1.0
    sims = x @ x.T
    return float((sims.sum() - n) / (n * (n - 1)))  # drop the diagonal of 1s


def _no_split(n, s=float("nan")):
    return SplitResult(s, False, np.zeros(n, dtype=int), np.zeros(n, dtype=bool))


def silhouette_split_test(embeddings, tau=DEFAULT_TAU, seed=42):
    """Always fit k=2, then let the silhouette score decide if the split is real.

    Silhouette is undefined for k=1, so "no attack cluster" is expressed as
    s < tau rather than as a k=1 fit.
    """
    x = _normalize(embeddings)
    n = len(x)
    if n < MIN_CHUNKS:
        return _no_split(n)

    with _KMEANS_OPENMP.limit(limits=1):
        labels = KMeans(n_clusters=2, random_state=seed, n_init=10).fit_predict(x)
    if len(set(labels)) < 2:  # e.g. duplicate points collapse into one cluster
        return _no_split(n)

    s = float(silhouette_score(x, labels))
    if s < tau:
        return _no_split(n, s)

    # Candidate poison = the smaller group; if sizes tie, the denser (tighter) one.
    sizes = [(labels == c).sum() for c in (0, 1)]
    if sizes[0] != sizes[1]:
        poison_label = int(np.argmin(sizes))
    else:
        poison_label = int(np.argmax([_mean_intra_cosine(x[labels == c]) for c in (0, 1)]))

    return SplitResult(s, True, labels, labels == poison_label)
