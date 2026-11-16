"""Fail-safe context floor: choose the chunks the generator may see.

The original policy kept the highest-similarity survivors when filtering left too
few chunks. That hands back the poison, because poison is built to score highest.
So the floor is never given similarity:

  1. Every chunk fusion kept goes in.
  2. Below k_min, chunks the verifier called SAFE fill the gap, and no more than
     the gap, taking the least represented source first.
  3. Still below k_min, it returns fewer chunks and signals low confidence.

A SAFE verdict only fills the floor. It never overrides a fusion flag otherwise,
so once fusion kept k_min chunks, verified chunks are not used. A dropped or
unresolved chunk is never returned.

Ties inside a source are broken by a seeded shuffle, not by pool order, because
retrieval returns the pool sorted by similarity.

Known limits: a verifier that wrongly says SAFE can still put poison in the fill,
which is why `filled` is reported. Fusion-kept chunks are not balanced by source.
Cutting the context down to top-k belongs to the retriever wrapper.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .fusion import KEEP
from .layers.verifier import VERIFIED

# The spec's example value. Stage 5 or 6 settles it; do not tune it on the toy corpus.
DEFAULT_K_MIN = 2


@dataclass
class FloorResult:
    selected: np.ndarray    # chunk indices: fusion-kept first, then fills
    filled: np.ndarray      # the indices in `selected` that came from the verifier
    low_confidence: bool    # True when fewer than k_min chunks could be returned


def context_floor(actions, sources, k_min=DEFAULT_K_MIN, seed=42):
    """Online step. `actions` comes from verify_ambiguous() and `sources` holds one
    provenance label per chunk."""
    actions = np.asarray(actions, dtype=object)
    if len(sources) != len(actions):
        raise ValueError("actions and sources must cover the same chunks")

    kept = np.flatnonzero(actions == KEEP)
    candidates = [int(i) for i in np.random.default_rng(seed).permutation(np.flatnonzero(actions == VERIFIED))]
    counts = Counter(sources[i] for i in kept)
    filled = []
    while candidates and len(kept) + len(filled) < k_min:
        chosen = min(candidates, key=lambda i: counts[sources[i]])  # first wins a tie
        candidates.remove(chosen)
        counts[sources[chosen]] += 1
        filled.append(chosen)

    selected = np.concatenate([kept, filled]).astype(int)
    return FloorResult(selected, np.array(filled, dtype=int), len(selected) < k_min)
