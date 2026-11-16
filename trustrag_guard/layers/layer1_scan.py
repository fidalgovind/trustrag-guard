"""Layer 1 scan: run both vector-space signals and hand Stage 3 one result.

Signals so far:
  - clustering: is there a second group, and which chunks are in it?
  - over-alignment (signal A): is a chunk more query-aligned than clean chunks ever are?

Isolation Forest / Mahalanobis join this scan in Stage 5, once a clean
reference exists to fit them on.
"""
from dataclasses import dataclass

import numpy as np

from .clustering import DEFAULT_TAU, SplitResult, silhouette_split_test
from .over_alignment import AlignmentResult, over_alignment_test


@dataclass
class Layer1Result:
    clustering: SplitResult      # clustering output (silhouette, split, labels, suspect)
    alignment: AlignmentResult   # signal A output (sims, threshold, flagged)
    flagged: np.ndarray          # bool per chunk: flagged by ANY signal


def layer1_scan(query_emb, chunk_embs, threshold, tau=DEFAULT_TAU, seed=42):
    """Score retrieved chunks. `threshold` is the clean over-alignment
    threshold fitted offline (over_alignment.fit_threshold)."""
    clustering = silhouette_split_test(chunk_embs, tau=tau, seed=seed)
    alignment = over_alignment_test(query_emb, chunk_embs, threshold)
    return Layer1Result(clustering, alignment, clustering.suspect | alignment.flagged)