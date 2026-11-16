"""Risk fusion v1: one 0-1 risk score per chunk, from every signal so far.

Each signal gives a grade in [0, 1], and the risk is their transparent weighted
sum (DESIGN.md section 5). The risk then sets one of three actions:
    risk <  low           keep
    low <= risk < high    ambiguous, for the Layer 3 verifier (Stage 4)
    risk >= high          drop

Only signal A can reach `high` by itself. Clustering flags the clean chunks when
poison is the majority, and coordination flags honest
template text, so neither may remove a chunk alone. fuse() refuses weights
that would allow it.

All weights and band edges are placeholders, not tuned on the toy corpus. Raw
per-signal scores are kept in the result, so Stage 5 can replace them with
conformal thresholds without a rewrite.

Known limit: with these weights only signal A can drop a chunk, and it misses
plain paraphrase. Such poison can reach the ambiguous band at best.
"""
from dataclasses import dataclass

import numpy as np

# Placeholders. The band is the spec's 0.30 / 0.85; Stage 5 replaces all of these.
WEIGHT_ALIGNMENT = 0.85
WEIGHT_COORDINATION = 0.10
WEIGHT_CLUSTERING = 0.05
RISK_LOW = 0.30
RISK_HIGH = 0.85

KEEP, AMBIGUOUS, DROP = "keep", "ambiguous", "drop"


@dataclass
class FusionResult:
    risk: np.ndarray            # 0-1 per chunk
    actions: np.ndarray         # KEEP / AMBIGUOUS / DROP per chunk
    # raw inputs, kept for calibration
    sims: np.ndarray            # cosine(query, chunk)
    alignment_grade: np.ndarray # 0 at the clean median, 1 at the threshold or above
    coordination: np.ndarray    # highest ROUGE-L with any other chunk
    cluster_suspect: np.ndarray # bool, from the silhouette split test


def alignment_grade(sims, threshold, clean_median):
    """Place each similarity on the clean scale: 0 at a typical clean retrieval
    (`clean_median`), 1 at the clean threshold. Both come from offline clean
    data, so no new number is needed."""
    span = max(threshold - clean_median, 1e-12)
    return np.clip((np.asarray(sims, dtype=np.float64) - clean_median) / span, 0.0, 1.0)


def fuse(layer1, coordination, clean_median,
         w_alignment=WEIGHT_ALIGNMENT, w_coordination=WEIGHT_COORDINATION,
         w_clustering=WEIGHT_CLUSTERING, low=RISK_LOW, high=RISK_HIGH):
    """Online step: combine a Layer1Result and a CoordinationResult.

    `clean_median` is the median of the same clean sims that fitted the
    threshold (over_alignment.clean_retrieval_sims).
    """
    if max(w_coordination, w_clustering) >= high:
        raise ValueError("clustering and coordination must not reach `high` alone")
    sims = layer1.alignment.sims
    if len(sims) != len(coordination.scores):
        raise ValueError("layer1 and coordination must score the same chunks")

    grade = alignment_grade(sims, layer1.alignment.threshold, clean_median)
    suspect = layer1.clustering.suspect
    risk = (w_alignment * grade
            + w_coordination * coordination.scores
            + w_clustering * suspect.astype(np.float64))

    actions = np.full(len(risk), KEEP, dtype=object)
    actions[risk >= low] = AMBIGUOUS
    actions[risk >= high] = DROP
    return FusionResult(risk, actions, sims, grade, coordination.scores, suspect)
