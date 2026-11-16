"""Offline calibration: fit thresholds from clean data only.

Each value is a conformal quantile of clean scores, so a chosen share of clean
chunks crosses it, instead of a hand-picked number.
The online code only reads the result.

Run it as a script to write thresholds.json, which git ignores:
    python -m trustrag_guard.calibration.build_thresholds
Add --reference to also rewrite reference_thresholds.json, the labelled file that
ships with the code. load_reference() refuses another embedding model, because
these values describe one embedder on one corpus.

Known limits: chunks in one pool are not independent, so the guarantee is
approximate, and the held-out check in tests/bench_calibrated.py tests it. The
guarantee is per chunk: false alarms concentrate on rank 1, which is usually the
correct passage. The values fit this corpus, this embedder and this retrieval
depth. Weights stay hand-set, and the fitted Layer 1 detectors are not added yet.
"""
import json
import math
import os

import numpy as np

from ..fusion import fuse
from ..layers.clustering import silhouette_split_test
from ..layers.layer1_scan import layer1_scan
from ..layers.text_space import coordination_test

ALPHA_THRESHOLD = 0.01   # share of clean top chunks above signal A's threshold
ALPHA_TAU = 0.01         # share of clean pools that split
ALPHA_HIGH = 0.01        # share of clean top chunks dropped
ALPHA_LOW = 0.05         # share of clean top chunks reaching the band
TOP_K = 5
SEED = 42
DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "thresholds.json")
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference_thresholds.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATASET = "BEIR NQ, 50,076 passage subset, seed 42"
FITTED_ON = "2026-11-06"


def conformal_quantile(scores, alpha):
    """Offline step: rank ceil((n + 1)(1 - alpha)) of the sorted scores. With too few
    scores to support alpha it returns inf, so nothing crosses it."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    ordered = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    return float(ordered[rank - 1]) if rank <= len(ordered) else math.inf


def split_questions(questions, seed=SEED):
    """One fixed split: (fitting half, held-out half)."""
    order = np.random.default_rng(seed).permutation(len(questions))
    half = len(questions) // 2
    return [questions[i] for i in order[:half]], [questions[i] for i in order[half:]]


def fit_values(pools, top_k=TOP_K):
    """Offline step. `pools` holds (query_emb, chunk_embs, texts), each sorted by
    similarity, from clean retrieval only."""
    sims = np.concatenate([(embs[:top_k] @ q) for q, embs, _ in pools])
    threshold = conformal_quantile(sims, ALPHA_THRESHOLD)
    clean_median = float(np.median(sims))
    tau = conformal_quantile([silhouette_split_test(embs).silhouette for _, embs, _ in pools], ALPHA_TAU)

    risks = []
    for q, embs, texts in pools:
        fused = fuse(layer1_scan(q, embs, threshold, tau=tau), coordination_test(texts), clean_median)
        risks.append(fused.risk[:top_k])
    risks = np.concatenate(risks)
    return {"threshold": threshold, "clean_median": clean_median, "tau": tau,
            "low": conformal_quantile(risks, ALPHA_LOW), "high": conformal_quantile(risks, ALPHA_HIGH),
            "n_pools": len(pools), "n_scores": int(len(risks))}


def save(values, path=DEFAULT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(values, f, indent=2)


def load(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_reference(values, path=REFERENCE_PATH, embedding_model=EMBEDDING_MODEL,
                   dataset=DATASET, fitted_on=FITTED_ON, over_retrieve=30):
    """Write the labelled values that ship with the code."""
    label = {"warning": "Fitted on this dataset with this embedder. Refit before using them elsewhere.",
             "embedding_model": embedding_model, "dataset": dataset, "fitted_on": fitted_on,
             "over_retrieve": over_retrieve, "top_k": TOP_K,
             "alphas": {"threshold": ALPHA_THRESHOLD, "tau": ALPHA_TAU,
                        "high": ALPHA_HIGH, "low": ALPHA_LOW}}
    save({**label, "values": values}, path)


def load_reference(embedding_model, path=REFERENCE_PATH):
    """The shipped values. Raises ValueError for any other embedding model."""
    reference = load(path)
    if reference["embedding_model"] != embedding_model:
        raise ValueError(f"thresholds were fitted with {reference['embedding_model']}, "
                         f"not {embedding_model}: refit them on your own clean data")
    return reference["values"]


if __name__ == "__main__":
    import sys

    from sentence_transformers import SentenceTransformer

    from ..retriever import DEFAULT_OVER_RETRIEVE, open_store

    data = os.path.join(os.path.dirname(DEFAULT_PATH), "data", "nq")
    model = SentenceTransformer(EMBEDDING_MODEL)
    store = open_store(os.path.join(data, "chroma"), "nq_clean")
    with open(os.path.join(data, "questions.json"), encoding="utf-8") as f:
        fitting, _ = split_questions(json.load(f)["reference"])

    pools = []
    for row in fitting:
        q = model.encode([row["question"]], normalize_embeddings=True)[0].astype(np.float64)
        hits = store.query(query_embeddings=[q.tolist()], n_results=DEFAULT_OVER_RETRIEVE,
                           include=["documents", "embeddings"])
        pools.append((q, np.asarray(hits["embeddings"][0], dtype=np.float64), list(hits["documents"][0])))

    values = fit_values(pools)
    save(values)
    if "--reference" in sys.argv:
        save_reference(values)
    print(json.dumps(values, indent=2))
    print("\nCalibration finished.")
