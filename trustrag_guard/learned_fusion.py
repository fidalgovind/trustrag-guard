"""Learned fusion: a small logistic regression in place of the hand-set weights.

An option beside the frozen Guard, which stays the default.
No existing module is changed: this file runs Layer 1 and Layer 2 itself and reuses
verify_ambiguous() and context_floor(), so guard_learned() follows the same path as
pipeline.guard() with only the fusion step replaced.

Features per chunk:
    alignment_z      (sim - clean_median) / (threshold - clean_median), unclipped,
                     on the chunk's own corpus scale so it transfers between corpora
    coordination     highest ROUGE-L with any other chunk (Layer 2)
    cluster_suspect  1 if the silhouette split marks the chunk, else 0
    silhouette       the pool's k=2 silhouette (nan -> 0)
    rank             retrieval rank / RANK_SCALE (shortcut check)
    tightness        Stage 2: mean cosine to the 4 nearest neighbours, z-scored in the pool
    wording          Stage 2: a TF-IDF text score, supplied by the caller

Offline: fit_model() and fit_bounds(). Online: scan_learned() and guard_learned().
The thresholds `low` and `high` are conformal quantiles of the learned score on clean
data, as build_thresholds.fit_values() does for the frozen fusion, so the false-alarm
promise is kept whatever the weights are.

Known limits: Layer 2 may now drop a chunk through its weight, so honest template
text can be dropped; the clean `high` quantile bounds this at about 1% of clean top
chunks but does not prevent it. The weights reflect PoisonedRAG-style poison only.
"""
import asyncio
import json
from dataclasses import dataclass

import numpy as np

from .calibration.build_thresholds import ALPHA_HIGH, ALPHA_LOW, conformal_quantile
from .context_floor import DEFAULT_K_MIN, context_floor
from .fusion import AMBIGUOUS, DROP, KEEP
from .layers.clustering import DEFAULT_TAU
from .layers.layer1_scan import layer1_scan
from .layers.text_space import DEFAULT_COPY_CUTOFF, coordination_test
from .layers.verifier import DEFAULT_MAX_VERIFIER_CALLS, verify_ambiguous

STAGE1_FEATURES = ("alignment_z", "coordination", "cluster_suspect", "silhouette", "rank")
RANK_SCALE = 30          # the retriever's over-retrieval k' (retriever.DEFAULT_OVER_RETRIEVE)
NEIGHBOURS = 4           # Stage 2 tightness: poison arrives as 5 near-copies, so 4 others
TOP_K = 5                # chunks per pool used for the conformal bounds, as fit_values()
SEED = 42


def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def tightness(chunk_embs, k=NEIGHBOURS):
    """Mean cosine to the k nearest other chunks, z-scored within the pool."""
    x = _normalize(chunk_embs)
    n = len(x)
    if n < 2:
        return np.zeros(n)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, n - 1)
    near = -np.sort(-sim, axis=1)[:, :k].mean(axis=1)
    return (near - near.mean()) / max(near.std(), 1e-6)


def chunk_features(query_emb, chunk_embs, texts, threshold, clean_median, tau=DEFAULT_TAU, seed=SEED,
                   with_tightness=False, layer1=None, coordination=None):
    """All per-chunk features as a dict of arrays. `layer1` and `coordination` may be
    passed in when already computed (the parallel path does this)."""
    if layer1 is None:
        layer1 = layer1_scan(query_emb, chunk_embs, threshold, tau=tau, seed=seed)
    if coordination is None:
        coordination = coordination_test(texts)
    sims = np.asarray(layer1.alignment.sims, dtype=np.float64)
    s = layer1.clustering.silhouette
    n = len(sims)
    out = {
        "sims": sims,
        "alignment_z": (sims - clean_median) / max(threshold - clean_median, 1e-12),
        "coordination": np.asarray(coordination.scores, dtype=np.float64),
        "cluster_suspect": np.asarray(layer1.clustering.suspect, dtype=np.float64),
        "silhouette": np.full(n, 0.0 if not np.isfinite(s) else float(s)),
        "rank": np.arange(n, dtype=np.float64) / RANK_SCALE,
    }
    if with_tightness:
        out["tightness"] = tightness(chunk_embs)
    return out


@dataclass
class LearnedModel:
    features: tuple          # feature names, in column order
    mean: np.ndarray         # standardisation, from training data only
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    low: float = float("nan")    # conformal bounds, fitted per corpus by fit_bounds()
    high: float = float("nan")

    def score(self, feats):
        """Probability-like score per chunk, from a dict of feature arrays."""
        X = np.column_stack([np.asarray(feats[f], dtype=np.float64) for f in self.features])
        z = (X - self.mean) / self.scale @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self):
        return {"features": list(self.features), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept, "low": self.low, "high": self.high}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["features"]), np.asarray(d["mean"]), np.asarray(d["scale"]),
                   np.asarray(d["coef"]), float(d["intercept"]), float(d.get("low", "nan")), float(d.get("high", "nan")))


def save(model, path):
    json.dump(model.to_dict(), open(path, "w", encoding="utf-8", newline="\n"), indent=1)


def load(path):
    return LearnedModel.from_dict(json.load(open(path, encoding="utf-8")))


def fit_model(X, y, features, seed=SEED):
    """Offline step. X: rows of chunks, columns in `features` order; y: 1 = poison.
    Settings are fixed in advance, with no search."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, dtype=np.float64)
    mean, scale = X.mean(axis=0), np.clip(X.std(axis=0), 1e-12, None)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
    clf.fit((X - mean) / scale, np.asarray(y, dtype=int))
    return LearnedModel(tuple(features), mean, scale, clf.coef_[0].astype(np.float64), float(clf.intercept_[0]))


def fit_bounds(model, clean_pool_features, top_k=TOP_K):
    """Offline step: `low` and `high` from the top-k chunks of clean pools, at the same
    alphas as the frozen fusion (5% reach the band, 1% dropped). Returns a new model."""
    scores = np.concatenate([model.score(f)[:top_k] for f in clean_pool_features])
    return LearnedModel(model.features, model.mean, model.scale, model.coef, model.intercept,
                        conformal_quantile(scores, ALPHA_LOW), conformal_quantile(scores, ALPHA_HIGH))


@dataclass
class LearnedResult:
    risk: np.ndarray       # learned score per chunk
    actions: np.ndarray    # KEEP / AMBIGUOUS / DROP per chunk
    features: dict         # the raw features, kept for analysis


def decide(model, feats):
    """Online step: score and act. `low` and `high` must be fitted first."""
    if not (np.isfinite(model.low) and np.isfinite(model.high)):
        raise ValueError("fit_bounds() must set low and high before scoring")
    risk = model.score(feats)
    actions = np.full(len(risk), KEEP, dtype=object)
    actions[risk >= model.low] = AMBIGUOUS
    actions[risk >= model.high] = DROP
    return LearnedResult(risk, actions, feats)


async def scan_learned(model, query_emb, chunk_embs, texts, threshold, clean_median,
                       tau=DEFAULT_TAU, seed=SEED, extra=None):
    """Both layers at once on worker threads, then the learned fusion. `extra` holds
    any caller-supplied features (Stage 2 wording)."""
    layer1, coordination = await asyncio.gather(
        asyncio.to_thread(layer1_scan, query_emb, chunk_embs, threshold, tau=tau, seed=seed),
        asyncio.to_thread(coordination_test, texts, cutoff=DEFAULT_COPY_CUTOFF),
    )
    feats = chunk_features(query_emb, chunk_embs, texts, threshold, clean_median, tau=tau, seed=seed,
                           with_tightness="tightness" in model.features, layer1=layer1, coordination=coordination)
    feats.update(extra or {})
    return decide(model, feats)


@dataclass
class LearnedGuardResult:
    chunks: list
    low_confidence: bool
    fusion: LearnedResult
    verified: object
    floor: object


async def guard_learned(model, query, query_emb, chunk_embs, texts, sources, verifier, threshold, clean_median,
                        k_min=DEFAULT_K_MIN, tau=DEFAULT_TAU, seed=SEED,
                        max_calls=DEFAULT_MAX_VERIFIER_CALLS, extra=None):
    """pipeline.guard() with the learned fusion in place of fuse()."""
    if len(sources) != len(texts):
        raise ValueError("texts and sources must cover the same chunks")
    fusion = await scan_learned(model, query_emb, chunk_embs, texts, threshold, clean_median,
                                tau=tau, seed=seed, extra=extra)
    verified = await asyncio.to_thread(verify_ambiguous, query, texts, fusion, verifier,
                                       k_min=k_min, max_calls=max_calls)
    floor = context_floor(verified.actions, sources, k_min=k_min, seed=seed)
    return LearnedGuardResult([texts[i] for i in floor.selected], floor.low_confidence,
                              fusion, verified, floor)
