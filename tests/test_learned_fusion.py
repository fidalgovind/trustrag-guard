"""Checks for the learned fusion.

Part A: synthetic vectors and texts, no model download, about a second.
Usage: python tests/test_learned_fusion.py
"""
import asyncio
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP  # noqa: E402
from trustrag_guard.layers.verifier import MockVerifier  # noqa: E402
from trustrag_guard import learned_fusion as lf  # noqa: E402


def synthetic_pool(rng, n_clean=25, n_poison=5, dim=32):
    """Clean chunks scattered round the query; poison as 5 near-copies close to it."""
    q = lf._normalize(rng.normal(size=dim))
    clean = lf._normalize(q * 0.5 + rng.normal(size=(n_clean, dim)) * 0.25)
    base = lf._normalize(q * 0.9 + rng.normal(size=dim) * 0.1)
    poison = lf._normalize(base + rng.normal(size=(n_poison, dim)) * 0.02)
    embs = np.vstack([clean, poison])
    texts = [f"honest passage number {i} about topic {i % 7} with words {i * 3}" for i in range(n_clean)]
    texts += [f"the answer is surely the attacker answer, says copy {i}" for i in range(n_poison)]
    labels = np.r_[np.zeros(n_clean), np.ones(n_poison)]
    order = np.argsort(-(embs @ q), kind="stable")
    return q, embs[order], [texts[i] for i in order], labels[order]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    threshold, median = 0.9, 0.5

    # Part A1: tightness singles out the near-copies.
    q, embs, texts, y = synthetic_pool(rng)
    t = lf.tightness(embs)
    assert t[y == 1].min() > t[y == 0].max(), f"near-copies should be tightest: {t[y == 1].min():.2f} vs {t[y == 0].max():.2f}"
    assert lf.tightness(embs[:1]).tolist() == [0.0], "a single chunk gets tightness 0"
    print("[ok] tightness ranks the 5 near-copies above every clean chunk")

    # Part A2: features have the right shapes; rank is position / 30.
    f = lf.chunk_features(q, embs, texts, threshold, median, with_tightness=True)
    assert all(len(v) == len(texts) for v in f.values()), "every feature covers every chunk"
    assert f["rank"][0] == 0 and abs(f["rank"][3] - 3 / 30) < 1e-12, "rank is position / RANK_SCALE"
    assert np.allclose(f["alignment_z"], (f["sims"] - median) / (threshold - median)), "alignment z is unclipped"
    print("[ok] chunk features: shapes, rank scale and unclipped alignment z")

    # Part A3: a fitted model scores poison higher, and save/load keeps it exact.
    pools = [synthetic_pool(rng) for _ in range(20)]
    feats = [lf.chunk_features(p[0], p[1], p[2], threshold, median) for p in pools]
    X = np.vstack([np.column_stack([fe[n] for n in lf.STAGE1_FEATURES]) for fe in feats])
    Y = np.concatenate([p[3] for p in pools])
    model = lf.fit_model(X, Y, lf.STAGE1_FEATURES)
    s = np.concatenate([model.score(fe) for fe in feats])
    assert s[Y == 1].mean() > s[Y == 0].mean() + 0.5, "poison should score well above clean"
    clean_feats = [lf.chunk_features(p[0], p[1][p[3] == 0], [x for x, l in zip(p[2], p[3]) if l == 0],
                                     threshold, median) for p in pools]
    model = lf.fit_bounds(model, clean_feats)
    assert 0 < model.low <= model.high <= 1, f"bounds out of order: {model.low}, {model.high}"
    path = os.path.join(tempfile.mkdtemp(), "w.json")
    lf.save(model, path)
    back = lf.load(path)
    assert np.array_equal(back.score(feats[0]), model.score(feats[0])) and back.low == model.low, "save/load must be exact"
    print("[ok] fit, conformal bounds in order, and an exact JSON round trip")

    # Part A4: decide() refuses an unbounded model; actions follow the bounds.
    unbounded = lf.LearnedModel(model.features, model.mean, model.scale, model.coef, model.intercept)
    try:
        lf.decide(unbounded, feats[0])
        raise AssertionError("decide() must refuse a model without bounds")
    except ValueError:
        pass
    r = lf.decide(model, feats[0])
    assert set(r.actions) <= {KEEP, AMBIGUOUS, DROP}
    assert all((a == DROP) == (x >= model.high) for a, x in zip(r.actions, r.risk)), "DROP exactly at or above high"
    print("[ok] decide(): refuses missing bounds; DROP exactly at or above `high`")

    # Part A5: the learned guard runs end to end with the mock verifier.
    p = pools[0]
    res = asyncio.run(lf.guard_learned(model, "q", p[0], p[1], p[2], [f"s{i}" for i in range(len(p[2]))],
                                       MockVerifier(), threshold, median))
    assert len(res.chunks) == len(res.floor.selected) and not any(res.fusion.actions[i] == DROP for i in res.floor.selected)
    print("[ok] guard_learned(): end to end, no dropped chunk reaches the context")

    print("\nLearned fusion checks finished.")
