"""Checks for conformal calibration.

Part A - synthetic scores (no model): the quantile, its coverage, and the plumbing
         of calibrated band edges into the pipeline.

The real numbers are in tests/bench_calibrated.py.
"""
import asyncio
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.fast_path import scan_parallel, scan_sequential  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, KEEP  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import MockVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.retriever import TrustRAGGuard, open_store  # noqa: E402


def part_a():
    assert bt.conformal_quantile(np.arange(1, 100), 0.1) == 90.0
    assert bt.conformal_quantile([3.0, 1.0, 2.0], 0.5) == 2.0
    print("[ok] the conformal rank matches worked examples")

    assert bt.conformal_quantile([1.0, 2.0], 0.01) == math.inf
    print("[ok] too few scores for alpha: inf, so nothing crosses")

    for bad in (0, 1, -0.1):
        try:
            bt.conformal_quantile([1.0], bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("[ok] an impossible alpha is refused")

    rng = np.random.default_rng(0)
    rates = []
    for _ in range(2000):
        fit, new = rng.normal(size=100), rng.normal(size=2000)
        rates.append((new > bt.conformal_quantile(fit, 0.05)).mean())
    assert abs(np.mean(rates) - 0.05) < 0.005, np.mean(rates)
    print(f"[ok] on independent scores, the crossing rate matches alpha ({np.mean(rates):.3f} vs 0.050)")

    items = list(range(200))
    fit, held = bt.split_questions(items)
    assert len(fit) == len(held) == 100 and not set(fit) & set(held) and bt.split_questions(items)[0] == fit
    print("[ok] the split is fixed and has no overlap")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.json")
        bt.save({"low": 0.5}, path)
        assert bt.load(path) == {"low": 0.5}
    print("[ok] values round-trip through the file")

    query = np.zeros(8)
    query[0] = 1.0
    embs = np.tile(np.eye(8)[1], (4, 1)) * 0.6 + query * 0.8
    embs[0] = np.eye(8)[1]                     # similarity 0, then three at 0.8
    texts = ["a b", "c d", "e f", "g h"]
    args = (query, embs, texts, 0.9, 0.5)
    default = scan_sequential(*args)
    moved = scan_sequential(*args, low=0.9, high=0.95)
    assert default.actions.tolist().count(AMBIGUOUS) == 3, default.actions
    assert moved.actions.tolist() == [KEEP] * 4, moved.actions
    assert asyncio.run(scan_parallel(*args, low=0.9, high=0.95)).actions.tolist() == [KEEP] * 4
    r = asyncio.run(guard("q", query, embs, texts, list("abcd"), MockVerifier(), 0.9, 0.5, low=0.9, high=0.95))
    assert r.fusion.actions.tolist() == [KEEP] * 4 and len(r.chunks) == 4
    print("[ok] calibrated band edges reach fusion through every path")

    ref = bt.load(bt.REFERENCE_PATH)
    for key in ("warning", "embedding_model", "dataset", "fitted_on", "over_retrieve", "top_k", "alphas"):
        assert ref[key], key
    assert set(ref["values"]) >= {"threshold", "clean_median", "tau", "low", "high"}
    print(f"[ok] the shipped file names its model, dataset and date ({ref['embedding_model']}, {ref['fitted_on']})")

    try:
        bt.load_reference("some-other-embedder")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        TrustRAGGuard.calibrated(open_store(name="refuse"), len, MockVerifier(), "some-other-embedder")
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[ok] another embedding model is refused")

    built = TrustRAGGuard.calibrated(open_store(name="accept"), len, MockVerifier(), ref["embedding_model"])
    assert (built.low, built.high, built.tau) == (ref["values"]["low"], ref["values"]["high"], ref["values"]["tau"])
    print("[ok] TrustRAGGuard.calibrated() uses the shipped values")


if __name__ == "__main__":
    warmup()
    part_a()
    print("\nStep 5.1 checks finished.")
