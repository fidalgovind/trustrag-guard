"""Checks for trustrag_guard/config.py: one configuration object for the guard.

Part A - synthetic data only, no model download, about a second:
         defaults match the modules, loading matches the shipped file, bad values
         are refused, and a config drives guard() and the wrapper exactly like the
         same values passed by hand.
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import uuid

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from trustrag_guard import config as cfg  # noqa: E402
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.context_floor import DEFAULT_K_MIN  # noqa: E402
from trustrag_guard.fusion import RISK_HIGH, RISK_LOW  # noqa: E402
from trustrag_guard.layers.clustering import DEFAULT_TAU  # noqa: E402
from trustrag_guard.layers.text_space import DEFAULT_COPY_CUTOFF  # noqa: E402
from trustrag_guard.layers.verifier import DEFAULT_MAX_VERIFIER_CALLS, MockVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def pool(n=12, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    embs = unit(rng.normal(size=(n, dim)))
    texts = [f"passage {i} says topic {i % 3} matters" for i in range(n)]
    return unit(rng.normal(size=dim)), embs, texts, [f"s{i % 4}" for i in range(n)]


def part_a():
    from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, TrustRAGGuard, open_store

    c = cfg.GuardConfig(threshold=0.8, clean_median=0.2)
    assert (c.tau, c.low, c.high, c.copy_cutoff, c.k_min, c.max_calls, c.over_retrieve, c.target_k) == (
        DEFAULT_TAU, RISK_LOW, RISK_HIGH, DEFAULT_COPY_CUTOFF, DEFAULT_K_MIN, DEFAULT_MAX_VERIFIER_CALLS,
        DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K), c
    print("[ok] defaults are the module constants, not copies")

    ref = cfg.GuardConfig.from_reference()
    shipped = json.load(open(bt.REFERENCE_PATH, encoding="utf-8"))["values"]
    assert all(getattr(ref, k) == shipped[k] for k in cfg.FITTED_KEYS), (ref, shipped)
    print("[ok] from_reference() returns the shipped values exactly")

    try:
        cfg.GuardConfig.from_reference("bge-small-en")
        raise AssertionError("another embedder was accepted")
    except ValueError:
        pass
    print("[ok] from_reference() refuses another embedder")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "thresholds.json")
        bt.save({"threshold": 0.7, "clean_median": 0.3, "tau": 0.2, "low": 0.5, "high": 0.9,
                 "n_pools": 3, "n_scores": 15}, path)
        f = cfg.GuardConfig.from_file(path, k_min=3)
        assert (f.threshold, f.clean_median, f.tau, f.low, f.high, f.k_min) == (0.7, 0.3, 0.2, 0.5, 0.9, 3)
    print("[ok] from_file() reads a thresholds.json and accepts overrides")

    bad = [dict(threshold=0.2, clean_median=0.5), dict(threshold=0.8, clean_median=0.2, low=0.9, high=0.5),
           dict(threshold=0.8, clean_median=0.2, tau=1.5), dict(threshold=0.8, clean_median=0.2, copy_cutoff=0),
           dict(threshold=0.8, clean_median=0.2, k_min=-1), dict(threshold=0.8, clean_median=0.2, target_k=0),
           dict(threshold=0.8, clean_median=0.2, over_retrieve=3, target_k=5), dict(threshold=1.2, clean_median=0.2)]
    for b in bad:
        try:
            cfg.GuardConfig(**b)
            raise AssertionError(f"accepted {b}")
        except ValueError:
            pass
    try:
        c.with_changes(low=0.95)
        raise AssertionError("with_changes() skipped validation")
    except ValueError:
        pass
    print(f"[ok] {len(bad) + 1} impossible configurations are refused")

    q, embs, texts, sources = pool()
    by_hand = asyncio.run(guard("q", q, embs, texts, sources, MockVerifier(), 0.8, 0.2,
                                k_min=2, tau=DEFAULT_TAU, seed=42, cutoff=DEFAULT_COPY_CUTOFF,
                                max_calls=3, low=RISK_LOW, high=RISK_HIGH))
    by_cfg = asyncio.run(guard("q", q, embs, texts, sources, MockVerifier(), **c.guard_kwargs()))
    assert np.array_equal(by_hand.fusion.risk, by_cfg.fusion.risk)
    assert np.array_equal(by_hand.floor.selected, by_cfg.floor.selected)
    print("[ok] guard(**config.guard_kwargs()) equals the same values passed by hand")

    store = open_store(name=f"t{uuid.uuid4().hex}")
    shipped_guard = TrustRAGGuard.calibrated(store, lambda _: q, MockVerifier(), "all-MiniLM-L6-v2")
    config_guard = TrustRAGGuard(collection=store, embed=lambda _: q, verifier=MockVerifier(),
                                 **ref.retriever_kwargs())
    fields = ("threshold", "clean_median", "tau", "low", "high", "k_min", "max_calls", "over_retrieve", "target_k")
    assert all(getattr(shipped_guard, k) == getattr(config_guard, k) for k in fields)
    print("[ok] TrustRAGGuard(**config.retriever_kwargs()) equals TrustRAGGuard.calibrated()")

    try:
        c.with_changes(seed=7).retriever_kwargs()
        raise AssertionError("a seed the wrapper ignores was accepted")
    except ValueError:
        pass
    print("[ok] retriever_kwargs() refuses settings the wrapper would silently ignore")

    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import trustrag_guard.config; "
             "print('chromadb' in sys.modules or 'langchain_core' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe, ROOT], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout
    print("[ok] importing config does not load Chroma or LangChain")


if __name__ == "__main__":
    print("=== Part A: configuration (synthetic) ===")
    part_a()
    print("\nConfig checks finished.")
