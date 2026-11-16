"""Checks for the parallel fast path.

Part A - synthetic vectors (no model): parallel equals sequential, and the two
         layers really overlap.
Part B - real model: parallel equals sequential on the toy pools.
Part C - real model: repeated parallel runs give one answer.
Part D - synthetic vectors: one-thread K-Means decides the same as the default.

Timing is in tests/bench_fast_path.py, not here.
"""
import asyncio
import contextlib
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard import fast_path  # noqa: E402
from trustrag_guard.fast_path import scan_parallel, scan_sequential  # noqa: E402
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP  # noqa: E402
from trustrag_guard.layers import clustering  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402

# Measured Stage 2 values. Parallel and sequential only have to agree, so nothing
# here is fitted or tuned.
THRESHOLD = 0.831
MEDIAN = 0.612


def same(a, b):
    return (np.array_equal(a.risk, b.risk) and np.array_equal(a.actions, b.actions)
            and np.array_equal(a.sims, b.sims) and np.array_equal(a.coordination, b.coordination)
            and np.array_equal(a.cluster_suspect, b.cluster_suspect))


def synthetic_pool(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 31))
    embs = rng.normal(size=(n, 16))
    embs[: n // 3] = rng.normal(size=16) + 0.1 * rng.normal(size=(n // 3, 16))  # a tight group
    query = rng.normal(size=16)
    words = [f"w{i}" for i in range(40)]
    texts = [" ".join(rng.choice(words, size=30)) for _ in range(n)]
    return query, embs, texts


# ---------------------------------------------------------------- Part A
def part_a():
    for seed in range(20):
        args = (*synthetic_pool(seed), 0.5, 0.2)
        assert same(asyncio.run(scan_parallel(*args)), scan_sequential(*args)), f"seed {seed}"
    print("[ok] parallel result equals sequential on 20 random pools")

    # Each stand-in waits for the other, so this passes only if both run at once.
    barrier = threading.Barrier(2, timeout=10)
    real = fast_path.layer1_scan, fast_path.coordination_test

    def wait_then(fn):
        def wrapped(*a, **k):
            barrier.wait()
            return fn(*a, **k)
        return wrapped

    fast_path.layer1_scan, fast_path.coordination_test = map(wait_then, real)
    try:
        asyncio.run(scan_parallel(*synthetic_pool(0), 0.5, 0.2))
    finally:
        fast_path.layer1_scan, fast_path.coordination_test = real
    print("[ok] the two layers run at the same time")

    query, embs, texts = synthetic_pool(1)
    try:
        asyncio.run(scan_parallel(query, embs, texts[:-1], 0.5, 0.2))
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[ok] mismatched chunks and texts are refused")

    empty = asyncio.run(scan_parallel(np.ones(16), np.zeros((0, 16)), [], 0.5, 0.2))
    assert empty.risk.shape == (0,)
    print("[ok] empty pool: no crash")


# ---------------------------------------------------------------- Part B
def load():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return None
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


def pools():
    case = OPENAI_CEO
    clean = [d.text for d in case.documents if not d.is_poison]
    paraphrase = [d.text for d in case.documents if d.is_poison]
    stuffed = [f"{case.query} {t}" for t in paraphrase]
    return (("clean only", clean), ("clean + 5 paraphrase", clean + paraphrase),
            ("clean + 5 stuffed (majority)", clean + stuffed))


def part_b(enc):
    if enc is None:
        return
    q = enc([OPENAI_CEO.query])[0]
    print("\n pool                           drop  ambiguous  keep")
    for name, texts in pools():
        args = (q, enc(texts), texts, THRESHOLD, MEDIAN)
        parallel = asyncio.run(scan_parallel(*args))
        assert same(parallel, scan_sequential(*args)), name
        print(f" {name:<30} {(parallel.actions == DROP).sum():>4}  "
              f"{(parallel.actions == AMBIGUOUS).sum():>9}  {(parallel.actions == KEEP).sum():>4}")
    print("[ok] parallel result equals sequential on the toy pools")

    import torch

    def thread_counts():
        before = torch.get_num_threads()
        clustering.silhouette_split_test(enc(pools()[1][1]))
        return before, torch.get_num_threads()

    on_main = thread_counts()
    on_worker = asyncio.run(asyncio.to_thread(thread_counts))
    assert on_main[0] == on_main[1] and on_worker[0] == on_worker[1], (on_main, on_worker)
    print("[ok] the K-Means limit leaves torch's thread count alone")


# ---------------------------------------------------------------- Part C
def part_c(enc):
    if enc is None:
        return
    q = enc([OPENAI_CEO.query])[0]
    texts = pools()[2][1]
    args = (q, enc(texts), texts, THRESHOLD, MEDIAN)

    async def repeat():
        return await asyncio.gather(*(scan_parallel(*args) for _ in range(10)))

    first = scan_sequential(*args)
    assert all(same(r, first) for r in asyncio.run(repeat()))
    print("[ok] 10 overlapping parallel runs give the same answer")


# ---------------------------------------------------------------- Part D
class NoLimit:
    def limit(self, limits):
        return contextlib.nullcontext()


def part_d():
    limited = clustering._KMEANS_OPENMP
    assert limited.lib_controllers, "no OpenMP library found to limit"
    for seed in range(200):
        embs = synthetic_pool(seed)[1]
        one = clustering.silhouette_split_test(embs)
        clustering._KMEANS_OPENMP = NoLimit()
        try:
            default = clustering.silhouette_split_test(embs)
        finally:
            clustering._KMEANS_OPENMP = limited
        assert one.split == default.split and abs(one.silhouette - default.silhouette) < 1e-9, seed
        assert np.array_equal(one.labels, default.labels), seed
        assert np.array_equal(one.suspect, default.suspect), seed
    print("[ok] one-thread K-Means matches the default on 200 random pools")


if __name__ == "__main__":
    warmup()
    part_a()
    enc = load()
    part_b(enc)
    part_c(enc)
    part_d()
    print("\nStep 3.5 checks finished.")
