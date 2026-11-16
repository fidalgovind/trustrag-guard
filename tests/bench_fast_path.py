"""Stage 3.5 gate: first latency measurement of the fast path.

The rule was fixed before this script existed:
Layer 1 + Layer 2 + fusion, run in parallel on 50 chunks of 200 words, must reach
p99 <= LIMIT_MS in all of RUNS separate process runs. The same limit applies
to the whole in-memory pipeline, with a mock verifier that costs nothing.

Each timed round does three things on the same pool: embeds the query, runs the
parts one after the other with a timer on each, and runs the parallel path. The
order of the last two alternates, so neither always goes first.

A long-lived event loop is reused, as a server would reuse one. asyncio.run()
would rebuild the loop and its threads every round.

The pool is pseudo-word text, so these timings do not come from real documents.
This times one laptop. It is not a general latency claim, and retrieval does not
exist yet.
"""
import asyncio
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fixtures.synthetic_pool import make_pool  # noqa: E402
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.fusion import fuse  # noqa: E402
from trustrag_guard.layers.layer1_scan import layer1_scan  # noqa: E402
from trustrag_guard.layers.text_space import coordination_test, warmup  # noqa: E402
from trustrag_guard.layers.verifier import MockVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402

LIMIT_MS = 75.0        # half the 150 ms fast-path target
L2_LIMIT_MS = 12.5     # the Layer 2 gate, used only to read a failure
WORDS = 200            # about 1,000 characters, as in the Layer 2 gate
RUNS = 3
DISCARD_ROUNDS = 20
TIMED_ROUNDS = 500
CONTEXT_ROUNDS = 200
SOURCES = [f"source{i}" for i in range(50)]
THRESHOLD = 0.831      # measured in Stage 2; these values do not affect timing
CLEAN_MEDIAN = 0.612

SCOPE = """included in the verdict: silhouette split, signal A, tokenizing, all-pairs
  ROUGE-L, coordination flag, fusion, thread hand-off
timed separately: query embedding (the only text embedded per round)
excluded: retrieval (not built), chunk embedding (done once before timing, as a
  vector store holds them), Numba warm-up, model loading, the first 20 rounds"""


def power_state():
    """Windows only: mains or battery, charge, and battery saver."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte),
                        ("percent", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                        ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

        s = Status()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        source = {0: "battery", 1: "mains"}.get(s.ac, "unknown")
        return f"{source}, {s.percent}% charge, battery saver {'on' if s.saver else 'off'}"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def clock(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - start) * 1000


def one_round(loop, encode, query, embs, texts, parallel_first):
    def sequential():
        layer1, l1 = clock(layer1_scan, q, embs, THRESHOLD)
        coordination, l2 = clock(coordination_test, texts)
        _, fusion = clock(fuse, layer1, coordination, CLEAN_MEDIAN)
        return l1, l2, fusion

    def parallel():
        coro = scan_parallel(q, embs, texts, THRESHOLD, CLEAN_MEDIAN)
        return clock(loop.run_until_complete, coro)[1]

    def pipeline():
        coro = guard(query, q, embs, texts, SOURCES, MockVerifier(), THRESHOLD, CLEAN_MEDIAN)
        return clock(loop.run_until_complete, coro)[1]

    q, embed = clock(encode, query)
    if parallel_first:
        par = parallel()
        l1, l2, fusion = sequential()
    else:
        l1, l2, fusion = sequential()
        par = parallel()
    return {"query": embed, "l1": l1, "l2": l2, "fusion": fusion,
            "sequential": l1 + l2 + fusion, "parallel": par, "with_query": embed + par,
            "pipeline": pipeline()}


def child():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    encode = lambda text: model.encode([text], normalize_embeddings=True)[0]
    texts = make_pool(WORDS)
    embs = model.encode(texts, normalize_embeddings=True)
    query = OPENAI_CEO.query

    warm_ms = clock(warmup)[1]
    loop = asyncio.new_event_loop()
    for i in range(DISCARD_ROUNDS):
        one_round(loop, encode, query, embs, texts, i % 2 == 0)
    rounds = [one_round(loop, encode, query, embs, texts, i % 2 == 0)
              for i in range(TIMED_ROUNDS)]

    # Context only: embeddings with no structure, Layer 1 alone.
    rng = np.random.default_rng(0)
    noise = rng.normal(size=embs.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    q = encode(query)
    noise_l1 = [clock(layer1_scan, q, noise, THRESHOLD)[1] for _ in range(CONTEXT_ROUNDS)]
    loop.close()

    summary = {k: np.percentile([r[k] for r in rounds], [50, 95, 99]).tolist() for k in rounds[0]}
    summary["l1_unstructured"] = np.percentile(noise_l1, [50, 95, 99]).tolist()
    summary["warmup_ms"] = warm_ms
    summary["power"] = power_state()
    print(json.dumps(summary))


def parent():
    print(f"machine: {platform.node()}, {os.cpu_count()} logical CPUs, {platform.system()} {platform.release()}")
    print(f"pool: 50 chunks of {WORDS} words, {DISCARD_ROUNDS} discarded + {TIMED_ROUNDS} timed rounds, {RUNS} runs")
    print(SCOPE)
    labels = (("query", "query embedding"), ("l1", "Layer 1"), ("l2", "Layer 2"), ("fusion", "fusion"),
              ("sequential", "L1 + L2 + fusion, sequential"), ("parallel", "L1 + L2 + fusion, PARALLEL"),
              ("with_query", "query + parallel"), ("pipeline", "whole pipeline, mock verifier"),
              ("l1_unstructured", "Layer 1, unstructured vectors"))
    results = []
    for run in range(1, RUNS + 1):
        done = subprocess.run([sys.executable, __file__, "--child"], capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stderr)
            raise SystemExit(f"run {run} failed")
        r = json.loads(done.stdout.strip().splitlines()[-1])
        results.append(r)
        print(f"\n--- run {run}: {r['power']}; Numba warm-up {r['warmup_ms']:.0f} ms (excluded) ---")
        print(" part                             p50      p95      p99")
        for key, name in labels:
            p50, p95, p99 = r[key]
            print(f" {name:<30} {p50:6.1f}   {p95:6.1f}   {p99:6.1f} ms")

    worst = max(r["parallel"][2] for r in results)
    helped = all(r["parallel"][2] < r["sequential"][2] for r in results)
    print(f"\nverdict: {'PASS' if worst <= LIMIT_MS else 'FAIL'} "
          f"(worst parallel p99 {worst:.1f} ms vs limit {LIMIT_MS} ms)")
    print(f"parallel helps: {'yes' if helped else 'no'} (parallel p99 below sequential p99 in all {RUNS} runs)")
    print(f"worst Layer 2 p99: {max(r['l2'][2] for r in results):.1f} ms vs Layer 2 limit {L2_LIMIT_MS} ms")
    print(f"worst p99 with the query embedded: {max(r['with_query'][2] for r in results):.1f} ms (reported, not the verdict)")
    pipe = max(r["pipeline"][2] for r in results)
    print(f"pipeline: {'PASS' if pipe <= LIMIT_MS else 'FAIL'} (worst p99 {pipe:.1f} ms vs limit {LIMIT_MS} ms; mock verifier)")


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        parent()
