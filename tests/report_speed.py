"""Evaluation report, run D: speed by pool size.

Fast path: query embedding, then Layer 1 + Layer 2 + fusion in parallel, on make_pool
chunks of about 200 words, at 5, 10, 30 and 50 chunks. 20 rounds discarded and 500
timed per size, so p99 is allowed. The verdict uses query + parallel scan, the
stricter of the two, against 150 ms at p99 for every size. Chunk embeddings are made
once before timing, as a store holds them. Retrieval is not included, because the
pool is synthetic; Stage 6 measured it on real pools (6.5 ms).

Verifier: one query forced to 1, 2 or 3 calls (the cap), on 200-word chunks, with
100 queries per count, so p50 and p95 only. Each chunk is new text, so Ollama's
cache cannot answer, and warm-up uses separate text. There is no verifier target.

Power is printed at both ends. Run alone, on mains. Times one laptop.
Each verifier count checks power before and after and is marked NOT VALID unless both
are mains. --verifier-only skips the fast path and writes report_speed_verifier.json.
Usage: python tests/report_speed.py [--verifier-only]
"""
import asyncio
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_fast_path import CLEAN_MEDIAN, THRESHOLD, clock, power_state  # noqa: E402
from bench_verifier_length import chunk_of  # noqa: E402
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fixtures.synthetic_pool import make_pool  # noqa: E402
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import OllamaVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.layers.verifier import MockVerifier  # noqa: E402

SIZES = (5, 10, 30, 50)
WORDS = 200
DISCARD_ROUNDS = 20
TIMED_ROUNDS = 500
CALLS = (1, 2, 3)
QUERIES_PER_COUNT = 100
LIMIT_MS = 150.0
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "report_speed.json")


def pct(values, points):
    return [float(x) for x in np.percentile(values, points)]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    verifier_only = "--verifier-only" in sys.argv
    if verifier_only:
        OUT = OUT.replace(".json", "_verifier.json")
    print(f"machine: {platform.node()}, {os.cpu_count()} logical CPUs; power at start: {power_state()}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    encode = lambda text: model.encode([text], normalize_embeddings=True)[0]
    texts_all = make_pool(WORDS)
    embs_all = model.encode(texts_all, normalize_embeddings=True)
    query = OPENAI_CEO.query
    print(f"Numba warm-up {clock(warmup)[1]:.0f} ms (excluded)")
    loop = asyncio.new_event_loop()
    out = {"fast_path": {}, "verifier": {}}

    print("\nFast path, ms (500 timed rounds per size)")
    print(" chunks  part                        p50     p95     p99")
    worst = 0.0
    for n in () if verifier_only else SIZES:
        texts, embs = texts_all[:n], embs_all[:n]
        sources = [f"source{i}" for i in range(n)]
        rows = {"query": [], "parallel": [], "with_query": [], "pipeline": []}
        for r in range(DISCARD_ROUNDS + TIMED_ROUNDS):
            q, t_q = clock(encode, query)
            t_p = clock(loop.run_until_complete, scan_parallel(q, embs, texts, THRESHOLD, CLEAN_MEDIAN))[1]
            t_g = clock(loop.run_until_complete, guard(query, q, embs, texts, sources, MockVerifier(), THRESHOLD, CLEAN_MEDIAN))[1]
            if r >= DISCARD_ROUNDS:
                rows["query"].append(t_q)
                rows["parallel"].append(t_p)
                rows["with_query"].append(t_q + t_p)
                rows["pipeline"].append(t_g)
        out["fast_path"][n] = {k: pct(v, [50, 95, 99]) for k, v in rows.items()}
        worst = max(worst, out["fast_path"][n]["with_query"][2])
        for key, name in (("query", "query embedding"), ("parallel", "L1 + L2 + fusion, parallel"),
                          ("with_query", "query + parallel (verdict)"), ("pipeline", "pipeline, mock verifier")):
            p50, p95, p99 = out["fast_path"][n][key]
            print(f" {n:>6}  {name:<26} {p50:6.1f}  {p95:6.1f}  {p99:6.1f}")
    loop.close()
    if not verifier_only:
        print(f"\nfast path verdict: {'MET' if worst < LIMIT_MS else 'NOT MET'} (worst p99 {worst:.1f} ms against {LIMIT_MS:.0f} ms)")

    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()
    for m in verifier.client.get("/api/ps").json()["models"]:
        print(f"loaded: {m['name']}, in GPU memory {m['size_vram'] / 1e9:.1f} of {m['size'] / 1e9:.1f} GB")
    warm = np.random.default_rng(7)
    for _ in range(6):
        verifier.ask(query, chunk_of(WORDS, warm))
    rng = np.random.default_rng(42)
    print("\nVerifier per query, ms (100 queries per count; p99 not allowed below 500)")
    print(" calls     p50     p95  slowest")
    for k in CALLS:
        before = power_state()
        per_query = []
        for _ in range(QUERIES_PER_COUNT):
            chunks = [chunk_of(WORDS, rng) for _ in range(k)]
            start = time.perf_counter()
            for c in chunks:
                verifier.verify(query, c)
            per_query.append((time.perf_counter() - start) * 1000)
        p50, p95 = pct(per_query, [50, 95])
        after = power_state()
        valid = before.startswith("mains") and after.startswith("mains")
        out["verifier"][k] = {"p50": p50, "p95": p95, "max": float(max(per_query)), "n": len(per_query),
                              "power_before": before, "power_after": after, "valid": valid}
        print(f" {k:>5} {p50:7.0f} {p95:7.0f} {max(per_query):8.0f}  {'valid' if valid else 'NOT VALID'} ({before} -> {after})")

    out["power_end"] = power_state()
    print(f"\npower at end: {out['power_end']}")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nSpeed checks finished.")
