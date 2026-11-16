"""Measurement: how long does the real verifier take per chunk?

Ollama caches a repeated prompt, so each call
uses different text from the one before, and the numbers are not flattered. The
model load and the first pass over each set are excluded.

The 200-word chunks are shuffled fixture sentences. They have a realistic length,
but they are not real documents. This times one laptop.

A percentile is printed only with enough samples behind it. Otherwise the
line shows n/a, and the slowest call is printed instead.
"""
import os
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.layers.verifier import OllamaVerifier  # noqa: E402

ESTIMATE_HIGH_MS = 300.0    # top of the DESIGN.md section 7 estimate
TOY_ROUNDS = 15
LONG_VARIANTS = 20
LONG_ROUNDS = 4
WORDS = 200
MIN_SAMPLES = {95: 100, 99: 500}    # minimum samples per percentile


def toy_chunks():
    plain = [d.text for d in OPENAI_CEO.documents]
    stuffed = [f"{OPENAI_CEO.query} {d.text}" for d in OPENAI_CEO.documents if d.is_poison]
    return plain + stuffed


def long_chunks():
    rng = np.random.default_rng(0)
    plain = [d.text for d in OPENAI_CEO.documents]
    out = []
    for _ in range(LONG_VARIANTS):
        words = []
        while len(words) < WORDS:
            words += plain[int(rng.integers(len(plain)))].split()
        out.append(" ".join(words[:WORDS]))
    return out


def percentile(times, q):
    if len(times) < MIN_SAMPLES[q]:
        return "n/a"
    return f"{np.percentile(times, q):.0f}"


def measure(verifier, texts, rounds):
    for text in texts:     # first pass, discarded
        verifier.verify(OPENAI_CEO.query, text)
    times = []
    for _ in range(rounds):
        for text in texts:
            start = time.perf_counter()
            verifier.verify(OPENAI_CEO.query, text)
            times.append((time.perf_counter() - start) * 1000)
    return times


if __name__ == "__main__":
    verifier = OllamaVerifier()
    start = time.perf_counter()
    try:
        verifier.warmup()
    except httpx.HTTPError as e:
        raise SystemExit(f"[skip] could not reach Ollama: {type(e).__name__}")
    print(f"model load: {(time.perf_counter() - start) * 1000:.0f} ms (excluded)")

    running = verifier.client.get("/api/ps").json()["models"]
    for m in running:
        print(f"loaded: {m['name']}, {m['size'] / 1e9:.1f} GB, in GPU memory: {m['size_vram'] / 1e9:.1f} GB")

    print("\n set                     calls      p50      p95      p99  slowest")
    results = {}
    for name, texts, rounds in (("toy chunks", toy_chunks(), TOY_ROUNDS),
                                (f"{WORDS}-word chunks", long_chunks(), LONG_ROUNDS)):
        times = measure(verifier, texts, rounds)
        p50 = np.median(times)
        results[name] = p50
        print(f" {name:<22} {len(times):>5}  {p50:7.0f}  {percentile(times, 95):>7}  "
              f"{percentile(times, 99):>7}  {max(times):7.0f} ms")

    worst = max(results.values())
    state = "wrong, correct DESIGN.md" if worst > ESTIMATE_HIGH_MS else "holds"
    print(f"\nestimate of 120-300 ms per chunk: {state} (highest p50 {worst:.0f} ms)")
    print("\nStep 4.4 timing finished.")
