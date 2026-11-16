"""How does verifier time grow with prompt length?

The rule: 30 different texts at each length, in shuffled
order, with the first calls discarded. Thirty samples support a median, not a tail,
so this prints the median and the fastest and slowest call, never p95 or p99.

Warm-up calls use separate text, because Ollama serves a repeated prompt from its
cache. The shared start of the prompt stays cached, as it would for real calls on
one query, so the fit uses new tokens only. The power state is printed at both
ends, because timing requires mains.

The text is shuffled fixture sentences, so its length is realistic but it is not
a real document. This times one laptop.

Usage: python tests/bench_verifier_length.py [ollama_url]
"""
import os
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.layers.verifier import DEFAULT_OLLAMA_URL, OllamaVerifier  # noqa: E402

LENGTHS = (25, 50, 100, 200, 300, 400)
TEXTS_PER_LENGTH = 30
DISCARD_CALLS = 6


def power_state():
    """Windows only: mains or battery, and the charge."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte),
                        ("percent", ctypes.c_ubyte), ("saver", ctypes.c_ubyte),
                        ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

        s = Status()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        return f"{ {0: 'BATTERY', 1: 'mains'}.get(s.ac, 'unknown') }, {s.percent}% charge"
    except Exception as e:
        return f"unknown ({type(e).__name__})"


def chunk_of(words, rng):
    plain = [d.text for d in OPENAI_CEO.documents]
    out = []
    while len(out) < words:
        out += plain[int(rng.integers(len(plain)))].split()
    return " ".join(out[:words])


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OLLAMA_URL
    verifier = OllamaVerifier(url=url, timeout=180)
    start = time.perf_counter()
    try:
        verifier.warmup()
    except httpx.HTTPError as e:
        raise SystemExit(f"[skip] could not reach Ollama: {type(e).__name__}")
    print(f"power at start: {power_state()}")
    print(f"server: {url}; model load {(time.perf_counter() - start) * 1000:.0f} ms (excluded)")
    for m in verifier.client.get("/api/ps").json()["models"]:
        print(f"loaded: {m['name']}, {m['size'] / 1e9:.1f} GB, in GPU memory: {m['size_vram'] / 1e9:.1f} GB")

    rng = np.random.default_rng(42)
    jobs = [(n, chunk_of(n, rng)) for n in LENGTHS for _ in range(TEXTS_PER_LENGTH)]
    order = rng.permutation(len(jobs))
    warm_rng = np.random.default_rng(7)
    for _ in range(DISCARD_CALLS):
        verifier.ask(OPENAI_CEO.query, chunk_of(100, warm_rng))

    times = {n: [] for n in LENGTHS}
    tokens = {n: [] for n in LENGTHS}
    cached = {n: [] for n in LENGTHS}
    for i in order:
        words, text = jobs[i]
        begin = time.perf_counter()
        reply = verifier.ask(OPENAI_CEO.query, text)
        times[words].append((time.perf_counter() - begin) * 1000)
        tokens[words].append(reply["prompt_eval_count"])
        cached[words].append(reply.get("prompt_eval_cached_count", 0))

    print("\n words  tokens  cached  calls      p50  fastest  slowest  ms per word")
    for n in LENGTHS:
        p50 = np.median(times[n])
        print(f" {n:>5}  {np.median(tokens[n]):>6.0f}  {np.median(cached[n]):>6.0f}  {len(times[n]):>5}  "
              f"{p50:>7.0f}  {min(times[n]):>7.0f}  {max(times[n]):>7.0f}  {p50 / n:>11.1f}")

    new = {n: np.array(tokens[n]) - np.array(cached[n]) for n in LENGTHS}
    x = np.array([t for n in LENGTHS for t in new[n]], dtype=np.float64)
    y = np.array([t for n in LENGTHS for t in times[n]], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\nfit: time = {intercept:.0f} ms + {slope:.2f} ms per new token (R-squared {r2:.3f})")
    print(f"power at end: {power_state()}")
    print("\nLength test finished.")
