"""Stage 3.2 gate: is all-pairs ROUGE-L fast enough?

The rule was fixed before this script first ran:
on 50 chunks of 200 words, p99 of the full online path must be <= LIMIT_MS.
Only the 200-word case decides the verdict; the others are for context.

History: scoring each pair with rouge_l_f() failed at p99 48.8 ms. This script
now times coordination_test(), the full Layer 2 online path.

This times one laptop, single-threaded. It answers this one question and is
not a general latency claim.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fixtures.synthetic_pool import POOL_SIZE, make_pool  # noqa: E402
from trustrag_guard.layers.text_space import coordination_test, warmup  # noqa: E402

LIMIT_MS = 12.5        # half the 25 ms Layer 2 ceiling (DESIGN.md section 7)
PRIMARY_WORDS = 200    # about 1,000 characters
OTHER_WORDS = (50, 100, 400)
DISCARD_ROUNDS = 20
TIMED_ROUNDS = 200


def full_path(texts):
    """What the online path must do: tokenize, map to ids, score every pair, flag."""
    return coordination_test(texts)


def measure(words):
    texts = make_pool(words)
    for _ in range(DISCARD_ROUNDS):
        full_path(texts)
    times = []
    for _ in range(TIMED_ROUNDS):
        start = time.perf_counter()
        full_path(texts)
        times.append((time.perf_counter() - start) * 1000)
    return np.percentile(times, [50, 95, 99])


if __name__ == "__main__":
    warmup()
    pairs = POOL_SIZE * (POOL_SIZE - 1) // 2
    print(f"pool: {POOL_SIZE} chunks, {pairs} pairs, limit p99 <= {LIMIT_MS} ms\n")
    print(" words      p50      p95      p99")
    results = {}
    for words in sorted(OTHER_WORDS + (PRIMARY_WORDS,)):
        results[words] = measure(words)
        p50, p95, p99 = results[words]
        tag = "  <- decides the verdict" if words == PRIMARY_WORDS else ""
        print(f" {words:>5}  {p50:6.1f}   {p95:6.1f}   {p99:6.1f} ms{tag}")

    p99 = results[PRIMARY_WORDS][2]
    verdict = "PASS: no prefilter needed" if p99 <= LIMIT_MS else "FAIL: over the limit"
    print(f"\nverdict: {verdict} (p99 {p99:.1f} ms vs limit {LIMIT_MS} ms)")
