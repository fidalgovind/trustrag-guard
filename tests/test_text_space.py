"""Checks for ROUGE-L and the coordinated-payload flag.

Part A - logic checks, a cross-check against rouge_score, and warmup timing.
Part B - pairwise ROUGE-L and the flag on the real toy fixture (text only, no model).
Part C - the same with query-stuffed poison, including the poison-majority pool.

Parts B and C only measure. No threshold exists yet, and none should be tuned
on the toy corpus.
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.layers.text_space import (  # noqa: E402
    coordination_test,
    lcs_length,
    pairwise_rouge_l,
    rouge_l_f,
    tokenize,
    tokens_to_ids,
    warmup,
)


def _arr(*xs):
    return np.array(xs, dtype=np.int32)


def _slow_lcs(a, b):
    """Plain Python reference, used to check the compiled version."""
    t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            t[i + 1][j + 1] = t[i][j] + 1 if x == y else max(t[i][j + 1], t[i + 1][j])
    return t[-1][-1]


# ---------------------------------------------------------------- Part A
def part_a():
    # Must come first: only the first call in a process shows the start-up cost.
    start = time.perf_counter()
    warmup()
    cold_ms = (time.perf_counter() - start) * 1000

    # known answer: ABCBDAB vs BDCABA has an LCS of length 4
    assert lcs_length(_arr(1, 2, 3, 2, 4, 1, 2), _arr(2, 4, 3, 1, 2, 1)) == 4
    assert lcs_length(_arr(1, 2, 3), _arr(3, 2, 1)) == 1
    print("[ok] known LCS answers")

    a = _arr(5, 6, 7, 8)
    assert rouge_l_f(a, a) == 1.0
    assert rouge_l_f(a, _arr(1, 2, 3)) == 0.0
    print("[ok] identical = 1.0, disjoint = 0.0")

    empty = _arr()
    assert lcs_length(empty, a) == 0 and lcs_length(a, empty) == 0
    assert rouge_l_f(empty, a) == 0.0 and rouge_l_f(empty, empty) == 0.0
    print("[ok] empty input gives 0")

    rng = np.random.default_rng(0)
    for _ in range(300):
        x = rng.integers(0, 6, rng.integers(0, 30)).astype(np.int32)
        y = rng.integers(0, 6, rng.integers(0, 30)).astype(np.int32)
        assert lcs_length(x, y) == _slow_lcs(x.tolist(), y.tolist())
        assert rouge_l_f(x, y) == rouge_l_f(y, x)
    print("[ok] 300 random pairs match a plain Python LCS, and F is symmetric")

    for _ in range(200):
        chunks = [rng.integers(0, 8, rng.integers(0, 200)).astype(np.int32) for _ in range(10)]
        m = pairwise_rouge_l(chunks)
        for i, j in itertools.combinations(range(10), 2):
            assert m[i, j] == m[j, i] == rouge_l_f(chunks[i], chunks[j]), (i, j)
    assert pairwise_rouge_l([]).shape == (0, 0)
    assert pairwise_rouge_l([a]).shape == (1, 1)
    assert pairwise_rouge_l([empty, empty])[0, 1] == 0.0
    print("[ok] pairwise_rouge_l equals rouge_l_f on 200 pools, lengths 0-199")

    vocab = {}
    ids = tokens_to_ids(["a", "b", "a"], vocab)
    assert ids.dtype == np.int32 and ids.tolist() == [0, 1, 0] and len(vocab) == 2
    assert tokens_to_ids(["b", "c"], vocab).tolist() == [1, 2]
    print("[ok] tokens_to_ids gives shared int32 ids")

    # coordination_test: copies flagged, unrelated chunks not, scores are graded
    r = coordination_test(["a b c d e f g h", "a b c d e f g x", "p q r s t", "u v w y z"], cutoff=0.5)
    assert r.flagged.tolist() == [True, True, False, False], r.scores
    assert r.scores[0] == 0.875 and r.scores[2] == 0.0
    close = coordination_test(["a b c d", "a b c x", "m n o p"], cutoff=0.5).scores[0]
    far = coordination_test(["a b c d", "a x y z", "m n o p"], cutoff=0.5).scores[0]
    assert close > far, (close, far)
    print("[ok] coordination_test flags copies and gives graded scores")

    assert not coordination_test(["a b c d e", "f g h i j", "k l m n o"]).flagged.any()
    assert not coordination_test(["a b c"]).flagged.any()
    assert coordination_test([]).scores.shape == (0,)
    assert not coordination_test(["", ""]).flagged.any()
    print("[ok] no copies, one chunk, no chunks, empty text: nothing flagged")

    # cross-check against the reference implementation, same tokenization
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    texts = OPENAI_CEO.texts() + [
        "The Eiffel Tower, in Paris, was completed in 1889!",
        "PARIS is home to the Eiffel-Tower; it was finished in 1889.",
        "!!! ???",
        "",
    ]
    pairs = list(itertools.combinations(texts, 2))
    for s, t in pairs:
        assert tokenize(s) == scorer._tokenizer.tokenize(s), s
        vocab = {}
        mine = rouge_l_f(tokens_to_ids(tokenize(s), vocab), tokens_to_ids(tokenize(t), vocab))
        ref = scorer.score(s, t)["rougeL"].fmeasure
        assert abs(mine - ref) < 1e-9, (s, t, mine, ref)
    print(f"[ok] {len(pairs)} text pairs agree with rouge_score, same tokens")

    # timing, to show why warmup() exists (not a latency benchmark)
    x = rng.integers(0, 50, 100).astype(np.int32)
    y = rng.integers(0, 50, 100).astype(np.int32)
    start = time.perf_counter()
    for _ in range(1000):
        rouge_l_f(x, y)
    warm_ms = (time.perf_counter() - start)
    print(f"[ok] first call {cold_ms:.0f} ms (lower once Numba's on-disk cache exists), "
          f"warm call {warm_ms:.3f} ms on 100-token chunks")


# ---------------------------------------------------------------- Part B / C
def _pairwise(texts, kinds, title):
    vocab = {}
    ids = [tokens_to_ids(tokenize(t), vocab) for t in texts]
    rows = [(rouge_l_f(ids[i], ids[j]), i, j)
            for i, j in itertools.combinations(range(len(texts)), 2)]
    rows.sort(key=lambda r: -r[0])

    print(f"\n--- {title} ---")
    print(" rouge-L  pair                  text A / text B")
    for f, i, j in rows[:6]:
        print(f"   {f:.3f}   {kinds[i]:<9} {kinds[j]:<9}  {texts[i][:46]} / {texts[j][:46]}")

    groups = {"clean-clean": [], "poison-poison": [], "clean-poison": []}
    for f, i, j in rows:
        n = (kinds[i] != "clean") + (kinds[j] != "clean")
        groups[("clean-clean", "clean-poison", "poison-poison")[n]].append(f)
    print()
    for name, vals in groups.items():
        if vals:
            print(f" {name:<14} mean {np.mean(vals):.3f}  max {np.max(vals):.3f}  ({len(vals)} pairs)")

    r = coordination_test(texts)
    print(f"\n flag at cutoff {r.cutoff}  (score = closest match in the pool)")
    for i in np.argsort(-r.scores):
        print(f"   {r.scores[i]:.3f}  {'FLAG' if r.flagged[i] else '-':<5} {kinds[i]:<9} {texts[i][:58]}")
    poison = np.array([k != "clean" for k in kinds])
    print(f" caught {int((r.flagged & poison).sum())}/{int(poison.sum())} poison, "
          f"wrongly flagged {int((r.flagged & ~poison).sum())} clean")


def part_b():
    case = OPENAI_CEO
    _pairwise(case.texts(), [d.attack_type for d in case.documents], "Part B: toy fixture")


def part_c():
    case = OPENAI_CEO
    clean = [d.text for d in case.documents if not d.is_poison]
    stuffed = [f"{case.query} {d.text}" for d in case.documents if d.is_poison]
    kinds = ["clean"] * len(clean) + ["stuffed"] * len(stuffed)
    _pairwise(clean + stuffed, kinds, "Part C: 5 stuffed poison (majority, where clustering failed)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    print("\nSteps 3.1-3.3 checks finished.")
