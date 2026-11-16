"""Checks for the fail-safe context floor.

Part A - hand-made actions (no model): the floor's rules.
Part B - real embeddings: what reaches the generator, with scripted verifiers.
Part C - real embeddings: a verifier that wrongly calls poison SAFE.

The verifier here is scripted, so these are bounds and not detection results.
k_min = 4 is shown only to expose a trade-off. It is not a tuned value.
"""
import asyncio
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.context_floor import context_floor  # noqa: E402
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import (  # noqa: E402
    UNSAFE,
    VERIFIED,
    MockVerifier,
    verify_ambiguous,
)

# Measured Stage 2 values, not fitted or tuned here.
THRESHOLD = 0.831
MEDIAN = 0.612


# ---------------------------------------------------------------- Part A
def part_a():
    r = context_floor([KEEP, VERIFIED, KEEP, VERIFIED], list("abcd"), k_min=2)
    assert r.selected.tolist() == [0, 2] and r.filled.size == 0 and not r.low_confidence, r
    print("[ok] a SAFE chunk is not used once fusion kept enough chunks")

    r = context_floor([KEEP, VERIFIED, VERIFIED, VERIFIED], list("abcd"), k_min=3)
    assert len(r.selected) == 3 and len(r.filled) == 2 and not r.low_confidence, r
    print("[ok] SAFE chunks fill only the gap up to k_min, and no more")

    r = context_floor([DROP, DROP, KEEP], list("abc"), k_min=3)
    assert r.selected.tolist() == [2] and r.low_confidence, r
    print("[ok] a dropped chunk is never returned, even below k_min")

    r = context_floor([AMBIGUOUS, AMBIGUOUS], list("ab"), k_min=2)
    assert r.selected.size == 0 and r.low_confidence, r
    print("[ok] an unresolved chunk is never returned")

    r = context_floor([DROP, VERIFIED], list("ab"), k_min=2)
    assert r.selected.tolist() == [1] and r.low_confidence, r
    r = context_floor([], [], k_min=2)
    assert r.selected.size == 0 and r.low_confidence, r
    print("[ok] low confidence is signalled when the floor cannot be met")

    for seed in range(50):
        r = context_floor([KEEP, VERIFIED, VERIFIED, VERIFIED], ["a", "a", "a", "b"], k_min=2, seed=seed)
        assert r.filled.tolist() == [3], (seed, r)
    print("[ok] the fill takes an unrepresented source first")

    params = inspect.signature(context_floor).parameters
    assert not {"sims", "similarity", "scores", "risk"} & set(params), params
    picks = [context_floor([VERIFIED] * 5, list("abcde"), k_min=1, seed=s).filled[0] for s in range(5000)]
    rates = np.bincount(picks, minlength=5) / 5000
    assert (np.abs(rates - 0.2) < 0.02).all(), rates
    print(f"[ok] similarity is not an input, and the top-ranked chunk has no preference ({rates[0]:.3f} vs 0.200)")

    a = context_floor([VERIFIED] * 6, list("aabbcc"), k_min=3)
    b = context_floor([VERIFIED] * 6, list("aabbcc"), k_min=3)
    assert a.selected.tolist() == b.selected.tolist()
    assert sorted(a.filled.tolist()) == sorted(a.selected.tolist())
    print("[ok] same seed, same choice")

    try:
        context_floor([KEEP], [], k_min=1)
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[ok] mismatched sources are refused")

    # A hand-built scenario. Index 1 is poison that the verifier called SAFE.
    actions = [KEEP, VERIFIED, KEEP]
    r = context_floor(actions, list("abc"), k_min=2)
    assert 1 not in r.selected
    print("[ok] wrongly SAFE poison stays out when fusion kept enough chunks")
    r = context_floor([DROP, VERIFIED, KEEP, DROP], list("abcd"), k_min=2)
    assert r.selected.tolist() == [2, 1] and r.filled.tolist() == [1], r
    print("[ok] below the floor, wrongly SAFE poison can enter, but only in `filled`")


# ---------------------------------------------------------------- Parts B, C
def load():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return None
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


def pools():
    docs = OPENAI_CEO.documents
    clean = [d for d in docs if not d.is_poison]
    poison = [d for d in docs if d.is_poison]
    stuffed = [(f"{OPENAI_CEO.query} {d.text}", d.source) for d in poison]
    plain = [(d.text, d.source) for d in clean]
    return (("clean only", plain, []),
            ("clean + 5 paraphrase", plain, [(d.text, d.source) for d in poison]),
            ("clean + 5 stuffed", plain, stuffed))


def run(enc, clean, poison, mock, k_min):
    q = enc([OPENAI_CEO.query])[0]
    chunks = clean + poison
    texts = [t for t, _ in chunks]
    sources = [s for _, s in chunks]
    f = asyncio.run(scan_parallel(q, enc(texts), texts, THRESHOLD, MEDIAN))
    v = verify_ambiguous(OPENAI_CEO.query, texts, f, mock)
    floor = context_floor(v.actions, sources, k_min=k_min)
    chosen = [texts[i] for i in floor.selected]
    bad = {t for t, _ in poison}
    return (len(chosen), len(floor.filled), sum(t in bad for t in chosen),
            any(OPENAI_CEO.correct_answer in t for t in chosen), floor.low_confidence)


def part_b(enc):
    if enc is None:
        return
    print("\n--- what reaches the generator ---")
    print(" pool                  verifier       k_min  chunks  filled  poison  answer  low conf")
    for name, clean, poison in pools():
        bad = [t for t, _ in poison]
        for label, mock in (("always right", lambda: MockVerifier(unsafe=bad)),
                            ("always SAFE", MockVerifier),
                            ("always UNSAFE", lambda: MockVerifier(default=UNSAFE))):
            for k_min in (2, 4):
                n, filled, leaked, answer, low = run(enc, clean, poison, mock(), k_min)
                print(f" {name:<21} {label:<14} {k_min:>5}  {n:>6}  {filled:>6}  {leaked:>6}  "
                      f"{'yes' if answer else 'NO':>6}  {'yes' if low else '-':>8}")


def part_c(enc):
    if enc is None:
        return
    print("\n--- verifier wrongly calls all poison SAFE ---")
    clean = [(d.text, d.source) for d in OPENAI_CEO.documents if not d.is_poison]
    poison = [(d.text, d.source) for d in OPENAI_CEO.documents if d.is_poison]

    n, filled, leaked, _, _ = run(enc, clean, poison, MockVerifier(), 2)
    assert leaked == 0 and filled == 0, (leaked, filled)
    print("[ok] fusion kept 2 chunks, so 0 of 5 wrongly SAFE poison chunks reach the generator")

    answers = clean[:2]  # the two ambiguous clean chunks; fusion keeps nothing in this pool
    n, filled, leaked, _, low = run(enc, answers, poison, MockVerifier(), 2)
    assert filled == 2 and leaked <= 2 and n == 2
    print(f"[ok] with nothing kept, the fill is capped at {filled}, and {leaked} of those {filled} are poison")


if __name__ == "__main__":
    warmup()
    part_a()
    enc = load()
    part_b(enc)
    part_c(enc)
    print("\nStep 4.2 checks finished.")
