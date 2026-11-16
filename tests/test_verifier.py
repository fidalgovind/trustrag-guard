"""Checks for the verifier interface, with a scripted stand-in.

Part A - hand-made fusion results (no model): only ambiguous chunks are sent, and
         the verdict decides them.
Part B - real embeddings: how many chunks reach the verifier on each toy pool.
Part C - real embeddings: best and worst case for Layer 3, using scripted answers.

The mock is not a detector. Part C shows what depends on the real verifier, and
no number here says how well a real model will do.
"""
import asyncio
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP, FusionResult  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import (  # noqa: E402
    SAFE,
    UNSAFE,
    VERIFIED,
    MockVerifier,
    Verdict,
    Verifier,
    verify_ambiguous,
)

# Measured Stage 2 values, not fitted or tuned here.
THRESHOLD = 0.831
MEDIAN = 0.612
TEXTS = ["keep me", "ask me", "drop me", "ask me too"]
QUERY = "a question"


def fusion(actions):
    n = len(actions)
    zeros = np.zeros(n)
    return FusionResult(zeros, np.array(actions, dtype=object), zeros, zeros, zeros, np.zeros(n, bool))


# ---------------------------------------------------------------- Part A
def part_a():
    assert isinstance(MockVerifier(), Verifier)
    print("[ok] MockVerifier satisfies the Verifier protocol")

    mixed = fusion([KEEP, AMBIGUOUS, DROP, AMBIGUOUS])
    mock = MockVerifier()
    r = verify_ambiguous(QUERY, TEXTS, mixed, mock)
    assert mock.calls == [(QUERY, "ask me"), (QUERY, "ask me too")], mock.calls
    assert r.n_checked == 2 and r.verdicts[0] is None and r.verdicts[2] is None
    print("[ok] only ambiguous chunks reach the verifier, with the query")

    assert r.actions.tolist() == [KEEP, VERIFIED, DROP, VERIFIED], r.actions
    r = verify_ambiguous(QUERY, TEXTS, mixed, MockVerifier(unsafe=["ask me"]))
    assert r.actions.tolist() == [KEEP, DROP, DROP, VERIFIED], r.actions
    print("[ok] SAFE marks an ambiguous chunk verified, not kept, and UNSAFE drops it")

    r = verify_ambiguous(QUERY, TEXTS, mixed, MockVerifier(unsafe=["keep me"]))
    assert r.actions[0] == KEEP
    r = verify_ambiguous(QUERY, TEXTS, fusion([KEEP, KEEP, DROP, KEEP]), MockVerifier())
    assert r.actions[2] == DROP and r.n_checked == 0
    print("[ok] keep and drop decisions are never reopened")

    for bad in ("safe", "SAFE ", "", "MAYBE"):
        r = verify_ambiguous(QUERY, TEXTS, mixed, MockVerifier(default=bad))
        assert r.actions.tolist() == [KEEP, DROP, DROP, DROP], (bad, r.actions)
    print("[ok] anything but an exact SAFE drops the chunk")

    class Broken:
        def verify(self, query, chunk):
            raise ConnectionError("verifier is down")

    try:
        verify_ambiguous(QUERY, TEXTS, mixed, Broken())
        raise AssertionError("expected ConnectionError")
    except ConnectionError:
        print("[ok] a failing verifier stops the call, so nothing passes unchecked")

    before = mixed.actions.tolist()
    r = verify_ambiguous(QUERY, TEXTS, mixed, MockVerifier(logprob=-0.25))
    assert mixed.actions.tolist() == before and AMBIGUOUS not in r.actions.tolist()  # no limits set
    assert r.verdicts[1] == Verdict(SAFE, -0.25)
    print("[ok] fusion is not modified, nothing stays ambiguous, logprob is kept")

    try:
        verify_ambiguous(QUERY, TEXTS[:-1], mixed, MockVerifier())
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[ok] mismatched texts are refused")

    r = verify_ambiguous(QUERY, [], fusion([]), MockVerifier())
    assert r.actions.shape == (0,) and r.n_checked == 0
    print("[ok] empty pool: no crash")

    # the call limits
    five = ["a", "b", "c", "d", "e"]
    pool = fusion([AMBIGUOUS] * 5)
    pool.risk = np.array([0.7, 0.3, 0.6, 0.4, 0.5])

    mock = MockVerifier()
    verify_ambiguous(QUERY, five, pool, mock)
    assert [c for _, c in mock.calls] == ["b", "d", "e", "c", "a"], mock.calls
    print("[ok] the lowest-risk chunk is checked first")

    mock = MockVerifier()
    r = verify_ambiguous(QUERY, TEXTS, fusion([KEEP, KEEP, AMBIGUOUS, AMBIGUOUS]), mock, k_min=2)
    assert mock.calls == [] and r.n_checked == 0
    assert r.actions.tolist() == [KEEP, KEEP, AMBIGUOUS, AMBIGUOUS], r.actions
    print("[ok] no verifier call when fusion already kept k_min chunks")

    mock = MockVerifier()
    r = verify_ambiguous(QUERY, five, pool, mock, k_min=2)
    assert r.n_checked == 2 and r.actions.tolist().count(VERIFIED) == 2
    print("[ok] checking stops once the floor can be filled")

    mock = MockVerifier(unsafe=["b", "d", "e"])
    r = verify_ambiguous(QUERY, five, pool, mock, k_min=2, max_calls=3)
    assert r.n_checked == 3 and r.actions.tolist().count(DROP) == 3
    assert r.actions.tolist().count(AMBIGUOUS) == 2 and r.verdicts[0] is None
    print("[ok] the cap holds, and unchecked chunks stay ambiguous")


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
    case = OPENAI_CEO
    clean = [d.text for d in case.documents if not d.is_poison]
    paraphrase = [d.text for d in case.documents if d.is_poison]
    four = [f"{case.query} {d.text}" for d in case.documents
            if d.attack_type in ("clumping", "dispersed")]
    five = [f"{case.query} {t}" for t in paraphrase]
    return (("clean only", clean, []), ("clean + 4 stuffed", clean, four),
            ("clean + 5 paraphrase", clean, paraphrase),
            ("clean + 5 stuffed (majority)", clean, five))


def scanned(enc):
    q = enc([OPENAI_CEO.query])[0]
    for name, clean, poison in pools():
        texts = clean + poison
        yield name, texts, poison, asyncio.run(scan_parallel(q, enc(texts), texts, THRESHOLD, MEDIAN))


def part_b(enc):
    if enc is None:
        return
    print("\n--- verifier load (DESIGN.md expects 0-3 chunks per query) ---")
    print(" pool                           chunks  sent  clean sent  poison sent")
    for name, texts, poison, f in scanned(enc):
        mock = MockVerifier()
        r = verify_ambiguous(OPENAI_CEO.query, texts, f, mock)
        sent = [chunk for _, chunk in mock.calls]
        assert r.n_checked == len(sent) == int((f.actions == AMBIGUOUS).sum())
        bad = sum(t in poison for t in sent)
        print(f" {name:<30} {len(texts):>6}  {len(sent):>4}  {len(sent) - bad:>10}  {bad:>11}")
    print("[ok] the verifier load equals fusion's ambiguous count on every pool")


def part_c(enc):
    if enc is None:
        return
    print("\n--- scripted answers: what depends on Layer 3 ---")
    print(" pool                           verifier       poison left  clean left")
    for name, texts, poison, f in scanned(enc):
        clean = [t for t in texts if t not in poison]
        for label, mock in (("always right", MockVerifier(unsafe=poison)),
                            ("always SAFE", MockVerifier()),
                            ("always UNSAFE", MockVerifier(default=UNSAFE))):
            r = verify_ambiguous(OPENAI_CEO.query, texts, f, mock)
            kept = [t for t, a in zip(texts, r.actions) if a != DROP]
            print(f" {name:<30} {label:<14} {sum(t in poison for t in kept):>7} of {len(poison)}"
                  f"  {sum(t in clean for t in kept):>6} of {len(clean)}")


if __name__ == "__main__":
    warmup()
    part_a()
    enc = load()
    part_b(enc)
    part_c(enc)
    print("\nStep 4.1 checks finished.")
