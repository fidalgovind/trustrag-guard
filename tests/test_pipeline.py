"""Checks for the in-memory pipeline.

Part A - synthetic vectors (no model): order of steps, failure behaviour, and that
         a slow verifier does not block the event loop.
Part B - real embeddings: the pipeline equals running each step by hand.
Part C - real embeddings: what the pipeline returns on the paraphrase pool.

The verifier is scripted, so nothing here measures detection. Timing is in
tests/bench_fast_path.py.
"""
import asyncio
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard import pipeline  # noqa: E402
from trustrag_guard.context_floor import context_floor  # noqa: E402
from trustrag_guard.fast_path import scan_parallel, scan_sequential  # noqa: E402
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import (  # noqa: E402
    SAFE,
    MockVerifier,
    Verdict,
    verify_ambiguous,
)
from trustrag_guard.pipeline import guard  # noqa: E402

# Measured Stage 2 values, not fitted or tuned here.
THRESHOLD = 0.831
MEDIAN = 0.612


def synthetic():
    """Unit vectors with chosen query similarities 0.3, 0.72, 0.72, 0.95: one keep,
    two ambiguous, one drop at the placeholder band (threshold 0.8, median 0.6)."""
    rng = np.random.default_rng(0)
    query = np.zeros(16)
    query[0] = 1.0
    embs = []
    for sim in (0.3, 0.72, 0.72, 0.95):
        side = rng.normal(size=16)
        side[0] = 0.0
        embs.append(sim * query + np.sqrt(1 - sim ** 2) * side / np.linalg.norm(side))
    texts = ["red apple pie", "blue river stone", "green field wind", "gold silver iron"]
    return query, np.array(embs), texts, ["s1", "s2", "s3", "s4"], 0.8, 0.6


# ---------------------------------------------------------------- Part A
def part_a():
    query, embs, texts, sources, thr, med = synthetic()
    args = (query, embs, texts, sources)
    expected = scan_sequential(query, embs, texts, thr, med).actions.tolist()
    assert expected.count(AMBIGUOUS) == 2 and KEEP in expected and DROP in expected, expected

    events = []
    real_scan = pipeline.scan_parallel

    async def recording_scan(*a, **k):
        out = await real_scan(*a, **k)
        events.append("fusion")
        return out

    class Recording(MockVerifier):
        def verify(self, q, chunk):
            events.append("verifier")
            return super().verify(q, chunk)

    pipeline.scan_parallel = recording_scan
    mock = Recording()
    try:
        r = asyncio.run(guard("q", *args, mock, thr, med, k_min=3))
    finally:
        pipeline.scan_parallel = real_scan
    assert events == ["fusion", "verifier", "verifier"], events
    assert [c for _, c in mock.calls] == [t for t, a in zip(texts, expected) if a == AMBIGUOUS]
    print("[ok] the verifier runs after fusion, and only on ambiguous chunks")

    assert r.fusion.sims.shape == (4,) and r.fusion.coordination.shape == (4,)
    assert r.fusion.alignment_grade.shape == (4,) and r.fusion.cluster_suspect.shape == (4,)
    assert Verdict(SAFE, 0.0) in r.verified.verdicts
    print("[ok] raw scores and verdicts stay available for calibration")

    mock = MockVerifier()
    try:
        asyncio.run(guard("q", query, embs, texts, sources[:-1], mock, thr, med))
        raise AssertionError("expected ValueError")
    except ValueError:
        assert mock.calls == []
        print("[ok] mismatched sources are refused before any verifier call")

    class Broken:
        def verify(self, q, chunk):
            raise ConnectionError("verifier is down")

    try:
        asyncio.run(guard("q", *args, Broken(), thr, med))
        raise AssertionError("expected ConnectionError")
    except ConnectionError:
        print("[ok] a failing verifier stops the call, so no chunks are returned")

    # This verifier waits for a task on the event loop, so it finishes only if the
    # pipeline leaves the loop free.
    released = threading.Event()

    class Slow(MockVerifier):
        def verify(self, q, chunk):
            assert released.wait(10), "the event loop was blocked"
            return super().verify(q, chunk)

    async def both():
        async def release():
            await asyncio.sleep(0.05)
            released.set()
        return (await asyncio.gather(guard("q", *args, Slow(), thr, med, k_min=3), release()))[0]

    assert len(asyncio.run(both()).verified.verdicts) == 4
    print("[ok] a slow verifier does not block the event loop")

    mock = MockVerifier(default="UNSAFE")
    r = asyncio.run(guard("q", *args, mock, thr, med, k_min=3, max_calls=1))
    assert len(mock.calls) == 1 and r.low_confidence, r
    assert r.chunks == [texts[expected.index(KEEP)]], r.chunks
    print("[ok] hitting the cap returns fewer chunks and signals low confidence")

    r = asyncio.run(guard("q", np.ones(16), np.zeros((0, 16)), [], [], MockVerifier(), thr, med))
    assert r.chunks == [] and r.low_confidence
    print("[ok] empty pool: no chunks, and low confidence")


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
    clean = [(d.text, d.source) for d in docs if not d.is_poison]
    poison = [(d.text, d.source) for d in docs if d.is_poison]
    stuffed = [(f"{OPENAI_CEO.query} {t}", s) for t, s in poison]
    return (("clean only", clean, []), ("clean + 5 paraphrase", clean, poison),
            ("clean + 5 stuffed (majority)", clean, stuffed))


def part_b(enc):
    if enc is None:
        return
    query = OPENAI_CEO.query
    q = enc([query])[0]
    for name, clean, poison in pools():
        chunks = clean + poison
        texts = [t for t, _ in chunks]
        sources = [s for _, s in chunks]
        embs = enc(texts)
        unsafe = [t for t, _ in poison]
        r = asyncio.run(guard(query, q, embs, texts, sources, MockVerifier(unsafe=unsafe), THRESHOLD, MEDIAN))

        f = asyncio.run(scan_parallel(q, embs, texts, THRESHOLD, MEDIAN))
        v = verify_ambiguous(query, texts, f, MockVerifier(unsafe=unsafe), k_min=2, max_calls=3)
        floor = context_floor(v.actions, sources)
        assert r.chunks == [texts[i] for i in floor.selected], name
        assert r.low_confidence == floor.low_confidence, name
        assert r.verified.actions.tolist() == v.actions.tolist(), name
        assert np.array_equal(r.fusion.risk, f.risk), name
    print("[ok] the pipeline equals running each step by hand on the toy pools")


def part_c(enc):
    if enc is None:
        return
    query = OPENAI_CEO.query
    q = enc([query])[0]
    name, clean, poison = pools()[1]
    chunks = clean + poison
    texts = [t for t, _ in chunks]
    sources = [s for _, s in chunks]
    embs = enc(texts)
    print(f"\n--- what the generator receives: {name}, k_min 2 ---")
    for label, mock in (("always right", MockVerifier(unsafe=[t for t, _ in poison])),
                        ("always SAFE", MockVerifier())):
        r = asyncio.run(guard(query, q, embs, texts, sources, mock, THRESHOLD, MEDIAN))
        print(f" {label}: {len(mock.calls)} verifier calls, low confidence {r.low_confidence}")
        for chunk in r.chunks:
            print(f"   - {chunk}")


if __name__ == "__main__":
    warmup()
    part_a()
    enc = load()
    part_b(enc)
    part_c(enc)
    print("\nStep 4.3 checks finished.")
