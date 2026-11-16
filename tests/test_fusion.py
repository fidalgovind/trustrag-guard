"""Checks for risk fusion v1.

Part A - logic checks on hand-made signals (no model).
Part B - real model: which action each chunk gets, across the toy pools.
Part C - clean-only pools: the same text the coordination flag wrongly flagged.

Parts B and C only measure. Weights and band edges are placeholders, and none
should be tuned on these results.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP, alignment_grade, fuse  # noqa: E402
from trustrag_guard.layers.clustering import SplitResult  # noqa: E402
from trustrag_guard.layers.layer1_scan import Layer1Result, layer1_scan  # noqa: E402
from trustrag_guard.layers.over_alignment import (  # noqa: E402
    AlignmentResult,
    clean_retrieval_sims,
    fit_threshold,
)
from trustrag_guard.layers.text_space import CoordinationResult, coordination_test  # noqa: E402
from test_over_alignment import REF_KB, REF_QUERIES  # noqa: E402

THRESHOLD = 0.8
MEDIAN = 0.6


def _signals(sims, coord, suspect):
    """Build signal results by hand, so each case is controlled exactly."""
    sims = np.array(sims, dtype=np.float64)
    coord = np.array(coord, dtype=np.float64)
    suspect = np.array(suspect, dtype=bool)
    layer1 = Layer1Result(
        SplitResult(0.5, bool(suspect.any()), suspect.astype(int), suspect),
        AlignmentResult(sims, THRESHOLD, sims > THRESHOLD),
        suspect | (sims > THRESHOLD),
    )
    return layer1, CoordinationResult(coord, 0.5, coord >= 0.5)


# ---------------------------------------------------------------- Part A
def part_a():
    assert np.allclose(alignment_grade([0.5, 0.6, 0.7, 0.8, 0.95], THRESHOLD, MEDIAN), [0, 0, 0.5, 1, 1])
    assert np.allclose(alignment_grade([0.7], 0.7, 0.7), [0.0])  # degenerate reference, no crash
    print("[ok] alignment grade: 0 at the clean median, 1 at the threshold")

    r = fuse(*_signals([0.9], [0.0], [False]), MEDIAN)
    assert r.actions.tolist() == [DROP], r.risk
    print("[ok] signal A above its threshold drops a chunk by itself")

    # the safety rule: maximum coordination plus clustering, no query alignment
    r = fuse(*_signals([0.6, 0.6], [1.0, 1.0], [True, False]), MEDIAN)
    assert (r.actions != DROP).all() and (r.risk < 0.85).all(), r.risk
    print("[ok] clustering and coordination cannot drop a chunk, alone or together")

    for bad in ({"w_coordination": 0.9}, {"w_clustering": 0.9}):
        try:
            fuse(*_signals([0.6], [0.0], [False]), MEDIAN, **bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("[ok] weights that break the safety rule are refused")

    r = fuse(*_signals([0.0, 0.75, 1.0], [0.0, 0.5, 1.0], [False, True, True]), MEDIAN)
    assert ((r.risk >= 0) & (r.risk <= 1)).all() and r.actions.tolist() == [KEEP, AMBIGUOUS, DROP], (r.risk, r.actions)
    print("[ok] risk stays in 0-1, and keep / ambiguous / drop follow the band")

    assert r.sims.tolist() == [0.0, 0.75, 1.0] and r.coordination.tolist() == [0.0, 0.5, 1.0]
    assert r.cluster_suspect.tolist() == [False, True, True]
    print("[ok] raw scores are kept in the result")

    try:
        fuse(_signals([0.6, 0.7], [0.0, 0.0], [False, False])[0], CoordinationResult(np.zeros(1), 0.5, np.zeros(1, bool)), MEDIAN)
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[ok] mismatched chunk counts are refused")

    r = fuse(*_signals([], [], []), MEDIAN)
    assert r.risk.shape == (0,)
    print("[ok] empty pool: no crash")


# ---------------------------------------------------------------- Part B
def _report(name, texts, poison, q, enc, thr, median):
    r = fuse(layer1_scan(q, enc(texts), thr), coordination_test(texts), median)
    is_poison = np.array([t in poison for t in texts])
    print(f"\n--- {name} ---")
    print("  risk  action     sim  grade  coord  clus  poison  text")
    for i in np.argsort(-r.risk):
        print(f" {r.risk[i]:.3f}  {r.actions[i]:<9} {r.sims[i]:.3f}  {r.alignment_grade[i]:.2f}   "
              f"{r.coordination[i]:.2f}   {'FLAG' if r.cluster_suspect[i] else '-':<4}  "
              f"{'yes' if is_poison[i] else 'no':<6}  {texts[i][:40]}")
    for label, mask in (("poison", is_poison), ("clean ", ~is_poison)):
        print(f" {label}: " + ", ".join(f"{a} {int(((r.actions == a) & mask).sum())}"
                                        for a in (DROP, AMBIGUOUS, KEEP)))


def load():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return None
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True)
    sims = clean_retrieval_sims(enc(REF_QUERIES), enc(REF_KB), top_k=5)
    return enc, fit_threshold(sims), float(np.median(sims))


def part_b(ctx):
    if ctx is None:
        return
    enc, thr, median = ctx
    case = OPENAI_CEO
    q = enc([case.query])[0]
    print(f"\nthreshold {thr:.3f}, clean median {median:.3f}")

    clean = [d.text for d in case.documents if not d.is_poison]
    paraphrase = [d.text for d in case.documents if d.is_poison]
    stuffed = [f"{case.query} {d.text}" for d in case.documents
               if d.attack_type in ("clumping", "dispersed")]
    stuffed_all = [f"{case.query} {t}" for t in paraphrase]
    for name, poison in (("clean only", []), ("clean + 4 stuffed", stuffed),
                         ("clean + 5 paraphrase", paraphrase),
                         ("clean + 5 stuffed (majority)", stuffed_all)):
        _report(name, clean + poison, poison, q, enc, thr, median)


# ---------------------------------------------------------------- Part C
def part_c(ctx):
    """Each clean query against the clean mini knowledge base. Nothing here is
    poison, so every drop is a false alarm. These are the texts that fitted the
    threshold, so this is an in-sample check and a lower bound on false alarms."""
    if ctx is None:
        return
    enc, thr, median = ctx
    embs = enc(REF_KB)
    drop = ambiguous = keep = 0
    for query in REF_QUERIES:
        q = enc([query])[0]
        sims = embs @ q
        top = np.argsort(-sims)[:20]
        texts = [REF_KB[i] for i in top]
        r = fuse(layer1_scan(q, embs[top], thr), coordination_test(texts), median)
        drop += int((r.actions == DROP).sum())
        ambiguous += int((r.actions == AMBIGUOUS).sum())
        keep += int((r.actions == KEEP).sum())
    print(f"\n--- clean only: {len(REF_QUERIES)} queries, top 20 of the clean KB ---")
    print(f" drop {drop}, ambiguous {ambiguous}, keep {keep}, "
          f"of {drop + ambiguous + keep} clean chunks (in-sample)")


if __name__ == "__main__":
    part_a()
    ctx = load()
    part_b(ctx)
    part_c(ctx)
    print("\nStep 3.4 checks finished.")
