"""layer1_scan checks: both signals through one call.

Part A - logic checks on synthetic vectors (no model).
Part B - real model: calibrate once, then scan three pools:
         clean only, stuffed poison mixed in, hand-written paraphrase poison,
         and stuffed poison as the majority of the pool.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.layers.layer1_scan import layer1_scan  # noqa: E402
from trustrag_guard.layers.over_alignment import (  # noqa: E402
    clean_retrieval_sims,
    fit_threshold,
)
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from test_over_alignment import REF_KB, REF_QUERIES, _vec_with_sim  # noqa: E402


# ---------------------------------------------------------------- Part A
def part_a():
    rng = np.random.default_rng(1)
    q = np.zeros(32); q[0] = 1.0

    # over-aligned chunks present, no cluster structure -> flagged by signal A only
    sims = (0.30, 0.35, 0.40, 0.45, 0.90)
    chunks = np.array([_vec_with_sim(rng, q, s) for s in sims])
    r = layer1_scan(q, chunks, threshold=0.6)
    assert r.alignment.flagged.tolist() == [False] * 4 + [True]
    assert r.flagged.tolist() == (r.clustering.suspect | r.alignment.flagged).tolist()
    assert r.flagged[4]
    print("[ok] over-aligned chunk flagged through the combined scan")

    # a real second cluster but nothing over-aligned -> flagged by clustering only
    a = np.zeros(32); a[1] = 1.0
    b = np.zeros(32); b[2] = 1.0
    noise = lambda c, n: c + 0.05 * rng.standard_normal((n, 32))
    chunks = np.vstack([noise(a, 8), noise(b, 3)])
    r = layer1_scan(q, chunks, threshold=0.6)
    assert not r.alignment.flagged.any()
    assert r.clustering.split and r.flagged.sum() == 3
    print("[ok] separate cluster flagged through the combined scan")

    # clean uniform pool -> nothing flagged
    chunks = np.array([_vec_with_sim(rng, q, s) for s in (0.30, 0.32, 0.35, 0.38, 0.40, 0.42)])
    r = layer1_scan(q, chunks, threshold=0.6)
    assert not r.flagged.any()
    print("[ok] clean pool: nothing flagged")


# ---------------------------------------------------------------- Part B
def part_b():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return

    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True)
    thr = fit_threshold(clean_retrieval_sims(enc(REF_QUERIES), enc(REF_KB), top_k=5))
    case = OPENAI_CEO
    q = enc([case.query])[0]
    print(f"\nthreshold (same clean reference as before) = {thr:.3f}")

    clean = [doc.text for doc in case.documents if not doc.is_poison]
    stuffed = [f"{case.query} {doc.text}" for doc in case.documents
               if doc.attack_type in ("clumping", "dispersed")]
    paraphrase = [doc.text for doc in case.documents if doc.is_poison]
    # all 5 poison chunks stuffed: poison is now the MAJORITY of the pool, which
    # breaks clustering's "smaller group = poison" rule.
    stuffed_all = [f"{case.query} {t}" for t in paraphrase]

    pools = {
        "clean only": (clean, []),
        "clean + stuffed poison": (clean + stuffed, stuffed),
        "clean + paraphrase poison": (clean + paraphrase, paraphrase),
        "clean + 5 stuffed poison (majority)": (clean + stuffed_all, stuffed_all),
    }
    for name, (texts, poison) in pools.items():
        r = layer1_scan(q, enc(texts), thr)
        is_poison = np.array([t in poison for t in texts])
        print(f"\n--- {name} ---  silhouette {r.clustering.silhouette:.3f}, "
              f"cluster split: {r.clustering.split}")
        print("   sim  A     clus  poison  text")
        for i in np.argsort(-r.alignment.sims):
            print(f" {r.alignment.sims[i]:.3f}  "
                  f"{'FLAG' if r.alignment.flagged[i] else '-':<5} "
                  f"{'FLAG' if r.clustering.suspect[i] else '-':<5} "
                  f"{'yes' if is_poison[i] else 'no':<7} {texts[i][:52]}")
        print(f"caught {int((r.flagged & is_poison).sum())}/{int(is_poison.sum())} poison, "
              f"wrongly flagged {int((r.flagged & ~is_poison).sum())} clean")


if __name__ == "__main__":
    part_a()
    part_b()
    print("\nlayer1_scan checks finished.")