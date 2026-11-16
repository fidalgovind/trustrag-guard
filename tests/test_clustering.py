"""Checks for the clustering baseline.

Part A - logic checks on synthetic vectors (no model, runs in a second).
Part B - the real toy corpus with MiniLM: prints what clustering catches and misses.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.layers.clustering import silhouette_split_test  # noqa: E402


# ---------------------------------------------------------------- Part A
def _blob(rng, center, n, noise):
    return center + noise * rng.standard_normal((n, len(center)))


def part_a():
    rng = np.random.default_rng(0)
    a = np.zeros(16); a[0] = 1.0
    b = np.zeros(16); b[1] = 1.0

    # 1) two clearly separate groups -> split, smaller group flagged
    x = np.vstack([_blob(rng, a, 8, 0.05), _blob(rng, b, 3, 0.05)])
    r = silhouette_split_test(x)
    assert r.split, f"expected a split, silhouette={r.silhouette:.3f}"
    assert list(np.where(r.suspect)[0]) == [8, 9, 10], r.suspect
    print(f"[ok] two groups: split, silhouette={r.silhouette:.3f}, suspect=small group")

    # 2) one uniform blob -> no split, nothing flagged
    x = _blob(rng, a, 12, 0.35)
    r = silhouette_split_test(x)
    assert not r.split and not r.suspect.any(), f"silhouette={r.silhouette:.3f}"
    print(f"[ok] one blob: no split, silhouette={r.silhouette:.3f}")

    # 3) too few chunks -> safe default (no split), no crash
    r = silhouette_split_test(_blob(rng, a, 3, 0.05))
    assert not r.split and not r.suspect.any()
    print("[ok] fewer than 4 chunks: no split")

    # 4) identical chunks -> no crash, no split
    r = silhouette_split_test(np.tile(a, (6, 1)))
    assert not r.split
    print("[ok] identical chunks: no split")


# ---------------------------------------------------------------- Part B
def part_b():
    from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:  # offline / model not downloadable
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return

    case = OPENAI_CEO
    emb = model.encode(case.texts(), normalize_embeddings=True)
    r = silhouette_split_test(emb)

    print(f"\nQuery: {case.query!r}")
    print(f"silhouette (k=2) = {r.silhouette:.3f}  ->  split = {r.split}\n")
    print(" flagged  type       text")
    for i, doc in enumerate(case.documents):
        flag = "YES" if r.suspect[i] else "-"
        print(f" {flag:<8} {doc.attack_type:<10} {doc.text[:60]}")

    poison = {i for i, doc in enumerate(case.documents) if doc.is_poison}
    flagged = set(np.where(r.suspect)[0])
    print(f"\ncaught  : {sorted(case.documents[i].attack_type for i in flagged & poison)}")
    print(f"missed  : {sorted(case.documents[i].attack_type for i in poison - flagged)}")
    print(f"clean chunks wrongly flagged: {len(flagged - poison)}")


if __name__ == "__main__":
    part_a()
    part_b()
    print("\nStep 2.1 checks finished.")
