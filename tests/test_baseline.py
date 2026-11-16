"""Stage 6 checks: the TrustRAG baseline port, on synthetic data."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from baseline_trustrag import conflict_answer, group_n_gram_filtering, k_mean_filtering, rouge  # noqa: E402


def part_a():
    assert rouge("the cat sat", "the cat sat") == 1.0 and rouge("alpha beta", "gamma delta") == 0.0
    print("[ok] ROUGE-L matches identical and disjoint text")

    rng = np.random.default_rng(0)
    plain = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota", "kappa lambda mu", "nu xi omicron"]
    embs = rng.normal(size=(5, 32))
    assert k_mean_filtering(embs, plain) == plain
    print("[ok] the n-gram gate returns everything when no pair overlaps")

    copies = [f"tim cook runs the company today version {i}" for i in range(3)]
    mixed = copies + ["alpha beta gamma delta", "epsilon zeta eta theta"]
    base = rng.normal(size=32)
    tight = base + 0.01 * rng.normal(size=(3, 32))
    others = rng.normal(size=(2, 32))
    kept = k_mean_filtering(np.vstack([tight, others]), mixed)
    assert not set(copies) & set(kept), kept
    assert set(kept) == set(mixed[3:]), kept
    print("[ok] a tight group of near-copies is removed and the rest stays")

    assert group_n_gram_filtering(copies) >= {0, 1}
    assert group_n_gram_filtering(plain) == set()
    print("[ok] the group n-gram filter deletes overlapping text only")

    seen = []

    def llm(prompt):
        seen.append(prompt)
        return f"step {len(seen)}"

    out = conflict_answer("who runs it", ["doc one"], llm)
    assert out == "step 3" and len(seen) == 3
    assert "Externally Retrieved Document0:doc one" in seen[1] and "Memorized Documents:step 1" in seen[1]
    assert "External information: step 2" in seen[2] and "Your own knowledge: step 1" in seen[2]
    print("[ok] the conflict stage chains its three prompts")


if __name__ == "__main__":
    part_a()
    print("\nBaseline checks finished.")
