"""Synthetic retrieved pool for timing runs (Stage 3.2 and Stage 3.5 gates).

The text is made of pseudo-words with a word-frequency skew like real prose. It
is for timing only: it says nothing about detection, and its embeddings are not
those of real documents.
"""
import numpy as np

POOL_SIZE = 50         # top of the k' = 30..50 over-retrieval range
VOCAB_SIZE = 5000
N_POISON = 5           # PoisonedRAG default: 5 near-copies of one claim


def make_pool(words, seed=0):
    """Text with a word-frequency skew like real prose (Zipf), plus N_POISON
    near-copies, so the pool has the shared words a real pool would have."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, VOCAB_SIZE + 1)
    weights /= weights.sum()

    def sentence(n):
        return rng.choice(VOCAB_SIZE, size=n, p=weights)

    pool = [" ".join(f"w{i}" for i in sentence(words)) for _ in range(POOL_SIZE - N_POISON)]
    base = sentence(words)
    for _ in range(N_POISON):
        copy = base.copy()
        edits = rng.integers(0, words, words // 10)
        copy[edits] = sentence(len(edits))
        pool.append(" ".join(f"w{i}" for i in copy))
    return pool
