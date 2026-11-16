"""Layer 2 - ROUGE-L and the coordinated-payload flag.

Layer 1 compares meaning (embeddings). This compares the words themselves:
ROUGE-L is built on the Longest Common Subsequence (LCS), the longest run of
tokens that appear in both chunks in the same order, gaps allowed. Chunks that
were generated or copied together share long ordered runs even when their
embeddings look ordinary.

coordination_test() turns those scores into one graded score per chunk: how
close is this chunk, in its text, to its nearest neighbour in the pool? No
prefilter is used: pairwise_rouge_l() scores every pair exactly and met the
Layer 2 time limit without one.

Why the code looks the way it does (DESIGN.md section 4):
  - Numba compiles on the FIRST call, which takes seconds and would wreck p99,
    so warmup() must run at start-up.
  - The compiled functions work on int32 arrays, so chunks are tokenized once
    and passing other dtypes would trigger a fresh compile. Use tokens_to_ids().
  - nogil=True lets this run in parallel with Layer 1.

Tokenization and the F-measure match the rouge_score package with
use_stemmer=False, and tests/test_text_space.py cross-checks that.

Known limits: rephrasing with different words lowers the score, because LCS
only matches identical tokens. Honest duplicates (mirrored pages, boilerplate)
score high too, so this flag alone must not decide that a chunk is poison.
"""
import re
from dataclasses import dataclass

import numpy as np
from numba import njit

_TOKEN = re.compile(r"[a-z0-9]+")

# Flag a chunk that shares at least half its text, in order, with another chunk.
# Starting guess, chosen without looking at toy results; Stage 5 calibrates it.
DEFAULT_COPY_CUTOFF = 0.5


def tokenize(text):
    """Lowercase, then keep runs of a-z and 0-9 (the rouge_score default)."""
    return _TOKEN.findall(text.lower())


def tokens_to_ids(tokens, vocab):
    """Map tokens to int32 ids. `vocab` is a dict shared across chunks, so the
    same word always gets the same id; new words are added to it."""
    return np.array([vocab.setdefault(t, len(vocab)) for t in tokens], dtype=np.int32)


@njit(nogil=True, cache=True)
def lcs_length(a, b):
    """Length of the longest common subsequence of two int32 arrays.

    Classic dynamic programme, keeping only two rows, so memory is O(min(n, m)).
    """
    if len(a) < len(b):
        a, b = b, a
    n = len(b)
    prev = np.zeros(n + 1, dtype=np.int32)
    curr = np.zeros(n + 1, dtype=np.int32)
    for i in range(len(a)):
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev, curr = curr, prev
    return int(prev[n])


@njit(nogil=True, cache=True)
def rouge_l_f(a, b):
    """ROUGE-L F-measure of two int32 arrays, in [0, 1]. Empty input gives 0.

    With precision = lcs / len(a) and recall = lcs / len(b), the F-measure
    2pr / (p + r) simplifies to 2 * lcs / (len(a) + len(b)), which is symmetric.
    """
    total = len(a) + len(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return 2.0 * lcs_length(a, b) / total


@njit(nogil=True, cache=True)
def _pairwise_rouge_l(flat, offsets, vocab_size):
    """All-pairs ROUGE-L F-measure, using bit-parallel LCS (Crochemore et al.,
    Hyyro). Each machine word updates 64 cells of the LCS table at once, which
    is about 64 times less work than lcs_length() and just as exact.
    """
    n = len(offsets) - 1
    out = np.zeros((n, n), dtype=np.float64)
    longest = int(np.max(offsets[1:] - offsets[:-1]))
    capacity = max((longest + 63) // 64, 1)
    masks = np.zeros((vocab_size, capacity), dtype=np.uint64)  # token -> positions in b
    v = np.zeros(capacity, dtype=np.uint64)
    one = np.uint64(1)
    zero = np.uint64(0)

    for i in range(n):
        b = flat[offsets[i]:offsets[i + 1]]
        m = len(b)
        words = (m + 63) // 64
        for p in range(m):
            masks[b[p], p // 64] |= one << np.uint64(p % 64)

        for j in range(i + 1, n):
            a = flat[offsets[j]:offsets[j + 1]]
            if m == 0 or len(a) == 0:
                continue
            v[:words] = ~zero
            for t in range(len(a)):
                row = masks[a[t]]
                carry = zero
                for w in range(words):
                    u = v[w] & row[w]
                    total = v[w] + u
                    carry_out = one if total < v[w] else zero
                    total_in = total + carry
                    if total_in < total:
                        carry_out = one
                    v[w] = total_in | (v[w] - u)
                    carry = carry_out

            lcs = 0                       # LCS = zero bits of v within m positions
            for w in range(words):
                z = ~v[w]
                if w == words - 1 and m % 64 != 0:
                    z &= (one << np.uint64(m % 64)) - one
                while z != zero:
                    z &= z - one
                    lcs += 1
            out[i, j] = out[j, i] = 2.0 * lcs / (m + len(a))

        for p in range(m):
            masks[b[p]] = zero
    return out


def pairwise_rouge_l(id_arrays):
    """Symmetric (n, n) matrix of ROUGE-L F for chunks from tokens_to_ids(),
    which must share one vocab. Same values as rouge_l_f() on each pair. The
    diagonal is 0 because a chunk is not compared with itself."""
    n = len(id_arrays)
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(x) for x in id_arrays])
    flat = np.concatenate(id_arrays).astype(np.int32)
    vocab_size = int(flat.max()) + 1 if len(flat) else 0
    return _pairwise_rouge_l(flat, offsets, vocab_size)


@dataclass
class CoordinationResult:
    scores: np.ndarray     # per chunk: highest ROUGE-L with any other chunk, in [0, 1]
    cutoff: float
    flagged: np.ndarray    # bool per chunk; True = scores >= cutoff


def coordination_test(texts, cutoff=DEFAULT_COPY_CUTOFF):
    """Online step: score each chunk by its closest text match in the pool.

    The graded `scores` are what fusion should use; `flagged` is a convenience.
    Fewer than two chunks gives scores of 0, so nothing is flagged.
    """
    vocab = {}
    matrix = pairwise_rouge_l([tokens_to_ids(tokenize(t), vocab) for t in texts])
    scores = matrix.max(axis=1) if len(texts) else np.zeros(0, dtype=np.float64)
    return CoordinationResult(scores, float(cutoff), scores >= cutoff)


def warmup():
    """Compile the Numba functions. Call once at start-up, before serving."""
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([2, 3, 4], dtype=np.int32)
    lcs_length(a, b)
    rouge_l_f(a, b)
    pairwise_rouge_l([a, b])
