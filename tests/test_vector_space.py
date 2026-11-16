"""Stage 1 tests: the embedding + cosine 'ruler' behaves sensibly.

Run it directly (prints a readable report):
    python tests/test_vector_space.py
or with pytest once installed:
    pytest tests/test_vector_space.py

There are two kinds of checks here:
  * test_cosine_math_synthetic  -> pure numpy, no model, always fast.
  * test_ruler_on_toy_corpus    -> loads all-MiniLM-L6-v2 (downloads once).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the package importable when run as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trustrag_guard.embeddings import cosine_matrix, cosine  # noqa: E402


def test_cosine_math_synthetic() -> None:
    """cosine_matrix on hand-made normalized vectors gives known answers."""
    # Three unit vectors: v0 and v1 identical, v2 orthogonal to both.
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    m = cosine_matrix(v)
    assert m.shape == (3, 3)
    assert np.isclose(m[0, 0], 1.0)          # a vector matches itself
    assert np.isclose(m[0, 1], 1.0)          # identical vectors -> 1
    assert np.isclose(m[0, 2], 0.0)          # orthogonal -> 0
    # Opposite vectors -> -1
    assert np.isclose(cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])), -1.0)
    print("[ok] synthetic cosine math")


def test_ruler_on_toy_corpus() -> None:
    """On the toy corpus, similarity should line up with meaning."""
    from trustrag_guard.embeddings import embed, query_similarities
    from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO

    case = OPENAI_CEO
    docs = case.texts()
    labels = case.poison_labels()

    # Query->document similarity: this is the signal A preview.
    sims = query_similarities(case.query, docs)

    print(f"\nQuery: {case.query!r}")
    print(f"(truth = {case.correct_answer}, attack target = {case.target_answer})\n")
    print(f"{'sim':>6}  {'poison':>6}  {'type':<10} text")
    for sim, doc, is_poison in sorted(
        zip(sims, case.documents, labels), key=lambda t: -t[0]
    ):
        tag = "POISON" if is_poison else "clean"
        print(f"{sim:6.3f}  {tag:>6}  {doc.attack_type:<10} {doc.text[:60]}")

    # Sanity: two paraphrases of the same clean fact are highly similar,
    # and a clean fact is far less similar to an unrelated sentence.
    emb = embed([
        "Sam Altman is the CEO of OpenAI.",
        "OpenAI's chief executive is Sam Altman.",
        "The Eiffel Tower is located in Paris.",
    ])
    para = cosine(emb[0], emb[1])
    unrelated = cosine(emb[0], emb[2])
    print(f"\nparaphrase similarity : {para:.3f}")
    print(f"unrelated similarity  : {unrelated:.3f}")
    assert para > 0.6, f"paraphrases should be similar, got {para:.3f}"
    assert unrelated < para, "unrelated text should be less similar than a paraphrase"
    print("[ok] toy-corpus ruler")


if __name__ == "__main__":
    test_cosine_math_synthetic()
    test_ruler_on_toy_corpus()
    print("\nAll Stage 1 checks passed.")
