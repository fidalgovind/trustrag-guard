"""Checks for query-conditional over-alignment (signal A).

Part A - logic checks on synthetic vectors (no model, runs in a second).
Part B - real toy corpus with MiniLM, calibrated on a small CLEAN reference,
         compared side by side with the clustering baseline.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.layers.clustering import silhouette_split_test  # noqa: E402
from trustrag_guard.layers.over_alignment import (  # noqa: E402
    clean_retrieval_sims,
    fit_threshold,
    over_alignment_test,
)
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402


# ---------------------------------------------------------------- Part A
def _vec_with_sim(rng, q, s):
    """Unit vector whose cosine with unit vector q is exactly s."""
    r = rng.standard_normal(len(q))
    r -= (r @ q) * q
    r /= np.linalg.norm(r)
    return s * q + np.sqrt(1 - s * s) * r


def part_a():
    rng = np.random.default_rng(0)
    q = np.zeros(32); q[0] = 1.0

    # fit_threshold == the requested quantile of the clean sample
    clean = np.linspace(0.2, 0.6, 101)
    assert abs(fit_threshold(clean, 0.99) - 0.596) < 1e-9
    print("[ok] fit_threshold returns the clean quantile")

    # over-aligned chunks flagged, ordinary chunks not
    chunks = np.array([_vec_with_sim(rng, q, s) for s in (0.35, 0.45, 0.55, 0.9, 0.95)])
    r = over_alignment_test(q, chunks, threshold=0.6)
    assert list(r.flagged) == [False, False, False, True, True], r.sims
    assert np.allclose(r.sims, [0.35, 0.45, 0.55, 0.9, 0.95])
    print("[ok] chunks above the clean threshold flagged, others kept")

    # clean_retrieval_sims keeps the best top_k per query, pooled
    kb = np.array([_vec_with_sim(rng, q, s) for s in (0.1, 0.5, 0.7, 0.3)])
    pooled = clean_retrieval_sims(q[None, :], kb, top_k=2)
    assert np.allclose(sorted(pooled), [0.5, 0.7])
    print("[ok] clean retrieval simulation pools the top-k sims")

    # empty reference is an error, not a silent threshold
    try:
        fit_threshold([])
        raise AssertionError("expected ValueError")
    except ValueError:
        print("[ok] empty reference rejected")


# ---------------------------------------------------------------- Part B
# Small CLEAN reference: 8 queries over a clean mini knowledge base. None of
# these are about OpenAI. Stage 5 replaces this with a real clean corpus.
REF_QUERIES = [
    "Who founded Microsoft?",
    "Where is the Eiffel Tower located?",
    "What is the capital of Australia?",
    "Who wrote Pride and Prejudice?",
    "When did the Berlin Wall fall?",
    "What is the boiling point of water at sea level?",
    "Who is the CEO of Nvidia?",
    "Who discovered penicillin?",
]
REF_KB = [
    "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    "Bill Gates served as the chief executive of Microsoft until 2000.",
    "Microsoft is headquartered in Redmond, Washington.",
    "Microsoft develops the Windows operating system and the Office suite.",
    "The Eiffel Tower stands on the Champ de Mars in Paris, France.",
    "The Eiffel Tower was completed in 1889 for the World's Fair.",
    "Paris is the capital and largest city of France.",
    "Gustave Eiffel's company designed and built the tower.",
    "Canberra is the capital city of Australia.",
    "Sydney is the most populous city in Australia.",
    "Canberra was chosen as the capital as a compromise between Sydney and Melbourne.",
    "Australia is both a country and a continent.",
    "Pride and Prejudice is a novel by Jane Austen published in 1813.",
    "Jane Austen also wrote Emma and Sense and Sensibility.",
    "The novel follows Elizabeth Bennet and Mr Darcy.",
    "Jane Austen was born in Hampshire, England, in 1775.",
    "The Berlin Wall fell on 9 November 1989.",
    "The wall had divided East and West Berlin since 1961.",
    "Germany was formally reunified in October 1990.",
    "Berlin is the capital of Germany.",
    "Water boils at 100 degrees Celsius at sea level.",
    "At higher altitudes water boils at a lower temperature.",
    "Water freezes at 0 degrees Celsius.",
    "Boiling water is commonly used to cook pasta.",
    "Jensen Huang is a co-founder and the chief executive officer of Nvidia.",
    "Nvidia was founded in 1993.",
    "Nvidia designs graphics processing units used in gaming and AI.",
    "Nvidia is headquartered in Santa Clara, California.",
    "Alexander Fleming discovered penicillin in 1928.",
    "Penicillin was developed into a usable drug by Howard Florey and Ernst Chain.",
    "Penicillin is an antibiotic used to treat bacterial infections.",
    "Fleming noticed mould killing bacteria on a culture plate.",
]


def part_b():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return None

    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True)

    # offline: calibrate on CLEAN retrievals only
    ref_sims = clean_retrieval_sims(enc(REF_QUERIES), enc(REF_KB), top_k=5)
    thr = fit_threshold(ref_sims)
    print(f"\nclean reference: {len(ref_sims)} retrieved sims | "
          f"median {np.median(ref_sims):.3f} | 95th {np.quantile(ref_sims, .95):.3f} | "
          f"max {ref_sims.max():.3f}")
    print(f"threshold (clean 99th percentile) = {thr:.3f}")

    # online: score the project's actual toy retrieval
    case = OPENAI_CEO
    kinds = [doc.attack_type for doc in case.documents]
    emb = enc(case.texts())
    q = enc([case.query])[0]
    a = over_alignment_test(q, emb, thr)
    c = silhouette_split_test(emb)

    print(f"\nQuery: {case.query!r}\n")
    print("   sim  signal A  clustering  type       text")
    for i in np.argsort(-a.sims):
        print(f" {a.sims[i]:.3f}  {'FLAG' if a.flagged[i] else '-':<8}  "
              f"{'FLAG' if c.suspect[i] else '-':<10}  "
              f"{kinds[i]:<10} {case.documents[i].text[:50]}")

    poison = {i for i, doc in enumerate(case.documents) if doc.is_poison}
    for name, flagged in (("signal A   ", set(np.where(a.flagged)[0])),
                          ("clustering ", set(np.where(c.suspect)[0]))):
        print(f"\n{name} caught : "
              f"{sorted(case.documents[i].attack_type for i in flagged & poison)}")
        print(f"{name} missed : "
              f"{sorted(case.documents[i].attack_type for i in poison - flagged)}")
        print(f"{name} clean chunks wrongly flagged: {len(flagged - poison)}")
    return enc, thr


# ---------------------------------------------------------------- Part C
# Part B's poison is hand-written paraphrase, never built to win retrieval, so
# it cannot test the over-alignment idea. PoisonedRAG-style poison is built so
# the poison text resembles the target question (black-box: question prepended
# to the false answer text). Same clean chunks, SAME threshold, no re-tuning.
def part_c(ctx):
    if ctx is None:
        return
    enc, thr = ctx
    case = OPENAI_CEO
    clean = [doc.text for doc in case.documents if not doc.is_poison]
    stuffed = [f"{case.query} {doc.text}" for doc in case.documents
               if doc.attack_type in ("clumping", "dispersed")]
    chunks = clean + stuffed
    kinds = ["clean"] * len(clean) + ["stuffed"] * len(stuffed)

    a = over_alignment_test(enc([case.query])[0], enc(chunks), thr)
    print(f"\n--- Part C: query-stuffed poison (threshold stays {thr:.3f}) ---")
    print("   sim  signal A  type     text")
    for i in np.argsort(-a.sims):
        print(f" {a.sims[i]:.3f}  {'FLAG' if a.flagged[i] else '-':<8}  "
              f"{kinds[i]:<8} {chunks[i][:60]}")
    caught = sum(a.flagged[i] for i, k in enumerate(kinds) if k == "stuffed")
    wrong = sum(a.flagged[i] for i, k in enumerate(kinds) if k == "clean")
    print(f"\nsignal A caught {caught}/{len(stuffed)} stuffed poison, "
          f"wrongly flagged {wrong} clean chunks")


if __name__ == "__main__":
    part_a()
    part_c(part_b())
    print("\nStep 2.2 checks finished.")
