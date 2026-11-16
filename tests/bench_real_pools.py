"""Measurement: the first real pools.

For each of PoisonedRAG's 100 NQ questions, it builds a pool of k' = 30 in three
versions: clean, poison as published (question removed), and poison with the
question put in front. Poison competes with the clean passages by cosine, as if it
were stored, so the Chroma store is not changed.

Signal A's threshold and clean median are refitted on 200 other questions. Weights,
band, tau, k_min and the cap are the unchanged placeholders. There is no pass mark
for detection, and no generator runs, so this is not an attack success rate.

Run tests/build_nq_store.py once first. Needs Ollama for the verifier.
"""
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_verifier_length import power_state  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP  # noqa: E402
from trustrag_guard.layers.clustering import silhouette_split_test  # noqa: E402
from trustrag_guard.layers.over_alignment import fit_threshold  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import MockVerifier, OllamaVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, open_store  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "nq")
LIMIT_MS = 150.0
DISCARD_ROUNDS = 20
TIMED_ROUNDS = 500
REFERENCE_TOP_K = 5


def retrieve(store, query_emb, k):
    hits = store.query(query_embeddings=[query_emb.tolist()], n_results=k,
                       include=["documents", "metadatas", "embeddings"])
    embs = np.asarray(hits["embeddings"][0], dtype=np.float64)
    sources = [m["source"] for m in hits["metadatas"][0]]
    return list(hits["ids"][0]), list(hits["documents"][0]), sources, embs


def build_pool(ids, texts, sources, embs, q, poison_texts, poison_embs):
    """Top k' by cosine of clean hits plus poison. Each poison chunk gets its own
    source label, which is the kindest case for the attacker."""
    n = len(poison_texts)
    all_ids = ids + [f"poison{i}" for i in range(n)]
    all_texts = texts + poison_texts
    all_sources = sources + [f"attacker{i}" for i in range(n)]
    all_embs = np.vstack([embs, poison_embs]) if n else embs
    top = np.argsort(-(all_embs @ q), kind="stable")[:DEFAULT_OVER_RETRIEVE]
    return ([all_ids[i] for i in top], [all_texts[i] for i in top],
            [all_sources[i] for i in top], all_embs[top])


def score(query, q, gold, pool, verifier, threshold, median):
    ids, texts, sources, embs = pool
    r = asyncio.run(guard(query, q, embs, texts, sources, verifier, threshold, median))
    final = [ids[i] for i in r.floor.selected[:DEFAULT_TARGET_K]]
    poison = np.array([i.startswith("poison") for i in ids])
    suspect = r.fusion.cluster_suspect
    bare = ids[:DEFAULT_TARGET_K]    # an undefended retriever: top k by similarity
    return {
        "gold_actions": [str(a) for i, a in zip(ids, r.fusion.actions) if i in gold],
        "gold_sims": [float(v) for i, v in zip(ids, r.fusion.sims) if i in gold],
        "bare_poison": sum(i.startswith("poison") for i in bare),
        "bare_gold": any(i in gold for i in bare),
        "silhouette": silhouette_split_test(embs).silhouette,
        "poison_pool": int(poison.sum()), "poison_final": sum(i.startswith("poison") for i in final),
        "gold_pool": any(i in gold for i in ids), "gold_final": any(i in gold for i in final),
        "n_final": len(final), "kept": int((r.fusion.actions == KEEP).sum()),
        "ambiguous": int((r.fusion.actions == AMBIGUOUS).sum()),
        "dropped": int((r.fusion.actions == DROP).sum()),
        "poison_dropped": int(((r.fusion.actions == DROP) & poison).sum()),
        "calls": r.verified.n_checked, "low_confidence": bool(r.low_confidence),
        "split": bool(suspect.any()), "suspect_poison": int((suspect & poison).sum()),
        "suspect_clean": int((suspect & ~poison).sum()),
        "signal_poison": int(((r.fusion.sims > threshold) & poison).sum()),
        "signal_clean": int(((r.fusion.sims > threshold) & ~poison).sum()),
    }


def summary(name, rows):
    n = len(rows)
    total = lambda key: sum(r[key] for r in rows)
    print(f"\n--- {name} ({n} questions) ---")
    print(f" poison retrieved: {total('poison_pool')} of {5 * n if 'clean' not in name else 0}")
    print(f" undefended top {DEFAULT_TARGET_K}: poison {total('bare_poison')}, questions with gold "
          f"{sum(r['bare_gold'] for r in rows)}")
    print(f" poison in final top {DEFAULT_TARGET_K}: {total('poison_final')}; "
          f"questions with any: {sum(r['poison_final'] > 0 for r in rows)}")
    print(f" poison dropped by fusion: {total('poison_dropped')}")
    print(f" signal A flagged: poison {total('signal_poison')}, clean {total('signal_clean')}")
    print(f" gold retrieved: {sum(r['gold_pool'] for r in rows)}; gold in final top {DEFAULT_TARGET_K}: "
          f"{sum(r['gold_final'] for r in rows)}")
    print(f" fusion kept at least k_min: {sum(r['kept'] >= 2 for r in rows)}; median kept "
          f"{np.median([r['kept'] for r in rows]):.0f}; median ambiguous {np.median([r['ambiguous'] for r in rows]):.0f}; "
          f"median dropped {np.median([r['dropped'] for r in rows]):.0f}")
    print(f" verifier calls: {total('calls')}; questions with any: {sum(r['calls'] > 0 for r in rows)}")
    print(f" low confidence: {sum(r['low_confidence'] for r in rows)}; empty context: "
          f"{sum(r['n_final'] == 0 for r in rows)}")
    actions = [a for r in rows for a in r["gold_actions"]]
    print(f" gold passages by fusion action: keep {actions.count(KEEP)}, ambiguous {actions.count(AMBIGUOUS)}, "
          f"drop {actions.count(DROP)}; median similarity {np.median([v for r in rows for v in r['gold_sims']]):.3f}")
    sil = [r["silhouette"] for r in rows]
    print(f" silhouette: median {np.median(sil):.3f}, highest {max(sil):.3f} (tau 0.25)")
    print(f" clustering split: {sum(r['split'] for r in rows)}; its suspects hold poison "
          f"{total('suspect_poison')}, clean {total('suspect_clean')}")


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print(f"power at start: {power_state()}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True).astype(np.float64)
    store = open_store(os.path.join(DATA, "chroma"), "nq_clean")
    data = json.load(open(os.path.join(DATA, "questions.json"), encoding="utf-8"))
    print(f"store: {store.count()} passages, targets {len(data['targets'])}, reference {len(data['reference'])}")
    warmup()

    sims = []
    for row in data["reference"]:
        q = enc([row["question"]])[0]
        sims += (retrieve(store, q, REFERENCE_TOP_K)[3] @ q).tolist()
    threshold, median = fit_threshold(sims), float(np.median(sims))
    print(f"refitted on {len(sims)} clean values: threshold {threshold:.3f}, clean median {median:.3f}")

    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()
    rows = {"clean": [], "poison as published": [], "poison with the question": []}
    for t in data["targets"]:
        question = t["question"]
        q = enc([question])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        plain = t["adv_texts"]
        stuffed = [f"{question} {x}" for x in plain]
        for name, poison in (("clean", []), ("poison as published", plain), ("poison with the question", stuffed)):
            pool = build_pool(ids, texts, sources, embs, q, poison, enc(poison) if poison else None)
            rows[name].append(score(question, q, set(t["gold"]), pool, verifier, threshold, median))

    for name, part in rows.items():
        summary(name, part)
    json.dump(rows, open(os.path.join(DATA, "real_pools.json"), "w", encoding="utf-8"))

    # The first true fast-path gate: everything online except the verifier.
    loop = asyncio.new_event_loop()
    quiet = MockVerifier()
    questions = [t["question"] for t in data["targets"]]

    def one_round(question):
        begin = time.perf_counter()
        q = enc([question])[0]
        embedded = time.perf_counter()
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        got = time.perf_counter()
        loop.run_until_complete(guard(question, q, embs, texts, sources, quiet, threshold, median, max_calls=0))
        done = time.perf_counter()
        return (embedded - begin) * 1000, (got - embedded) * 1000, (done - got) * 1000, (done - begin) * 1000

    for i in range(DISCARD_ROUNDS):
        one_round(questions[i % len(questions)])
    timed = np.array([one_round(questions[i % len(questions)]) for i in range(TIMED_ROUNDS)])
    print(f"\n--- fast path on real pools, {TIMED_ROUNDS} rounds, no verifier ---")
    print(" part                 p50      p95      p99")
    for j, name in enumerate(("query embedding", "Chroma retrieval", "layers + floor", "whole fast path")):
        p50, p95, p99 = np.percentile(timed[:, j], [50, 95, 99])
        print(f" {name:<18} {p50:6.1f}   {p95:6.1f}   {p99:6.1f} ms")
    p99 = np.percentile(timed[:, 3], 99)
    print(f"\nverdict: {'PASS' if p99 <= LIMIT_MS else 'FAIL'} (p99 {p99:.1f} ms vs limit {LIMIT_MS} ms)")
    print(f"power at end: {power_state()}")
    print("\nReal-pool measurement finished.")
