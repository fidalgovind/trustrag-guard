"""Measurement: calibrated values against today's.

First the promise: on held-out clean questions, how many top chunks cross each band
edge? Then the real-pool table on the 100 target questions, once with today's values
and once with calibrated ones, each with the SAFE what-if.

The target questions are the test set. Nothing may be refitted after reading this.
No generator runs, so this is not an attack success rate.

Run tests/build_nq_store.py and build_thresholds first. Needs Ollama.
"""
import asyncio
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_real_pools import DATA, REFERENCE_TOP_K, build_pool, retrieve  # noqa: E402
from bench_verifier_length import power_state  # noqa: E402
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.context_floor import context_floor  # noqa: E402
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP, RISK_HIGH, RISK_LOW  # noqa: E402
from trustrag_guard.layers.clustering import DEFAULT_TAU  # noqa: E402
from trustrag_guard.layers.over_alignment import fit_threshold  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import SAFE, OllamaVerifier  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, open_store  # noqa: E402

NAMES = ("clean", "poison as published", "poison with the question")


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print(f"power at start: {power_state()}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True).astype(np.float64)
    store = open_store(os.path.join(DATA, "chroma"), "nq_clean")
    data = json.load(open(os.path.join(DATA, "questions.json"), encoding="utf-8"))
    warmup()

    sims = []
    for row in data["reference"]:
        q = enc([row["question"]])[0]
        sims += (retrieve(store, q, REFERENCE_TOP_K)[3] @ q).tolist()
    today = {"threshold": fit_threshold(sims), "clean_median": float(np.median(sims)),
             "tau": DEFAULT_TAU, "low": RISK_LOW, "high": RISK_HIGH}
    fitted = bt.load()
    print("today:     ", {k: round(today[k], 3) for k in today})
    print("calibrated:", {k: round(fitted[k], 3) for k in today}, "from", fitted["n_scores"], "scores")

    def scan(q, pool, v):
        _, texts, _, embs = pool
        return asyncio.run(scan_parallel(q, embs, texts, v["threshold"], v["clean_median"],
                                         tau=v["tau"], low=v["low"], high=v["high"]))

    _, held = bt.split_questions(data["reference"])
    n = low = high = split = 0
    gold_bare = gold_kept = with_gold = 0
    for row in held:
        q = enc([row["question"]])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        f = scan(q, (ids, texts, sources, embs), fitted)
        top = f.actions[:DEFAULT_TARGET_K]
        n += len(top)
        low += int((top != KEEP).sum())
        high += int((top == DROP).sum())
        split += bool(f.cluster_suspect.any())
        final = [ids[i] for i in context_floor(f.actions, sources).selected[:DEFAULT_TARGET_K]]
        gold = set(row["gold"])
        if gold & set(ids):
            with_gold += 1
            gold_bare += bool(gold & set(ids[:DEFAULT_TARGET_K]))
            gold_kept += bool(gold & set(final))
    print(f"\n--- held-out clean questions ({len(held)}) ---")
    print(f" top chunks reaching the band or above: {low} of {n} ({low / n:.1%}), promised about 5%")
    print(f" top chunks dropped: {high} of {n} ({high / n:.1%}), promised about 1%")
    print(f" pools split: {split} of {len(held)}, promised about 1%")
    print(f" gold in top {DEFAULT_TARGET_K}: {gold_kept} defended, {gold_bare} undefended, of {with_gold} with gold retrieved")

    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()
    asked = {}

    def is_safe(question, text):
        key = (question, text)
        if key not in asked:
            asked[key] = verifier.verify(question, text).label == SAFE
        return asked[key]

    results = {}
    for label, values in (("today", today), ("calibrated", fitted)):
        rows = {n: dict(p_now=0, q_now=0, g_now=0, p_if=0, q_if=0, g_if=0, amb=[], kept=[]) for n in NAMES}
        for t in data["targets"]:
            question, gold = t["question"], set(t["gold"])
            q = enc([question])[0]
            ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
            plain = t["adv_texts"]
            for name, poison in zip(NAMES, ([], plain, [f"{question} {x}" for x in plain])):
                pool = build_pool(ids, texts, sources, embs, q, poison, enc(poison) if poison else None)
                p_ids, p_texts, p_sources, _ = pool
                f = scan(q, pool, values)
                amb = np.flatnonzero(f.actions == AMBIGUOUS)
                safe = {int(i) for i in amb if is_safe(question, p_texts[i])}
                now = [p_ids[int(i)] for i in context_floor(f.actions, p_sources).selected[:DEFAULT_TARGET_K]]
                what_if = [p_ids[i] for i in range(len(p_ids)) if f.actions[i] == KEEP or i in safe][:DEFAULT_TARGET_K]
                r = rows[name]
                r["amb"].append(len(amb))
                r["kept"].append(int((f.actions == KEEP).sum()))
                for tag, chosen in (("now", now), ("if", what_if)):
                    bad = sum(i.startswith("poison") for i in chosen)
                    r[f"p_{tag}"] += bad
                    r[f"q_{tag}"] += bad > 0
                    r[f"g_{tag}"] += any(i in gold for i in chosen)
        results[label] = rows

    print("\nUnits: poison counts chunks out of 500 top-5 slots. Questions and gold count questions out of 100.")
    print(f" top {DEFAULT_TARGET_K}, 100 questions                       poison  questions  gold   poison-if  questions-if  gold-if  median ambiguous")
    for label, rows in results.items():
        for name in NAMES:
            r = rows[name]
            print(f" {label:<11} {name:<26} {r['p_now']:>6}  {r['q_now']:>9}  {r['g_now']:>4}   {r['p_if']:>9}  "
                  f"{r['q_if']:>12}  {r['g_if']:>7}  {np.median(r['amb']):>16.0f}")
    print(f"\nverifier calls: {len(asked)}")
    print(f"power at end: {power_state()}")
    print("\nCalibrated measurement finished.")
