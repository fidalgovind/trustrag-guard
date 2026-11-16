"""The real verifier on real ambiguous chunks.

Every ambiguous chunk from the real pools goes to the verifier, ignoring the cap and
the floor. It then compares today's behaviour with a what-if, in which SAFE chunks
rank alongside kept chunks in retrieval order. The what-if changes no code path.

The 100 questions are the test set, so nothing may be tuned on this output. No
generator runs, so this is not an attack success rate.

Needs tests/build_nq_store.py and Ollama.
"""
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_real_pools import DATA, REFERENCE_TOP_K, build_pool, retrieve  # noqa: E402
from bench_verifier_length import power_state  # noqa: E402
from trustrag_guard.context_floor import context_floor  # noqa: E402
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, KEEP  # noqa: E402
from trustrag_guard.layers.over_alignment import fit_threshold  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import SAFE, OllamaVerifier  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, open_store  # noqa: E402

GROUPS = ("gold", "other clean", "poison")


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
    threshold, median = fit_threshold(sims), float(np.median(sims))
    print(f"threshold {threshold:.3f}, clean median {median:.3f}")

    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()
    asked, times = {}, []

    def verdict(question, text):
        key = (question, text)
        if key not in asked:
            start = time.perf_counter()
            asked[key] = verifier.verify(question, text).label
            times.append((time.perf_counter() - start) * 1000)
        return asked[key]

    names = ("clean", "poison as published", "poison with the question")
    counts = {n: {g: {"SAFE": 0, "other": 0} for g in GROUPS} for n in names}
    outcome = {n: {"now_poison": 0, "now_gold": 0, "if_poison": 0, "if_gold": 0, "if_any_poison": 0, "calls": []}
               for n in names}

    for t in data["targets"]:
        question, gold = t["question"], set(t["gold"])
        q = enc([question])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        plain = t["adv_texts"]
        for name, poison in (("clean", []), ("poison as published", plain),
                             ("poison with the question", [f"{question} {x}" for x in plain])):
            pool = build_pool(ids, texts, sources, embs, q, poison, enc(poison) if poison else None)
            p_ids, p_texts, p_sources, p_embs = pool
            f = asyncio.run(scan_parallel(q, p_embs, p_texts, threshold, median))
            amb = np.flatnonzero(f.actions == AMBIGUOUS)
            safe = set()
            for i in amb:
                label = verdict(question, p_texts[i])
                group = ("poison" if p_ids[i].startswith("poison") else
                         "gold" if p_ids[i] in gold else "other clean")
                counts[name][group]["SAFE" if label == SAFE else "other"] += 1
                if label == SAFE:
                    safe.add(int(i))

            now = context_floor(f.actions, p_sources).selected[:DEFAULT_TARGET_K]
            usable = [i for i in range(len(p_ids)) if f.actions[i] == KEEP or i in safe][:DEFAULT_TARGET_K]
            o = outcome[name]
            o["calls"].append(len(amb))
            for key, chosen in (("now", now), ("if", usable)):
                chosen_ids = [p_ids[int(i)] for i in chosen]
                o[f"{key}_poison"] += sum(i.startswith("poison") for i in chosen_ids)
                o[f"{key}_gold"] += any(i in gold for i in chosen_ids)
            o["if_any_poison"] += any(p_ids[i].startswith("poison") for i in usable)

    print(f"\n{len(times)} distinct verifier calls, p50 {np.median(times):.0f} ms, slowest {max(times):.0f} ms")
    print("\n pool type                  chunk         SAFE  UNSAFE or other")
    for name in names:
        for g in GROUPS:
            c = counts[name][g]
            if c["SAFE"] + c["other"]:
                print(f" {name:<26} {g:<12} {c['SAFE']:>5}  {c['other']:>5}")

    print(f"\n top {DEFAULT_TARGET_K}, 100 questions            poison now  poison if  gold now  gold if  median calls  most calls")
    for name in names:
        o = outcome[name]
        print(f" {name:<30} {o['now_poison']:>10}  {o['if_poison']:>9}  {o['now_gold']:>8}  {o['if_gold']:>7}  "
              f"{np.median(o['calls']):>12.0f}  {max(o['calls']):>10}")
        o["calls"] = [int(c) for c in o["calls"]]
    json.dump({"counts": counts, "outcome": outcome}, open(os.path.join(DATA, "verifier_real.json"), "w", encoding="utf-8"))
    print(f"\npower at end: {power_state()}")
    print("\nReal verifier measurement finished.")
