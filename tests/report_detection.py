"""Evaluation report, run C: detection only, no generator.

For each dataset, every pool goes through the undefended top 5, TrustRAG Stage 1, the
Guard (real verifier, called only by its own rule), and three single-signal
ablations. Each ablation drops only what its signal flags on its own, then takes the
top 5 survivors in rank order with no floor:
    signal A only     sim >= fitted threshold
    clustering only   the suspect cluster, when silhouette >= fitted tau
    layer 2 only      coordination >= DEFAULT_COPY_CUTOFF (0.5, shipped, UNCALIBRATED)
Held-out false alarms use the held-out half of the clean reference questions.

Units: poison in chunks of 500 slots; questions with poison, gold, empty in questions
of 100; CCR in gold chunks retained, of those in the undefended top 5.
Nothing is tuned. Values: per-corpus thresholds.json, or the shipped reference for NQ.
Usage: python tests/report_detection.py nq|hotpotqa|msmarco
"""
import asyncio
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from baseline_trustrag import k_mean_filtering  # noqa: E402
from bench_adaptive import SIMCSE  # noqa: E402
from bench_real_pools import build_pool, retrieve  # noqa: E402
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fusion import DROP, KEEP  # noqa: E402
from trustrag_guard.layers.text_space import DEFAULT_COPY_CUTOFF, warmup  # noqa: E402
from trustrag_guard.layers.verifier import OllamaVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, open_store  # noqa: E402

SYSTEMS = ("undefended", "TrustRAG Stage 1", "TrustRAG-Guard", "signal A only", "clustering only", "layer 2 only (uncalibrated)")

if __name__ == "__main__":
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    name = sys.argv[1]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", name)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True).astype(np.float64)
    tokenizer = AutoTokenizer.from_pretrained(SIMCSE)
    simcse = AutoModel.from_pretrained(SIMCSE).eval()

    def simcse_embed(texts):
        batch = tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = simcse(**batch, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1][:, 0, :].numpy()

    store = open_store(os.path.join(data_dir, "chroma"), f"{name}_clean")
    data = json.load(open(os.path.join(data_dir, "questions.json"), encoding="utf-8"))
    if name == "nq":
        v, source = bt.load_reference("all-MiniLM-L6-v2"), "shipped reference_thresholds.json"
    else:
        v, source = bt.load(os.path.join(data_dir, "thresholds.json")), f"data/{name}/thresholds.json"
    print(f"{name}: values from {source}:", {k: round(v[k], 3) for k in ("threshold", "clean_median", "tau", "low", "high")})
    adaptive_path = os.path.join(data_dir, "adaptive.json")
    adaptive = json.load(open(adaptive_path, encoding="utf-8")) if os.path.exists(adaptive_path) else None
    pools = ["clean", "published", "with question"] + (["diverse", "capped", "benign"] if adaptive else [])
    warmup()
    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()

    def scan(q, embs, texts):
        return asyncio.run(scan_parallel(q, embs, texts, v["threshold"], v["clean_median"],
                                         tau=v["tau"], low=v["low"], high=v["high"]))

    per_q = {s: {p: [] for p in pools} for s in SYSTEMS}
    for number, t in enumerate(data["targets"]):
        question, gold = t["question"], set(t["gold"])
        q = enc([question])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        extra = adaptive[t["id"]] if adaptive else {}
        attacks = {"clean": [], "published": t["adv_texts"],
                   "with question": [f"{question}.{x}" for x in t["adv_texts"]], **extra}
        for pool_name in pools:
            poison = attacks[pool_name]
            p_ids, p_texts, p_sources, p_embs = build_pool(ids, texts, sources, embs, q, poison, enc(poison) if poison else None)
            by_text = dict(zip(p_texts, p_ids))
            base_gold = [i for i in p_ids[:DEFAULT_TARGET_K] if i in gold]
            f = scan(q, p_embs, p_texts)
            ours = asyncio.run(guard(question, q, p_embs, p_texts, p_sources, verifier, v["threshold"], v["clean_median"],
                                     tau=v["tau"], low=v["low"], high=v["high"]))
            calls = int(ours.verified.n_checked)
            top5 = p_texts[:DEFAULT_TARGET_K]
            alone = {"signal A only": f.sims >= v["threshold"],
                     "clustering only": np.asarray(f.cluster_suspect, dtype=bool),
                     "layer 2 only (uncalibrated)": f.coordination >= DEFAULT_COPY_CUTOFF}
            chosen_by = {"undefended": top5, "TrustRAG Stage 1": k_mean_filtering(simcse_embed(top5), top5),
                         "TrustRAG-Guard": ours.chunks[:DEFAULT_TARGET_K]}
            for s, mask in alone.items():
                chosen_by[s] = [p_texts[i] for i in range(len(p_texts)) if not mask[i]][:DEFAULT_TARGET_K]
            for s in SYSTEMS:
                chosen = [by_text[c] for c in chosen_by[s]]
                per_q[s][pool_name].append(dict(
                    poison=sum(i.startswith("poison") for i in chosen), top=len(base_gold),
                    kept=sum(i in chosen for i in base_gold), gold=any(i in gold for i in chosen),
                    empty=not chosen, calls=calls if s == "TrustRAG-Guard" else 0))
        if (number + 1) % 10 == 0:
            print(f"  {number + 1} of {len(data['targets'])} questions", flush=True)

    print("\nUnits: poison in chunks of 500; with poison, gold and empty in questions of 100; CCR in gold chunks.")
    print(f" {'pool':<14} {'system':<28} {'poison':>6} {'with poison':>11} {'gold':>5} {'CCR':>14} {'empty':>5} {'calls':>5}")
    summary = {}
    for p in pools:
        for s in SYSTEMS:
            rows = per_q[s][p]
            top, kept = sum(r["top"] for r in rows), sum(r["kept"] for r in rows)
            row = dict(poison=sum(r["poison"] for r in rows), with_poison=sum(r["poison"] > 0 for r in rows),
                       gold=sum(r["gold"] for r in rows), kept=kept, top=top,
                       empty=sum(r["empty"] for r in rows), calls=sum(r["calls"] for r in rows))
            summary.setdefault(s, {})[p] = row
            ccr = f"{kept / top:.1%} {kept}/{top}" if top else "n/a"
            print(f" {p:<14} {s:<28} {row['poison']:>6} {row['with_poison']:>11} {row['gold']:>5} {ccr:>14} {row['empty']:>5} {row['calls']:>5}")

    _, held = bt.split_questions(data["reference"])
    chunks = flagged = dropped = q_any = q_drop = 0
    for row in held:
        q = enc([row["question"]])[0]
        _, texts, _, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        acts = scan(q, embs, texts).actions[:DEFAULT_TARGET_K]
        not_keep = [a != KEEP for a in acts]
        drops = [a == DROP for a in acts]
        chunks += len(acts)
        flagged += sum(not_keep)
        dropped += sum(drops)
        q_any += any(not_keep)
        q_drop += any(drops)
    fa = dict(questions=len(held), chunks=chunks, not_keep=flagged, dropped=dropped, q_not_keep=q_any, q_dropped=q_drop)
    print(f"\nheld-out clean false alarms ({len(held)} questions, {chunks} top-5 chunks):")
    print(f" per chunk: not KEEP {flagged} ({flagged / chunks:.1%}), dropped {dropped} ({dropped / chunks:.1%})")
    print(f" per question: any not KEEP {q_any} ({q_any / len(held):.0%}), any dropped {q_drop} ({q_drop / len(held):.0%})")
    json.dump({"values_source": source, "values": v, "summary": summary, "per_question": per_q, "held_out": fa},
              open(os.path.join(data_dir, "report_detection.json"), "w", encoding="utf-8"), default=float)
    print("\nDetection checks finished.")
