"""Learned fusion, step 1: a labelled chunk dataset for one corpus.

For every target question and pool (clean, published, with question, and the adaptive
pools when built), the same pool of 30 as run C: per-chunk features (learned_fusion),
label (1 = poison), ids, sources and texts. Also stored, so the comparisons use the same
pools: the frozen Guard's fusion actions and final top 5 (real verifier, called only by
its own rule) and TrustRAG Stage 1's top 5. For the clean reference questions, the
features of each clean pool and the fixed fitting / held-out split.

Harness check: the frozen Guard's poison counts must equal run C's
(data/<corpus>/report_detection.json) for every pool both have.

Output: data/learned/chunks_<corpus>.json (git-ignored; holds attack text, evaluation only).
Usage: python tests/build_chunk_dataset.py nq|hotpotqa|msmarco
"""
import asyncio
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from baseline_trustrag import k_mean_filtering  # noqa: E402
from bench_real_pools import build_pool, retrieve  # noqa: E402
from report_common import load_questions  # noqa: E402
from trustrag_guard import learned_fusion as lf  # noqa: E402
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import OllamaVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, open_store  # noqa: E402

SIMCSE = "princeton-nlp/sup-simcse-bert-base-uncased"
FEATURES = lf.STAGE1_FEATURES + ("tightness",)


def as_lists(feats):
    return {k: [float(x) for x in feats[k]] for k in FEATURES + ("sims",)}


if __name__ == "__main__":
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    name = sys.argv[1]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", name)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "learned")
    os.makedirs(out_dir, exist_ok=True)
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
    data = load_questions(name)
    v = bt.load(os.path.join(data_dir, "thresholds.json"))
    adaptive_path = os.path.join(data_dir, "adaptive.json")
    adaptive = json.load(open(adaptive_path, encoding="utf-8")) if os.path.exists(adaptive_path) else None
    pools = ["clean", "published", "with question"] + (["diverse", "capped", "benign"] if adaptive else [])
    print(f"{name}: pools {pools}; values", {k: round(v[k], 3) for k in ("threshold", "clean_median", "tau", "low", "high")})
    warmup()
    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()

    targets = []
    for number, t in enumerate(data["targets"]):
        question = t["question"]
        q = enc([question])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        extra = adaptive[t["id"]] if adaptive else {}
        attacks = {"clean": [], "published": t["adv_texts"],
                   "with question": [f"{question}.{x}" for x in t["adv_texts"]], **extra}
        row = dict(id=t["id"], question=question, gold=list(t["gold"]), correct=t.get("correct"),
                   incorrect=t.get("incorrect"), pools={})
        for pool_name in pools:
            poison = attacks[pool_name]
            p_ids, p_texts, p_sources, p_embs = build_pool(ids, texts, sources, embs, q, poison, enc(poison) if poison else None)
            feats = lf.chunk_features(q, p_embs, p_texts, v["threshold"], v["clean_median"], tau=v["tau"], with_tightness=True)
            ours = asyncio.run(guard(question, q, p_embs, p_texts, p_sources, verifier, v["threshold"], v["clean_median"],
                                     tau=v["tau"], low=v["low"], high=v["high"]))
            top5 = p_texts[:DEFAULT_TARGET_K]
            by_text = dict(zip(p_texts, p_ids))
            row["pools"][pool_name] = dict(
                ids=p_ids, sources=p_sources, texts=p_texts, label=[int(i.startswith("poison")) for i in p_ids],
                features=as_lists(feats), frozen_actions=[str(a) for a in ours.fusion.actions],
                frozen_top5=[by_text[c] for c in ours.chunks[:DEFAULT_TARGET_K]],
                frozen_calls=int(ours.verified.n_checked),
                trustrag_top5=[by_text[c] for c in k_mean_filtering(simcse_embed(top5), top5)])
        targets.append(row)
        if (number + 1) % 10 == 0:
            print(f"  {number + 1} of {len(data['targets'])} questions", flush=True)

    fit, held = bt.split_questions(data["reference"])
    reference = []
    for part, rows in (("fit", fit), ("held", held)):
        for r in rows:
            q = enc([r["question"]])[0]
            _, texts, _, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
            feats = lf.chunk_features(q, embs, texts, v["threshold"], v["clean_median"], tau=v["tau"], with_tightness=True)
            reference.append(dict(part=part, texts=texts, features=as_lists(feats)))

    # Harness check against run C.
    rc_path = os.path.join(data_dir, "report_detection.json")
    if os.path.exists(rc_path):
        rc = json.load(open(rc_path, encoding="utf-8"))["summary"]["TrustRAG-Guard"]
        for pool_name in pools:
            if pool_name in rc:
                mine = sum(sum(i.startswith("poison") for i in t["pools"][pool_name]["frozen_top5"]) for t in targets)
                status = "match" if mine == rc[pool_name]["poison"] else "MISMATCH"
                print(f"harness check {pool_name}: frozen Guard poison {mine}, run C {rc[pool_name]['poison']} -> {status}")
    json.dump({"corpus": name, "values": v, "pools": pools, "targets": targets, "reference": reference},
              open(os.path.join(out_dir, f"chunks_{name}.json"), "w", encoding="utf-8"), default=float)
    print(f"\nwrote data/learned/chunks_{name}.json: {len(targets)} targets x {len(pools)} pools, {len(reference)} reference pools")
    print("\nChunk dataset finished.")
