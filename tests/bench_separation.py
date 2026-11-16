"""Separation test for the fitted detectors.

Can a fitted detector tell published poison from a correct passage at rank 1? It runs
on PoisonedRAG's HotpotQA questions, which nothing here has used, and everything is
fitted on clean HotpotQA.

Word count is a baseline: published poison is short, so a detector that matches it is
noticing length. Fused risk uses thresholds refitted on this corpus.

Run tests/build_nq_store.py hotpotqa first.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from bench_real_pools import build_pool, retrieve  # noqa: E402
from bench_verifier_length import power_state  # noqa: E402
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.fusion import fuse  # noqa: E402
from trustrag_guard.layers.fitted import fit_detectors, isolation_score, mahalanobis_score  # noqa: E402
from trustrag_guard.layers.fitted import save as save_models  # noqa: E402
from trustrag_guard.layers.layer1_scan import layer1_scan  # noqa: E402
from trustrag_guard.layers.text_space import coordination_test, warmup  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, open_store  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "hotpotqa")
MODELS = os.path.join(os.path.dirname(__file__), "..", "fitted_models")
N_FIT_PASSAGES = 20_000
ALPHA = 0.05
MIN_AUROC, MIN_CAUGHT, MAX_FLAGGED = 0.80, 0.50, 0.10
BOOTSTRAPS = 1000
SEED = 42


def auroc_interval(pos, neg, seed=SEED):
    """Bootstrap over questions. `pos` and `neg` map a question to its scores."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(pos) | set(neg))
    out = []
    for _ in range(BOOTSTRAPS):
        picked = rng.choice(len(keys), len(keys))
        a = [v for i in picked for v in pos.get(keys[i], [])]
        b = [v for i in picked for v in neg.get(keys[i], [])]
        if a and b:
            out.append(roc_auc_score([1] * len(a) + [0] * len(b), a + b))
    return np.percentile(out, [2.5, 97.5])


def flat(d):
    return [v for vals in d.values() for v in vals]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print(f"power at start: {power_state()}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True).astype(np.float64)
    store = open_store(os.path.join(DATA, "chroma"), "hotpotqa_clean")
    data = json.load(open(os.path.join(DATA, "questions.json"), encoding="utf-8"))
    print(f"store {store.count()}, test questions {len(data['targets'])}, reference {len(data['reference'])}")
    warmup()

    fitting, held = bt.split_questions(data["reference"])
    pools = []
    for row in fitting:
        q = enc([row["question"]])[0]
        ids, texts, _, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        pools.append((q, embs, texts))
    values = bt.fit_values(pools)
    bt.save(values, os.path.join(DATA, "thresholds.json"))
    print("refitted on clean HotpotQA:", {k: round(v, 3) for k, v in values.items()})

    test_gold = {g for t in data["targets"] for g in t["gold"]}
    all_ids = store.get(include=[])["ids"]
    pool_ids = [i for i in all_ids if i not in test_gold]
    picks = np.random.default_rng(SEED).permutation(len(pool_ids))[:N_FIT_PASSAGES]
    clean = []
    for s in range(0, len(picks), 4000):
        got = store.get(ids=[pool_ids[i] for i in picks[s:s + 4000]], include=["embeddings"])
        clean.append(np.asarray(got["embeddings"], dtype=np.float64))
    clean = np.vstack(clean)
    fitted = fit_detectors(clean)
    os.makedirs(MODELS, exist_ok=True)
    save_models(fitted, os.path.join(MODELS, "hotpotqa_minilm.joblib"))
    print(f"detectors fitted on {len(clean)} clean passages")

    def signals(q, embs, texts):
        fused = fuse(layer1_scan(q, embs, values["threshold"], tau=values["tau"]),
                     coordination_test(texts), values["clean_median"])
        return {"similarity": embs @ q, "fused risk": fused.risk,
                "word count": -np.array([len(t.split()) for t in texts], dtype=np.float64),
                "Isolation Forest": isolation_score(fitted, embs),
                "Mahalanobis": mahalanobis_score(fitted, embs)}

    names = ("similarity", "fused risk", "word count", "Isolation Forest", "Mahalanobis")
    cut_scores = {n: [] for n in names}
    for row, (q, embs, texts) in zip(fitting, pools):
        hits = retrieve(store, q, 1)[0]
        if hits[0] in set(row["gold"]):
            s = signals(q, embs, texts)
            for n in names:
                cut_scores[n].append(float(s[n][0]))
    cuts = {n: bt.conformal_quantile(cut_scores[n], ALPHA) for n in names}
    print(f"cutoffs from {len(cut_scores['similarity'])} rank-1 gold chunks of fitting questions")

    gold_scores = {n: {} for n in names}
    poison_scores = {f: {n: {} for n in names} for f in ("published", "with question")}
    in_pool = {"published": 0, "with question": 0}
    ccr_top, ccr_kept = 0, 0
    for t in data["targets"]:
        question, gold = t["question"], set(t["gold"])
        q = enc([question])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        s = signals(q, embs, texts)
        if ids[0] in gold:
            for n in names:
                gold_scores[n][t["id"]] = [float(s[n][0])]
        for form, poison in (("published", t["adv_texts"]),
                             ("with question", [f"{question} {x}" for x in t["adv_texts"]])):
            pool = build_pool(ids, texts, sources, embs, q, poison, enc(poison))
            p_ids, p_texts, _, p_embs = pool
            mask = np.array([i.startswith("poison") for i in p_ids])
            in_pool[form] += int(mask.sum())
            s2 = signals(q, p_embs, p_texts)
            for n in names:
                poison_scores[form][n][t["id"]] = s2[n][mask].tolist()

    print(f"\nrank-1 gold chunks on test questions: {len(flat(gold_scores['similarity']))}; "
          f"poison in pool: {in_pool}")
    print("\n signal             form            AUROC  95% interval   caught  gold flagged  pass")
    verdict = {}
    for form in ("published", "with question"):
        for n in names:
            a, b = poison_scores[form][n], gold_scores[n]
            fa, fb = flat(a), flat(b)
            auroc = roc_auc_score([1] * len(fa) + [0] * len(fb), fa + fb)
            lo, hi = auroc_interval(a, b)
            caught = float(np.mean(np.array(fa) > cuts[n]))
            flagged = float(np.mean(np.array(fb) > cuts[n]))
            ok = auroc >= MIN_AUROC and caught >= MIN_CAUGHT and flagged <= MAX_FLAGGED
            if form == "published":
                verdict[n] = ok
            print(f" {n:<18} {form:<14} {auroc:6.3f}  {lo:.3f}-{hi:.3f}   {caught:6.1%}  {flagged:12.1%}  {'yes' if ok else 'no'}")
    passed = [n for n in ("Isolation Forest", "Mahalanobis") if verdict[n]]
    print(f"\nverdict: {'PASS: ' + ', '.join(passed) if passed else 'FAIL: neither detector meets the rule'}")
    print(f"word count alone passes: {verdict['word count']}")
    print(f"power at end: {power_state()}")
    print("\nSeparation test finished.")
