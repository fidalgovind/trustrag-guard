"""Learned fusion, Stage 1: leave-one-corpus-out training and detection-only evaluation.
A development evaluation: all 300 target questions were used before.

For each held-out corpus T, with training corpora A and B:
  1. Nested selection, on A and B only: each candidate is trained on one, its bounds are
     fitted on the other's clean reference fitting half, and it is scored there (F1 with
     DROP as detection, pools clean + published + with question); the mean of the two
     directions decides, and a difference under 0.02 goes to the candidate with fewer
     features.
  2. Every candidate is then trained on A + B (same pools), its bounds are fitted on T's
     clean reference fitting half, and it is evaluated once on T. All candidates are
     reported (the rank ablation); the selected one is marked, chosen before T is seen.
Adaptive pools are never trained on and never used for selection.

Comparisons on the same pools, from data/learned/chunks_<corpus>.json: the frozen Guard
and TrustRAG Stage 1 (which sees only the top 5, so its precision and recall are over
the top 5).

Units: precision / recall / F1 over the chunks of the pool of 30 (TrustRAG: top 5);
poison in chunks of 500; with poison, gold and empty in questions of 100; CCR in gold
chunks kept, of those in the undefended top 5.

Outputs: data/learned/stage1_detection.json, data/learned/weights_stage1_<corpus>.json.
Needs Ollama only if a learned Guard falls below the floor (the verifier rule).
Usage: python tests/report_learned_detection.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard import learned_fusion as lf  # noqa: E402
from trustrag_guard.context_floor import context_floor  # noqa: E402
from trustrag_guard.fusion import DROP, KEEP  # noqa: E402
from trustrag_guard.layers.verifier import DEFAULT_MAX_VERIFIER_CALLS, OllamaVerifier, verify_ambiguous  # noqa: E402

CORPORA = ("nq", "hotpotqa", "msmarco")
TRAIN_POOLS = ("clean", "published", "with question")
CANDIDATES = {"with rank": lf.STAGE1_FEATURES,
              "without rank": tuple(f for f in lf.STAGE1_FEATURES if f != "rank")}
TIE = 0.02
TOP_K = 5
BOOTSTRAPS = 2000
SEED = 42
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "learned")


def rows(corpus, features, pools=TRAIN_POOLS):
    X, y = [], []
    for t in corpus["targets"]:
        for p in pools:
            if p in t["pools"]:
                f = t["pools"][p]["features"]
                X.append(np.column_stack([f[n] for n in features]))
                y.append(np.asarray(t["pools"][p]["label"]))
    return np.vstack(X), np.concatenate(y)


def bounded(model, corpus):
    clean = [r["features"] for r in corpus["reference"] if r["part"] == "fit"]
    return lf.fit_bounds(model, clean)


def prf(detected, label):
    detected, label = np.asarray(detected, bool), np.asarray(label, bool)
    tp, fp, fn = int((detected & label).sum()), int((detected & ~label).sum()), int((~detected & label).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if tp else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def drop_f1(model, corpus, pools=TRAIN_POOLS):
    det, lab = [], []
    for t in corpus["targets"]:
        for p in pools:
            if p in t["pools"]:
                r = lf.decide(model, t["pools"][p]["features"])
                det.append(r.actions == DROP)
                lab.append(np.asarray(t["pools"][p]["label"]))
    return prf(np.concatenate(det), np.concatenate(lab))["f1"]


def interval(values, rng):
    a = np.asarray(values, dtype=np.float64)
    picks = rng.integers(0, len(a), (BOOTSTRAPS, len(a)))
    return [float(x) for x in np.percentile(a[picks].sum(axis=1), [2.5, 97.5])]


if __name__ == "__main__":
    data = {c: json.load(open(os.path.join(OUT, f"chunks_{c}.json"), encoding="utf-8")) for c in CORPORA}
    verifier = None
    verdicts = {}

    def verifier_for():
        global verifier
        if verifier is None:
            verifier = OllamaVerifier(timeout=180)
            verifier.warmup()
        return verifier

    class CachedVerifier:
        def verify(self, query, chunk):
            key = (query, chunk)
            if key not in verdicts:
                verdicts[key] = verifier_for().verify(query, chunk)
            return verdicts[key]

    results = {"protocol": "development evaluation", "folds": {}}
    rng = np.random.default_rng(SEED)
    for test in CORPORA:
        a, b = [c for c in CORPORA if c != test]
        fold = results["folds"].setdefault(test, {"train": [a, b], "inner": {}, "candidates": {}})

        # 1. Nested selection on the training corpora only.
        for name, feats in CANDIDATES.items():
            scores = []
            for tr, va in ((a, b), (b, a)):
                m = bounded(lf.fit_model(*rows(data[tr], feats), feats), data[va])
                scores.append(drop_f1(m, data[va]))
            fold["inner"][name] = dict(f1_each=scores, mean=float(np.mean(scores)), n_features=len(feats))
        ranked = sorted(CANDIDATES, key=lambda n: -fold["inner"][n]["mean"])
        best = ranked[0]
        for other in ranked[1:]:
            if fold["inner"][best]["mean"] - fold["inner"][other]["mean"] < TIE and \
                    fold["inner"][other]["n_features"] < fold["inner"][best]["n_features"]:
                best = other
        fold["selected"] = best
        print(f"\n=== held-out {test}; trained on {a} + {b}; inner DROP-F1: "
              + ", ".join(f"{n} {fold['inner'][n]['mean']:.3f}" for n in CANDIDATES) + f" -> selected: {best}")

        # 2. Train on A + B, fit bounds on T's clean reference, evaluate once on T.
        for name, feats in CANDIDATES.items():
            Xa, ya = rows(data[a], feats)
            Xb, yb = rows(data[b], feats)
            model = bounded(lf.fit_model(np.vstack([Xa, Xb]), np.r_[ya, yb], feats), data[test])
            lf.save(model, os.path.join(OUT, f"weights_stage1_{test}_{name.replace(' ', '_')}.json"))
            train_f1 = [drop_f1(bounded(model, data[c]), data[c]) for c in (a, b)]
            res = dict(weights=dict(zip(feats, model.coef.tolist())), intercept=model.intercept,
                       low=model.low, high=model.high, train_f1=train_f1, held_f1=drop_f1(model, data[test]), pools={})
            for p in data[test]["pools"]:
                det_drop, det_any, lab = [], [], []
                per_q = []
                for t in data[test]["targets"]:
                    pool = t["pools"][p]
                    r = lf.decide(model, pool["features"])
                    det_drop.append(r.actions == DROP)
                    det_any.append(r.actions != KEEP)
                    lab.append(np.asarray(pool["label"]))
                    ver = verify_ambiguous(t["question"], pool["texts"], r, CachedVerifier(),
                                           k_min=2, max_calls=DEFAULT_MAX_VERIFIER_CALLS)
                    floor = context_floor(ver.actions, pool["sources"])
                    chosen = [pool["ids"][i] for i in floor.selected[:TOP_K]]
                    gold = set(t["gold"])
                    base_gold = [i for i in pool["ids"][:TOP_K] if i in gold]
                    per_q.append(dict(poison=sum(i.startswith("poison") for i in chosen), top=len(base_gold),
                                      kept=sum(i in chosen for i in base_gold), gold=any(i in gold for i in chosen),
                                      empty=not chosen, calls=int(ver.n_checked)))
                lab = np.concatenate(lab)
                res["pools"][p] = dict(drop=prf(np.concatenate(det_drop), lab), not_keep=prf(np.concatenate(det_any), lab),
                                       per_question=per_q)
            # Held-out clean false alarms on T's reference held-out half.
            acts = [lf.decide(model, r["features"]).actions[:TOP_K] for r in data[test]["reference"] if r["part"] == "held"]
            res["held_out"] = dict(questions=len(acts), chunks=sum(len(x) for x in acts),
                                   not_keep=int(sum((x != KEEP).sum() for x in acts)), dropped=int(sum((x == DROP).sum() for x in acts)),
                                   q_not_keep=int(sum((x != KEEP).any() for x in acts)), q_dropped=int(sum((x == DROP).any() for x in acts)))
            fold["candidates"][name] = res

        # Comparisons on the same pools: frozen Guard and TrustRAG Stage 1.
        for system in ("frozen Guard", "TrustRAG Stage 1"):
            res = {"pools": {}}
            for p in data[test]["pools"]:
                det_drop, det_any, lab, per_q = [], [], [], []
                for t in data[test]["targets"]:
                    pool = t["pools"][p]
                    label = np.asarray(pool["label"])
                    if system == "frozen Guard":
                        acts = np.asarray(pool["frozen_actions"], dtype=object)
                        det_drop.append(acts == DROP); det_any.append(acts != KEEP); lab.append(label)
                        chosen = pool["frozen_top5"]
                        calls = pool["frozen_calls"]
                    else:
                        top = pool["ids"][:TOP_K]
                        removed = np.array([i not in pool["trustrag_top5"] for i in top])
                        det_drop.append(removed); det_any.append(removed); lab.append(label[:TOP_K])
                        chosen = pool["trustrag_top5"]
                        calls = 0
                    gold = set(t["gold"])
                    base_gold = [i for i in pool["ids"][:TOP_K] if i in gold]
                    per_q.append(dict(poison=sum(i.startswith("poison") for i in chosen), top=len(base_gold),
                                      kept=sum(i in chosen for i in base_gold), gold=any(i in gold for i in chosen),
                                      empty=not chosen, calls=calls))
                lab = np.concatenate(lab)
                res["pools"][p] = dict(drop=prf(np.concatenate(det_drop), lab), not_keep=prf(np.concatenate(det_any), lab),
                                       per_question=per_q)
            fold["candidates"][system] = res

        # Paired differences in poison reaching the top 5, learned (each) minus frozen Guard.
        fold["paired_poison"] = {}
        for name in CANDIDATES:
            for p in data[test]["pools"]:
                d = [x["poison"] - y["poison"] for x, y in zip(fold["candidates"][name]["pools"][p]["per_question"],
                                                               fold["candidates"]["frozen Guard"]["pools"][p]["per_question"])]
                fold["paired_poison"][f"{name} | {p}"] = dict(diff=int(sum(d)), interval=interval(d, rng))

        # Rank shortcut diagnostics: pooled correlation of rank with alignment z.
        rk, az = [], []
        for t in data[test]["targets"]:
            for p in TRAIN_POOLS:
                rk += t["pools"][p]["features"]["rank"]; az += t["pools"][p]["features"]["alignment_z"]
        fold["rank_alignment_corr"] = float(np.corrcoef(rk, az)[0, 1])

    # Print the tables.
    def fmt(x):
        return "  n/a" if x != x else f"{x:5.2f}"

    print("\nUnits: P/R/F1 over the pool of 30 (TrustRAG: top 5); poison in chunks of 500; with poison, gold, empty in questions of 100; CCR gold chunks.")
    for test, fold in results["folds"].items():
        print(f"\n##### held-out corpus: {test} (trained on {' + '.join(fold['train'])}); selected before testing: {fold['selected']}")
        print(f" {'pool':<14} {'system':<18} {'DROP P':>6} {'R':>5} {'F1':>5} | {'notKEEP P':>9} {'R':>5} {'F1':>5} | {'poison':>6} {'w/poison':>8} {'gold':>4} {'CCR':>14} {'empty':>5} {'calls':>5}")
        for p in results["folds"][test]["candidates"]["frozen Guard"]["pools"]:
            for system in list(CANDIDATES) + ["frozen Guard", "TrustRAG Stage 1"]:
                r = fold["candidates"][system]["pools"][p]
                q = r["per_question"]
                top, kept = sum(x["top"] for x in q), sum(x["kept"] for x in q)
                ccr = f"{kept / top:.1%} {kept}/{top}" if top else "n/a"
                d, n = r["drop"], r["not_keep"]
                if p == "clean":
                    det = f" {'FP ' + str(d['fp']):>18} | {'FP ' + str(n['fp']):>21} |"
                else:
                    det = f" {fmt(d['precision']):>6} {fmt(d['recall']):>5} {fmt(d['f1']):>5} | {fmt(n['precision']):>9} {fmt(n['recall']):>5} {fmt(n['f1']):>5} |"
                print(f" {p:<14} {system:<18}{det} {sum(x['poison'] for x in q):>6} {sum(x['poison'] > 0 for x in q):>8} "
                      f"{sum(x['gold'] for x in q):>4} {ccr:>14} {sum(x['empty'] for x in q):>5} {sum(x['calls'] for x in q):>5}")
        for name in CANDIDATES:
            c = fold["candidates"][name]
            h = c["held_out"]
            print(f" {name}: weights " + ", ".join(f"{k} {w:+.2f}" for k, w in c["weights"].items())
                  + f"; low {c['low']:.3f}, high {c['high']:.3f}")
            print(f"   held-out clean false alarms: not KEEP {h['not_keep']}/{h['chunks']} ({h['not_keep'] / h['chunks']:.1%}), "
                  f"dropped {h['dropped']} ({h['dropped'] / h['chunks']:.1%}); per question {h['q_not_keep']} and {h['q_dropped']} of {h['questions']}")
            print(f"   DROP-F1 (clean + published + with question): training corpora {', '.join(f'{x:.3f}' for x in c['train_f1'])}; "
                  f"held-out {c['held_f1']:.3f}; gap {np.mean(c['train_f1']) - c['held_f1']:+.3f}")
        print(f" rank vs alignment z correlation, pooled over pools: {fold['rank_alignment_corr']:+.2f} (within a pool rank is exactly the similarity order)")
        print(" paired poison in top 5, learned minus frozen Guard, chunks (95% interval):")
        for k, x in fold["paired_poison"].items():
            print(f"   {k:<34} {x['diff']:>+5} ({x['interval'][0]:+.0f} to {x['interval'][1]:+.0f})")

    print(f"\nverifier calls made by learned Guards: {len(verdicts)}")
    json.dump(results, open(os.path.join(OUT, "stage1_detection.json"), "w", encoding="utf-8"), default=float)
    print("\nStage 1 detection finished.")
