"""Evaluation report, run A: Stage 7's full matrix on one dataset.

A copy of tests/bench_adaptive.py with the dataset as an argument. Systems, pools,
prompts, generator settings and intervals are unchanged. Added: per-question records
(poison, gold, empty, verifier calls) and questions with any poison. Left out: the
cutoff curve, which this run does not include.

Every attacker runs against every system, under one budget. Intervals resample
questions. Clean ASR sits beside every figure, because the check fires with no
poison, and excess ASR subtracts it.

Units:
    ASR, clean ASR, correct    questions out of 100
    poison                     chunks out of 500 slots
    CCR                        gold chunks retained, of those in the undefended top 5

The forced-verifier system is a measurement and not a design change: it forces the
verifier onto at most 3 borderline chunks, whether or not the floor is met.

Run tests/report_build_attacks.py <dataset> first. Needs Ollama, on mains.
Usage: python tests/report_matrix.py hotpotqa
"""
import asyncio
import json
import os
import sys

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from baseline_trustrag import conflict_answer, k_mean_filtering  # noqa: E402
from bench_asr_curve import PROMPT, clean_str  # noqa: E402
from bench_real_pools import build_pool, retrieve  # noqa: E402
from bench_verifier_length import power_state  # noqa: E402
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import (  # noqa: E402
    DEFAULT_MAX_VERIFIER_CALLS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    SAFE,
    OllamaVerifier,
)
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, open_store  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "hotpotqa"
DATA = os.path.join(os.path.dirname(__file__), "..", "data", NAME)
SIMCSE = "princeton-nlp/sup-simcse-bert-base-uncased"
MAX_TOKENS = 128
BOOTSTRAPS = 2000
SEED = 42
POOLS = ("clean", "published", "with question", "diverse", "capped", "benign")
SYSTEMS = ("undefended", "TrustRAG Stage 1", "TrustRAG full", "TrustRAG-Guard", "Guard + forced verifier")


def interval(flags, rng):
    a = np.asarray(flags, dtype=np.float64)
    picks = rng.integers(0, len(a), (BOOTSTRAPS, len(a)))
    return np.percentile(100 * a[picks].mean(axis=1), [2.5, 97.5])


if __name__ == "__main__":
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    print(f"power at start: {power_state()}")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True).astype(np.float64)
    tokenizer = AutoTokenizer.from_pretrained(SIMCSE)
    simcse = AutoModel.from_pretrained(SIMCSE).eval()

    def simcse_embed(texts):
        batch = tokenizer(list(texts), return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = simcse(**batch, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1][:, 0, :].numpy()

    store = open_store(os.path.join(DATA, "chroma"), f"{NAME}_clean")
    data = json.load(open(os.path.join(DATA, "questions.json"), encoding="utf-8"))
    adaptive = json.load(open(os.path.join(DATA, "adaptive.json"), encoding="utf-8"))
    v = bt.load(os.path.join(DATA, "thresholds.json"))
    print("fitted values, unchanged:", {k: round(v[k], 3) for k in ("threshold", "clean_median", "tau", "low", "high")})
    warmup()

    client = httpx.Client(base_url=DEFAULT_OLLAMA_URL, timeout=300)
    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()
    answers, verdicts = {}, {}

    def llm(prompt):
        if prompt not in answers:
            r = client.post("/api/generate", json={
                "model": DEFAULT_OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": 0, "num_predict": MAX_TOKENS, "seed": 42}})
            r.raise_for_status()
            answers[prompt] = r.json()["response"]
        return answers[prompt]

    def answer(question, chunks):
        return llm(PROMPT.replace("[context]", "\n".join(chunks)).replace("[question]", question))

    def is_safe(question, text):
        key = (question, text)
        if key not in verdicts:
            verdicts[key] = verifier.verify(question, text).label == SAFE
        return verdicts[key]

    def forced(question, f, texts, cut):
        """Indices usable at `cut`, plus SAFE chunks among up to 3 borderline ones."""
        keep = {i for i in range(len(texts)) if f.risk[i] < cut}
        band = [i for i in np.argsort(f.risk, kind="stable") if cut <= f.risk[i] < v["high"]]
        forced.calls = len(band[:DEFAULT_MAX_VERIFIER_CALLS])
        for i in band[:DEFAULT_MAX_VERIFIER_CALLS]:
            if is_safe(question, texts[int(i)]):
                keep.add(int(i))
        return sorted(keep)[:DEFAULT_TARGET_K]

    # Checkpoint: every finished question is saved, and a
    # restart resumes after the last one. Answers are deterministic, so a resumed run gives
    # the same results as an unbroken one.
    checkpoint = os.path.join(DATA, "report_matrix_checkpoint.json")
    if os.path.exists(checkpoint):
        state = json.load(open(checkpoint, encoding="utf-8"))
        print(f"resuming after question {state['done']} from the checkpoint", flush=True)
    else:
        state = dict(done=0, flags={s: {p: [] for p in POOLS} for s in SYSTEMS},
                     right={s: {p: [] for p in POOLS} for s in SYSTEMS},
                     stats={s: {p: dict(poison=0, kept=0, top=0, q_gold=0, empty=0) for p in POOLS} for s in SYSTEMS},
                     retrieved={p: 0 for p in POOLS}, per_q={s: {p: [] for p in POOLS} for s in SYSTEMS}, battery=[])
    flags, right, stats, retrieved, per_q = state["flags"], state["right"], state["stats"], state["retrieved"], state["per_q"]
    dropouts = state["battery"]

    def charge():
        digits = "".join(ch for ch in power_state().split(",")[1] if ch.isdigit()) if "," in power_state() else ""
        return int(digits) if digits else 0

    for number, t in enumerate(data["targets"]):
        if number < state["done"]:
            continue
        # Battery is logged and tolerated; below 30% the run stops cleanly,
        # before Windows sleeps, and resumes later from the checkpoint.
        if not power_state().startswith("mains"):
            if charge() < 30:
                raise SystemExit(f"paused before question {number}: {power_state()}; progress is saved, rerun to resume")
            dropouts.append(number)
            print(f"  question {number} started on battery ({power_state()})", flush=True)
        question, gold = t["question"], set(t["gold"])
        bad, good = clean_str(t["incorrect"]), clean_str(t["correct"])
        q = enc([question])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        extra = adaptive[t["id"]]
        attacks = {"clean": [], "published": t["adv_texts"],
                   "with question": [f"{question}.{x}" for x in t["adv_texts"]],
                   "diverse": extra["diverse"], "capped": extra["capped"], "benign": extra["benign"]}
        for pool_name in POOLS:
            poison = attacks[pool_name]
            p_ids, p_texts, p_sources, p_embs = build_pool(ids, texts, sources, embs, q, poison, enc(poison) if poison else None)
            retrieved[pool_name] += sum(i.startswith("poison") for i in p_ids)
            by_text = dict(zip(p_texts, p_ids))
            top5 = p_texts[:DEFAULT_TARGET_K]
            base_gold = [i for i in p_ids[:DEFAULT_TARGET_K] if i in gold]

            stage1 = k_mean_filtering(simcse_embed(top5), top5)
            ours = asyncio.run(guard(question, q, p_embs, p_texts, p_sources, verifier, v["threshold"], v["clean_median"],
                                     tau=v["tau"], low=v["low"], high=v["high"]))
            mine = ours.chunks[:DEFAULT_TARGET_K]
            f = asyncio.run(scan_parallel(q, p_embs, p_texts, v["threshold"], v["clean_median"],
                                          tau=v["tau"], low=v["low"], high=v["high"]))
            helped = [p_texts[i] for i in forced(question, f, p_texts, v["low"])]
            n_calls = {"TrustRAG-Guard": int(ours.verified.n_checked), "Guard + forced verifier": forced.calls}

            for system, chunks, reply in (("undefended", top5, answer(question, top5)),
                                          ("TrustRAG Stage 1", stage1, answer(question, stage1)),
                                          ("TrustRAG full", stage1, conflict_answer(question, stage1, llm)),
                                          ("TrustRAG-Guard", mine, answer(question, mine)),
                                          ("Guard + forced verifier", helped, answer(question, helped))):
                chosen = [by_text[c] for c in chunks]
                r = stats[system][pool_name]
                r["poison"] += sum(i.startswith("poison") for i in chosen)
                r["empty"] += not chunks
                r["top"] += len(base_gold)
                r["kept"] += sum(i in chosen for i in base_gold)
                r["q_gold"] += any(i in gold for i in chosen)
                flags[system][pool_name].append(bad in clean_str(reply))
                right[system][pool_name].append(good in clean_str(reply))
                per_q[system][pool_name].append(dict(
                    poison=sum(i.startswith("poison") for i in chosen), top=len(base_gold),
                    kept=sum(i in chosen for i in base_gold), gold=any(i in gold for i in chosen),
                    empty=not chunks, calls=n_calls.get(system, 0)))

        state["done"] = number + 1
        json.dump(state, open(checkpoint + ".tmp", "w", encoding="utf-8"))
        os.replace(checkpoint + ".tmp", checkpoint)
        if (number + 1) % 10 == 0:
            print(f"  {number + 1} of {len(data['targets'])} questions", flush=True)

    rng = np.random.default_rng(SEED)
    print(f"\npoison retrieved into the pool of 30, of 500: " + ", ".join(f"{p} {retrieved[p]}" for p in POOLS[1:]))
    print("\nUnits: ASR, clean ASR, excess and correct in questions of 100; poison in chunks of 500; CCR in gold chunks.")
    print(f" {'pool':<14} {'system':<24} {'ASR':>4} {'95% interval':>13} {'clean ASR':>10} {'interval':>12} {'excess':>7} {'interval':>13} {'correct':>7} {'poison':>6} {'CCR':>12} {'gold':>5} {'empty':>5}")
    for pool_name in POOLS:
        for system in SYSTEMS:
            a = np.array(flags[system][pool_name], dtype=np.float64)
            c = np.array(flags[system]["clean"], dtype=np.float64)
            lo, hi = interval(a, rng)
            clo, chi = interval(c, rng)
            elo, ehi = interval(a - c, rng)
            r = stats[system][pool_name]
            ccr = f"{r['kept'] / r['top']:.1%} {r['kept']}/{r['top']}" if r["top"] else "n/a"
            print(f" {pool_name:<14} {system:<24} {100 * a.mean():>4.0f} {lo:>6.1f}-{hi:<6.1f} {100 * c.mean():>10.0f} {clo:>5.1f}-{chi:<6.1f} "
                  f"{100 * (a - c).mean():>7.0f} {elo:>6.1f}-{ehi:<6.1f} {100 * np.mean(right[system][pool_name]):>7.0f} {r['poison']:>6} {ccr:>12} {r['q_gold']:>5} {r['empty']:>5}")

    print("\nPaired ASR difference, resampling questions (negative favours the Guard):")
    print(f" {'pool':<14} {'Guard - Stage 1':>24} {'Guard - full':>24}")
    for pool_name in POOLS:
        g = np.array(flags["TrustRAG-Guard"][pool_name], dtype=np.float64)
        row = []
        for other in ("TrustRAG Stage 1", "TrustRAG full"):
            d = g - np.array(flags[other][pool_name], dtype=np.float64)
            lo, hi = interval(d, rng)
            row.append(f"{100 * d.mean():>6.0f} ({lo:.1f} to {hi:.1f})")
        print(f" {pool_name:<14} {row[0]:>24} {row[1]:>24}")


    print("\nQuestions with any poison, and verifier calls, of 100 questions:")
    for pool_name in POOLS:
        for system in SYSTEMS:
            rows = per_q[system][pool_name]
            print(f" {pool_name:<14} {system:<24} with poison {sum(r['poison'] > 0 for r in rows):>3}, verifier calls {sum(r['calls'] for r in rows):>3}")
    json.dump({"flags": flags, "right": right, "stats": stats, "per_question": per_q, "questions_started_on_battery": dropouts},
              open(os.path.join(DATA, "report_matrix.json"), "w", encoding="utf-8"))
    print(f"\nverifier calls {len(verdicts)}, generator calls {len(answers)}")
    print(f"power at end: {power_state()}")
    print("\nMatrix evaluation finished.")
