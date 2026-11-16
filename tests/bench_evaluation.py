"""Stage 6 evaluation. Run once.

Four systems on one shared setup: undefended, TrustRAG Stage 1, TrustRAG full, and
TrustRAG-Guard at the calibrated point. Results are split by attack form.

Units:
    poison      chunks in the context, out of 500 slots
    with poison questions with any poison in the context, out of 100
    ASR         questions whose answer contains the incorrect answer, out of 100
    correct     questions whose answer contains the correct answer, out of 100
    CCR         gold chunks retained, of gold chunks in the undefended top 5

ASR uses PoisonedRAG's prompt and substring check. The generator is local llama3.2:3b,
for every system. Clean ASR shows how often the check fires with no poison.

Run tests/build_nq_store.py msmarco first. Needs Ollama, on mains.
"""
import asyncio
import json
import math
import os
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from baseline_trustrag import conflict_answer, k_mean_filtering  # noqa: E402
from bench_asr_curve import PROMPT, clean_str  # noqa: E402
from bench_real_pools import build_pool, retrieve  # noqa: E402
from bench_verifier_length import power_state  # noqa: E402
from trustrag_guard.calibration import build_thresholds as bt  # noqa: E402
from trustrag_guard.context_floor import context_floor  # noqa: E402
from trustrag_guard.fast_path import scan_parallel  # noqa: E402
from trustrag_guard.fusion import AMBIGUOUS, DROP, KEEP, fuse  # noqa: E402
from trustrag_guard.layers.layer1_scan import layer1_scan  # noqa: E402
from trustrag_guard.layers.text_space import coordination_test, warmup  # noqa: E402
from trustrag_guard.layers.verifier import (  # noqa: E402
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    MockVerifier,
    OllamaVerifier,
)
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.retriever import DEFAULT_OVER_RETRIEVE, DEFAULT_TARGET_K, open_store  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "msmarco"
DATA = os.path.join(os.path.dirname(__file__), "..", "data", DATASET)
SIMCSE = "princeton-nlp/sup-simcse-bert-base-uncased"
MAX_TOKENS = 128
ROUNDS = 500
DISCARD = 20
FORMS = ("clean", "published", "with question")
SYSTEMS = ("undefended", "TrustRAG Stage 1", "TrustRAG full", "TrustRAG-Guard")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


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

    store = open_store(os.path.join(DATA, "chroma"), f"{DATASET}_clean")
    data = json.load(open(os.path.join(DATA, "questions.json"), encoding="utf-8"))
    print(f"store {store.count()}, test questions {len(data['targets'])}, reference {len(data['reference'])}")
    warmup()

    fitting, held = bt.split_questions(data["reference"])
    pools = []
    for row in fitting:
        q = enc([row["question"]])[0]
        _, texts, _, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        pools.append((q, embs, texts))
    v = bt.fit_values(pools)
    bt.save(v, os.path.join(DATA, "thresholds.json"))
    print("fitted on clean data:", {k: round(x, 3) for k, x in v.items()})

    n = low = high = split = 0
    for row in held:
        q = enc([row["question"]])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        f = asyncio.run(scan_parallel(q, embs, texts, v["threshold"], v["clean_median"],
                                      tau=v["tau"], low=v["low"], high=v["high"]))
        top = f.actions[:DEFAULT_TARGET_K]
        n += len(top)
        low += int((top != KEEP).sum())
        high += int((top == DROP).sum())
        split += bool(f.cluster_suspect.any())
    print(f"\nheld-out clean questions: {low} of {n} top chunks reached the band ({low / n:.1%}, promised about 5%); "
          f"{high} dropped ({high / n:.1%}, about 1%); {split} of {len(held)} pools split (about 1%)", flush=True)

    client = httpx.Client(base_url=DEFAULT_OLLAMA_URL, timeout=300)
    verifier = OllamaVerifier(timeout=180)
    verifier.warmup()
    cache = {}

    def llm(prompt):
        if prompt not in cache:
            r = client.post("/api/generate", json={
                "model": DEFAULT_OLLAMA_MODEL, "prompt": prompt, "stream": False,
                "options": {"temperature": 0, "num_predict": MAX_TOKENS, "seed": 42}})
            r.raise_for_status()
            cache[prompt] = r.json()["response"]
        return cache[prompt]

    def answer(question, chunks):
        return llm(PROMPT.replace("[context]", "\n".join(chunks)).replace("[question]", question))

    tally = {s: {f: dict(poison=0, with_poison=0, asr=0, correct=0, kept=0, top=0, q_gold=0, empty=0, calls=0, low_conf=0)
                 for f in FORMS} for s in SYSTEMS}

    for number, t in enumerate(data["targets"]):
        question, gold = t["question"], set(t["gold"])
        bad, good = clean_str(t["incorrect"]), clean_str(t["correct"])
        q = enc([question])[0]
        ids, texts, sources, embs = retrieve(store, q, DEFAULT_OVER_RETRIEVE)
        for form, poison in zip(FORMS, ([], t["adv_texts"], [f"{question}.{x}" for x in t["adv_texts"]])):
            p_ids, p_texts, p_sources, p_embs = build_pool(ids, texts, sources, embs, q, poison, enc(poison) if poison else None)
            by_text = dict(zip(p_texts, p_ids))
            top5 = p_texts[:DEFAULT_TARGET_K]
            base_gold = [i for i in p_ids[:DEFAULT_TARGET_K] if i in gold]

            stage1 = k_mean_filtering(simcse_embed(top5), top5)
            ours = asyncio.run(guard(question, q, p_embs, p_texts, p_sources, verifier, v["threshold"], v["clean_median"],
                                     tau=v["tau"], low=v["low"], high=v["high"]))
            mine = ours.chunks[:DEFAULT_TARGET_K]

            for system, chunks, reply in (("undefended", top5, answer(question, top5)),
                                          ("TrustRAG Stage 1", stage1, answer(question, stage1)),
                                          ("TrustRAG full", stage1, conflict_answer(question, stage1, llm)),
                                          ("TrustRAG-Guard", mine, answer(question, mine))):
                r = tally[system][form]
                chosen = [by_text[c] for c in chunks]
                count = sum(i.startswith("poison") for i in chosen)
                r["poison"] += count
                r["with_poison"] += count > 0
                r["asr"] += bad in clean_str(reply)
                r["correct"] += good in clean_str(reply)
                r["empty"] += not chunks
                r["top"] += len(base_gold)
                r["kept"] += sum(i in chosen for i in base_gold)
                r["q_gold"] += any(i in gold for i in chosen)
            tally["TrustRAG-Guard"][form]["calls"] += ours.verified.n_checked
            tally["TrustRAG-Guard"][form]["low_conf"] += ours.low_confidence
        if (number + 1) % 10 == 0:
            print(f"  {number + 1} of {len(data['targets'])} questions", flush=True)

    print("\nUnits: poison in chunks of 500; with poison, ASR, correct and gold in questions of 100; CCR in gold chunks.")
    print(f" {'system':<18} {'form':<14} {'poison':>6} {'with':>5} {'ASR':>4} {'95% interval':>13} {'correct':>7} {'CCR':>16} {'gold':>5} {'empty':>5}")
    for form in FORMS:
        for system in SYSTEMS:
            r = tally[system][form]
            lo, hi = wilson(r["asr"], 100)
            ccr = r["kept"] / r["top"] if r["top"] else float("nan")
            print(f" {system:<18} {form:<14} {r['poison']:>6} {r['with_poison']:>5} {r['asr']:>4} {lo:>6.1f}-{hi:<6.1f} "
                  f"{r['correct']:>7} {ccr:>7.1%} {r['kept']:>3}/{r['top']:<3} {r['q_gold']:>5} {r['empty']:>5}")
    g = tally["TrustRAG-Guard"]
    print(f"\nverifier calls: clean {g['clean']['calls']}, published {g['published']['calls']}, with question {g['with question']['calls']}")
    print(f"low confidence: clean {g['clean']['low_conf']}, published {g['published']['low_conf']}, with question {g['with question']['low_conf']}")
    json.dump(tally, open(os.path.join(DATA, "evaluation.json"), "w", encoding="utf-8"))

    # Latency per layer, on clean pools, with no verifier.
    loop = asyncio.new_event_loop()
    questions = [t["question"] for t in data["targets"]]
    names = ("query embedding", "retrieval", "Layer 1", "Layer 2", "fusion + floor", "whole fast path", "TrustRAG Stage 1")

    def clock(fn, *a):
        s = time.perf_counter()
        out = fn(*a)
        return out, (time.perf_counter() - s) * 1000

    def one(question):
        q, a = clock(lambda x: enc([x])[0], question)
        (ids, texts, sources, embs), b = clock(retrieve, store, q, DEFAULT_OVER_RETRIEVE)
        l1, c = clock(lambda: layer1_scan(q, embs, v["threshold"], tau=v["tau"]))
        l2, d = clock(coordination_test, texts)
        f, e = clock(lambda: fuse(l1, l2, v["clean_median"], low=v["low"], high=v["high"]))
        _, e2 = clock(context_floor, f.actions, sources)
        s = time.perf_counter()
        loop.run_until_complete(guard(question, q, embs, texts, sources, MockVerifier(), v["threshold"], v["clean_median"],
                                      tau=v["tau"], low=v["low"], high=v["high"], max_calls=0))
        whole = a + b + (time.perf_counter() - s) * 1000
        _, base = clock(lambda: k_mean_filtering(simcse_embed(texts[:DEFAULT_TARGET_K]), texts[:DEFAULT_TARGET_K]))
        return a, b, c, d, e + e2, whole, base

    for i in range(DISCARD):
        one(questions[i % len(questions)])
    timed = np.array([one(questions[i % len(questions)]) for i in range(ROUNDS)])
    print(f"\n--- latency, {ROUNDS} rounds, no verifier or generator ---")
    print(" part                  p50      p95      p99")
    for j, name in enumerate(names):
        p50, p95, p99 = np.percentile(timed[:, j], [50, 95, 99])
        print(f" {name:<19} {p50:6.1f}   {p95:6.1f}   {p99:6.1f} ms")
    print(f"\ngenerator calls: {len(cache)}")
    print(f"power at end: {power_state()}")
    print("\nEvaluation finished.")
