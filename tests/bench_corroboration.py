"""Corroboration test, on development data.

Question: is a true answer backed by more documents outside the retrieved pool
than an injected answer? For each PoisonedRAG question, this counts passages in
the full BEIR corpus that contain the answer and at least half of the question's
content words. Labelled gold passages are excluded, and injected poison is not in
the corpus files, so the counts are support from outside the pool.

This is an oracle test: it uses the known correct and incorrect answer strings.
A real system would extract candidate answers with a reader.

Usage: python tests/bench_corroboration.py [nq hotpotqa msmarco]
"""
import io
import json
import os
import re
import sys
import time
import zipfile

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MIN_SHARED = 2
STOP = set("""what which who whom whose when where why how many much the and for are was were did does
do is of in on at to a an by with from that this these those has have had been being his her its
their they you your can could would should will about into than then there here also not but all
any each other some such only own same both more most very name named called first last one two
year years many long take takes""".split())
WORD = re.compile(r"[a-z0-9]+")


def content_words(question):
    return {w for w in WORD.findall(question.lower()) if len(w) >= 3 and w not in STOP}


def pattern(answer):
    return r"(?<![a-z0-9])" + re.escape(answer) + r"(?![a-z0-9])"


def run(name):
    attacks = json.load(open(os.path.join(DATA, name, f"poisonedrag_{name}.json"), encoding="utf-8"))
    targets = json.load(open(os.path.join(DATA, name, "questions.json"), encoding="utf-8"))["targets"]
    gold = {t["id"]: set(t["gold"]) for t in targets}
    rows = []
    for key, v in attacks.items():
        good, bad = str(v["correct answer"]).strip().lower(), str(v["incorrect answer"]).strip().lower()
        words = content_words(v["question"])
        need = max(MIN_SHARED, -(-len(words) // 2))
        applicable = good not in ("yes", "no") and bad not in ("yes", "no") and len(good) > 1 and len(bad) > 1
        rows.append(dict(id=key, good=good, bad=bad, words=words, need=need, applicable=applicable, n_good=0, n_bad=0))
    # Each answer is searched on its own, so an answer inside a longer one ("york" in
    # "new york") is still found. An answer can only match if all its words are in the
    # line, so the exact pattern runs only on those lines (a combined regex took ~9 h).
    answers = sorted({r[k] for r in rows if r["applicable"] for k in ("good", "bad")})
    checks = {a: (set(WORD.findall(a)), re.compile(pattern(a))) for a in answers}
    by_first = {}
    always = []
    for a, (toks, _) in checks.items():
        if toks:
            by_first.setdefault(sorted(toks)[0], []).append(a)
        else:
            always.append(a)
    first_keys = set(by_first)
    by_answer = {}
    for r in rows:
        if r["applicable"]:
            by_answer.setdefault(r["good"], []).append((r, "n_good"))
            by_answer.setdefault(r["bad"], []).append((r, "n_bad"))

    start = time.perf_counter()
    lines = 0
    with zipfile.ZipFile(os.path.join(DATA, name, f"{name}.zip")) as archive:
        with archive.open(f"{name}/corpus.jsonl") as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                lines += 1
                if lines % 1000000 == 0:
                    print(f"  {name}: {lines:,} passages", flush=True)
                low = line.lower()
                line_toks = set(WORD.findall(low))
                cands = always + [a for t in line_toks & first_keys for a in by_first[t]]
                found = {a for a in cands if checks[a][0] <= line_toks and checks[a][1].search(low)}
                if not found:
                    continue
                doc = json.loads(line)
                tokens = set(WORD.findall(f"{doc.get('title', '')} {doc['text']}".lower()))
                for answer in found:
                    for r, field in by_answer.get(answer, []):
                        if doc["_id"] in gold.get(r["id"], ()):
                            continue
                        if len(r["words"] & tokens) >= r["need"]:
                            r[field] += 1
    seconds = time.perf_counter() - start
    return rows, lines, seconds


if __name__ == "__main__":
    names = sys.argv[1:] or ["nq", "hotpotqa", "msmarco"]
    pooled = dict(n=0, win=0, tie=0, loss=0, bad_zero_good_some=0, both_zero=0)
    results = {}
    for name in names:
        rows, lines, seconds = run(name)
        app = [r for r in rows if r["applicable"]]
        win = sum(r["n_good"] > r["n_bad"] for r in app)
        loss = sum(r["n_good"] < r["n_bad"] for r in app)
        tie = len(app) - win - loss
        bz = sum(r["n_bad"] == 0 and r["n_good"] > 0 for r in app)
        both = sum(r["n_bad"] == 0 and r["n_good"] == 0 for r in app)
        print(f"\n--- {name}: {lines:,} passages scanned in {seconds:.0f} s; applicable questions {len(app)} of {len(rows)} (others are yes/no) ---", flush=True)
        print(f" correct answer has more support: {win}; tie: {tie}; incorrect has more: {loss}")
        print(f" incorrect has zero support and correct has some: {bz}; both zero: {both}")
        print(f" median support: correct {np.median([r['n_good'] for r in app]):.0f}, incorrect {np.median([r['n_bad'] for r in app]):.0f}")
        for key, val in (("n", len(app)), ("win", win), ("tie", tie), ("loss", loss), ("bad_zero_good_some", bz), ("both_zero", both)):
            pooled[key] += val
        results[name] = [{k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()} for r in rows]
    n = pooled["n"]
    print(f"\n--- pooled over {', '.join(names)}: {n} applicable questions ---")
    print(f" correct more: {pooled['win']} ({pooled['win'] / n:.1%}); tie: {pooled['tie']} ({pooled['tie'] / n:.1%}); incorrect more: {pooled['loss']} ({pooled['loss'] / n:.1%})")
    print(f" incorrect zero while correct has some: {pooled['bad_zero_good_some']} ({pooled['bad_zero_good_some'] / n:.1%}); both zero: {pooled['both_zero']} ({pooled['both_zero'] / n:.1%})")
    ok = pooled["win"] / n >= 0.60 and pooled["loss"] / n <= 0.15
    print(f"\nverdict: {'CONTINUE to a reader-based test' if ok else 'STOP: the signal does not meet the bar'}")
    json.dump(results, open(os.path.join(DATA, "corroboration.json"), "w", encoding="utf-8"))
    print("\nCorroboration test finished.")
