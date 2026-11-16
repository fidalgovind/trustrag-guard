"""Shared loader for the learned-fusion scripts.

data/nq/questions.json predates the store builder's answer fields, so its targets lack
`correct` and `incorrect`. load_questions() adds them in memory from the PoisonedRAG
file, matched by id, after checking that question and poison texts agree. No data file
is changed, so the frozen results and their fingerprints stay valid.
"""
import json
import os

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def load_questions(name):
    path = os.path.join(DATA_ROOT, name, "questions.json")
    data = json.load(open(path, encoding="utf-8"))
    if any("incorrect" not in t for t in data["targets"]):
        source = json.load(open(os.path.join(DATA_ROOT, name, f"poisonedrag_{name}.json"), encoding="utf-8"))
        for t in data["targets"]:
            s = source[t["id"]]
            if s["question"] != t["question"] or s["adv_texts"] != t["adv_texts"]:
                raise ValueError(f"{name} {t['id']}: PoisonedRAG file disagrees with questions.json")
            t["correct"], t["incorrect"] = s["correct answer"], s["incorrect answer"]
    return data
