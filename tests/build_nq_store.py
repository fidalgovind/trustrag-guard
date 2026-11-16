"""Build a clean store for the real-pool measurements.

Usage: python tests/build_nq_store.py [nq|hotpotqa|msmarco]    (default nq)

Needs two files in data/<dataset>, which git ignores:
    <dataset>.zip               BEIR, CC BY-SA 4.0
    poisonedrag_<dataset>.json  PoisonedRAG's published attack texts, MIT licence

Keeps the gold passages of the 100 target questions and of 200 other test
questions, plus a random sample, about 50,000 passages. Writes a Chroma store and
questions.json beside it. Run it once. It stops if the store already exists.

The store is far smaller than the full corpus, so retrieval here is easier and
faster than at full size.
"""
import csv
import io
import json
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.retriever import open_store  # noqa: E402

DATASET = sys.argv[1] if len(sys.argv) > 1 else "nq"
DATA = os.path.join(os.path.dirname(__file__), "..", "data", DATASET)
COLLECTION = f"{DATASET}_clean"
N_REFERENCE = 200
TARGET_PASSAGES = 50_000
CORPUS_SIZE = {"nq": 2_681_468, "hotpotqa": 5_233_329, "msmarco": 8_841_823}[DATASET]
QRELS = "train" if DATASET == "msmarco" else "test"   # as PoisonedRAG does
SEED = 42
BATCH = 4_000


def read_lines(archive, name):
    with archive.open(name) as raw:
        yield from io.TextIOWrapper(raw, encoding="utf-8")


if __name__ == "__main__":
    store = open_store(os.path.join(DATA, "chroma"), COLLECTION)
    if store.count():
        raise SystemExit(f"store already holds {store.count()} passages")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    attacks = json.load(open(os.path.join(DATA, f"poisonedrag_{DATASET}.json"), encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    with zipfile.ZipFile(os.path.join(DATA, f"{DATASET}.zip")) as archive:
        qrels = {}
        for row in csv.DictReader(read_lines(archive, f"{DATASET}/qrels/{QRELS}.tsv"), delimiter="\t"):
            qrels.setdefault(row["query-id"], []).append(row["corpus-id"])
        questions = {}
        for line in read_lines(archive, f"{DATASET}/queries.jsonl"):
            q = json.loads(line)
            questions[q["_id"]] = q["text"]

        missing = [k for k in attacks if k not in qrels]
        changed = [k for k in attacks if k in questions
                   and questions[k].strip("? ").lower() != attacks[k]["question"].strip("? ").lower()]
        print(f"targets {len(attacks)}, without gold passages {len(missing)}, question text differs {len(changed)}")

        others = sorted(k for k in qrels if k not in attacks and k in questions)
        reference = [others[i] for i in rng.permutation(len(others))[:N_REFERENCE]]
        keep = {d for k in list(attacks) + reference for d in qrels.get(k, [])}

        rate = TARGET_PASSAGES / CORPUS_SIZE
        ids, texts, sources = [], [], []
        for line in read_lines(archive, f"{DATASET}/corpus.jsonl"):
            lucky = rng.random() < rate
            doc = json.loads(line)
            if lucky or doc["_id"] in keep:
                ids.append(doc["_id"])
                texts.append(doc["text"])
                sources.append(doc.get("title") or "untitled")
    print(f"passages kept {len(ids)}, of which gold {len(keep & set(ids))} of {len(keep)}")

    start = time.perf_counter()
    for i in range(0, len(ids), BATCH):
        part = range(i, min(i + BATCH, len(ids)))
        embs = model.encode([texts[j] for j in part], normalize_embeddings=True, batch_size=64)
        store.add(ids=[ids[j] for j in part], embeddings=embs.tolist(),
                  documents=[texts[j] for j in part],
                  metadatas=[{"source": sources[j]} for j in part])
        print(f"  stored {part[-1] + 1} of {len(ids)}, {time.perf_counter() - start:.0f} s", flush=True)

    out = {"targets": [{"id": k, "question": v["question"], "gold": qrels.get(k, []),
                        "adv_texts": v["adv_texts"], "incorrect": v["incorrect answer"],
                        "correct": v["correct answer"]} for k, v in attacks.items()],
           "reference": [{"id": k, "question": questions[k], "gold": qrels[k]} for k in reference]}
    json.dump(out, open(os.path.join(DATA, "questions.json"), "w", encoding="utf-8"))
    print(f"\nstore holds {store.count()} passages.\nStore build finished.")
