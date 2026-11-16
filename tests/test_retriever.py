"""Checks for the retriever wrapper over Chroma.

Part A - synthetic vectors, in-memory Chroma (no model): stored embeddings survive,
         the wrapper equals guard(), and provenance reaches the floor.
Part B - real embedder on the toy corpus, through `.invoke()`.

The verifier is a mock here. Real pools are measured in tests/bench_real_pools.py.
"""
import asyncio
import os
import sys
import uuid

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from langchain_core.retrievers import BaseRetriever  # noqa: E402
from trustrag_guard.fixtures.toy_corpus import OPENAI_CEO  # noqa: E402
from trustrag_guard.layers.text_space import warmup  # noqa: E402
from trustrag_guard.layers.verifier import MockVerifier  # noqa: E402
from trustrag_guard.pipeline import guard  # noqa: E402
from trustrag_guard.retriever import TrustRAGGuard, open_store  # noqa: E402

# Measured Stage 2 values, not fitted or tuned here.
THRESHOLD = 0.831
MEDIAN = 0.612


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def filled(n=12, seed=0):
    """A fresh in-memory collection of `n` random unit vectors."""
    rng = np.random.default_rng(seed)
    embs = unit(rng.normal(size=(n, 16)))
    texts = [f"passage number {i} about topic {i % 3}" for i in range(n)]
    store = open_store(name=f"t{uuid.uuid4().hex}")
    store.add(ids=[f"d{i}" for i in range(n)], embeddings=embs.tolist(), documents=texts,
              metadatas=[{"source": f"s{i % 4}"} for i in range(n)])
    return store, embs, texts


def wrapper(store, query_vec, **kw):
    return TrustRAGGuard(collection=store, embed=lambda _: query_vec, verifier=kw.pop("verifier", MockVerifier()),
                         threshold=0.8, clean_median=0.2, over_retrieve=8, **kw)


# ---------------------------------------------------------------- Part A
def part_a():
    store, embs, texts = filled()
    assert store._client.get_settings().anonymized_telemetry is False
    print("[ok] telemetry is off")
    query = unit(np.random.default_rng(1).normal(size=16))
    retriever = wrapper(store, query)
    assert isinstance(retriever, BaseRetriever)
    print("[ok] TrustRAGGuard is a LangChain BaseRetriever")

    q, got, got_texts, sources, ids = retriever.fetch("q")
    assert len(ids) == 8
    for vec, doc_id in zip(got, ids):
        assert np.allclose(vec, embs[int(doc_id[1:])], atol=1e-6), doc_id
    assert np.array_equal(np.argsort(-(got @ q)), np.arange(8)), "not in similarity order"
    print("[ok] stored embeddings come back unchanged, nearest first, with no re-embedding")

    result, *_ = asyncio.run(retriever.scan("q"))
    direct = asyncio.run(guard("q", q, got, got_texts, sources, MockVerifier(), 0.8, 0.2))
    assert result.chunks == direct.chunks and np.array_equal(result.fusion.risk, direct.fusion.risk)
    print("[ok] the wrapper equals guard() on the same chunks")

    assert sources == [f"s{int(i[1:]) % 4}" for i in ids]
    print("[ok] provenance reaches the floor from the chunk metadata")

    docs = wrapper(store, query, target_k=3).invoke("q")
    assert len(docs) == 3 and all(d.metadata["source"].startswith("s") for d in docs)
    assert docs[0].id in ids and not docs[0].metadata["low_confidence"]
    print("[ok] .invoke() returns at most target_k Documents, with metadata")

    docs = asyncio.run(wrapper(store, query, target_k=3).ainvoke("q"))
    assert len(docs) == 3
    print("[ok] .ainvoke() works too")

    class Broken:
        def verify(self, q, chunk):
            raise ConnectionError("verifier is down")

    # everything just over the threshold's lower band, so the verifier is needed
    near = unit(embs[0] + 0.0)
    try:
        TrustRAGGuard(collection=store, embed=lambda _: near, verifier=Broken(), threshold=2.0,
                      clean_median=-1.0, over_retrieve=8, k_min=8).invoke("q")
        raise AssertionError("expected ConnectionError")
    except ConnectionError:
        print("[ok] a verifier failure reaches the caller, so nothing passes unchecked")

    empty = open_store(name=f"t{uuid.uuid4().hex}")
    docs = wrapper(empty, query).invoke("q")
    assert docs == []
    print("[ok] empty store: no documents, no crash")


# ---------------------------------------------------------------- Part B
def part_b():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"[skip] real-model demo (could not load model: {type(e).__name__})")
        return
    enc = lambda texts: model.encode(list(texts), normalize_embeddings=True)
    docs = OPENAI_CEO.documents
    texts = [d.text for d in docs]
    stored = enc(texts)
    store = open_store(name=f"t{uuid.uuid4().hex}")
    store.add(ids=[str(i) for i in range(len(docs))], embeddings=stored.tolist(), documents=texts,
              metadatas=[{"source": d.source} for d in docs])
    retriever = TrustRAGGuard(collection=store, embed=lambda t: enc([t])[0], verifier=MockVerifier(),
                              threshold=THRESHOLD, clean_median=MEDIAN, over_retrieve=9)
    _, got, *_ = retriever.fetch(OPENAI_CEO.query)
    fresh = enc(retriever.fetch(OPENAI_CEO.query)[2])
    assert np.allclose(got, fresh, atol=1e-5)
    print("[ok] stored embeddings match the embedder's output for the same text")
    print(f"\n--- {OPENAI_CEO.query} ---")
    for d in retriever.invoke(OPENAI_CEO.query):
        print(f" risk {d.metadata['risk']:.3f}  {d.metadata['source']:<12} {d.page_content[:60]}")


if __name__ == "__main__":
    warmup()
    part_a()
    part_b()
    print("\nStep 4.5 checks finished.")
