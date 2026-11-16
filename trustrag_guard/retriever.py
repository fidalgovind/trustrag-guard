"""Retriever wrapper: the middleware the application actually calls.

It over-retrieves from a Chroma collection with the stored embeddings, so no chunk
is embedded again online, runs guard(), and returns LangChain Documents. The
generator sees at most `target_k` chunks.

It wraps a Chroma collection and not a LangChain retriever, because the defense
needs the stored embeddings and a plain retriever does not return them.

Use `.invoke(query)` or `.ainvoke(query)`. `scan(query)` also returns the raw
GuardResult for calibration. Import it from this module: keeping it out of the
package `__init__` means the layers load without Chroma or LangChain.

Use TrustRAGGuard.calibrated() for the shipped thresholds. It refuses an embedder
they were not fitted on. Built directly, the wrapper keeps the placeholders.

Chroma's anonymous telemetry is on by default, so open_store() turns it off.

Known limits: a verifier failure stops the call. An empty result, or
`low_confidence` in a Document's metadata, is the caller's signal to abstain.
Fusion-kept chunks are cut to target_k in retrieval order, which is safe because
only unflagged chunks and floor fills are in the list.
"""
import asyncio
from typing import Any, Callable

import chromadb
import numpy as np
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .calibration.build_thresholds import load_reference
from .context_floor import DEFAULT_K_MIN
from .fusion import RISK_HIGH, RISK_LOW
from .layers.clustering import DEFAULT_TAU
from .layers.verifier import DEFAULT_MAX_VERIFIER_CALLS
from .pipeline import guard

DEFAULT_OVER_RETRIEVE = 30   # bottom of the k' = 30..50 range; Stage 6 settles it
DEFAULT_TARGET_K = 5         # chunks the generator receives, as in PoisonedRAG
UNKNOWN_SOURCE = "unknown"


def open_store(path=None, name="chunks"):
    """A cosine Chroma collection with telemetry off; in memory when `path` is None."""
    settings = Settings(anonymized_telemetry=False)
    client = (chromadb.PersistentClient(path=str(path), settings=settings) if path
              else chromadb.EphemeralClient(settings=settings))
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


class TrustRAGGuard(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection: Any
    embed: Callable         # text -> unit vector, the embedder that filled the store
    verifier: Any
    threshold: float        # fitted offline from clean data (over_alignment)
    clean_median: float
    over_retrieve: int = DEFAULT_OVER_RETRIEVE
    target_k: int = DEFAULT_TARGET_K
    k_min: int = DEFAULT_K_MIN
    max_calls: int = DEFAULT_MAX_VERIFIER_CALLS
    tau: float = DEFAULT_TAU
    low: float = RISK_LOW
    high: float = RISK_HIGH

    @classmethod
    def calibrated(cls, collection, embed, verifier, embedding_model, **settings):
        """Build with the shipped thresholds. `embedding_model` must be the one
        that filled the store."""
        v = load_reference(embedding_model)
        return cls(collection=collection, embed=embed, verifier=verifier,
                   threshold=v["threshold"], clean_median=v["clean_median"],
                   tau=v["tau"], low=v["low"], high=v["high"], **settings)

    def fetch(self, query):
        """Over-retrieve: (query embedding, chunk embeddings, texts, sources, ids)."""
        query_emb = np.asarray(self.embed(query), dtype=np.float64)
        hits = self.collection.query(query_embeddings=[query_emb.tolist()],
                                     n_results=self.over_retrieve,
                                     include=["documents", "metadatas", "embeddings"])
        metadata = hits["metadatas"][0]
        sources = [(m or {}).get("source", UNKNOWN_SOURCE) for m in metadata]
        embs = np.asarray(hits["embeddings"][0], dtype=np.float64).reshape(len(sources), len(query_emb))
        return query_emb, embs, list(hits["documents"][0]), sources, list(hits["ids"][0])

    async def scan(self, query):
        """Retrieve and guard. Returns (GuardResult, texts, sources, ids)."""
        query_emb, embs, texts, sources, ids = await asyncio.to_thread(self.fetch, query)
        result = await guard(query, query_emb, embs, texts, sources, self.verifier,
                             self.threshold, self.clean_median,
                             k_min=self.k_min, max_calls=self.max_calls,
                             tau=self.tau, low=self.low, high=self.high)
        return result, texts, sources, ids

    def to_documents(self, result, sources, ids):
        chosen = result.floor.selected[:self.target_k]
        filled = set(result.floor.filled.tolist())
        return [Document(page_content=result.chunks[n], id=ids[i],
                         metadata={"source": sources[i], "risk": float(result.fusion.risk[i]),
                                   "verified": int(i) in filled,
                                   "low_confidence": result.low_confidence})
                for n, i in enumerate(chosen)]

    async def _aget_relevant_documents(self, query, *, run_manager):
        result, _, sources, ids = await self.scan(query)
        return self.to_documents(result, sources, ids)

    def _get_relevant_documents(self, query, *, run_manager):
        result, _, sources, ids = asyncio.run(self.scan(query))
        return self.to_documents(result, sources, ids)
