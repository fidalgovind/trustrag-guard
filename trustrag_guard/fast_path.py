"""Fast path (Stage 3.5): run Layer 1 and Layer 2 side by side, then fuse.

Both layers use threads rather than processes because the embeddings and the
text are already in memory. Compiled work can overlap: the Numba ROUGE-L loop
releases the GIL (nogil=True), and so do parts of scikit-learn and numpy. Python
work cannot: tokenizing and K-Means bookkeeping still take turns, so the gain has
a limit. tests/bench_fast_path.py measures it.

scan_sequential() is the plain reference. scan_parallel() must return the same
result.

Known limit: this covers Layer 1, Layer 2 and fusion only. Retrieval arrives in
Stage 4, and Layer 3 is called after this, on the ambiguous chunks.
"""
import asyncio

from .fusion import RISK_HIGH, RISK_LOW, fuse
from .layers.clustering import DEFAULT_TAU
from .layers.layer1_scan import layer1_scan
from .layers.text_space import DEFAULT_COPY_CUTOFF, coordination_test


def scan_sequential(query_emb, chunk_embs, texts, threshold, clean_median,
                    tau=DEFAULT_TAU, seed=42, cutoff=DEFAULT_COPY_CUTOFF,
                    low=RISK_LOW, high=RISK_HIGH):
    """One layer after the other. `threshold` and `clean_median` are fitted
    offline from clean data (over_alignment)."""
    layer1 = layer1_scan(query_emb, chunk_embs, threshold, tau=tau, seed=seed)
    coordination = coordination_test(texts, cutoff=cutoff)
    return fuse(layer1, coordination, clean_median, low=low, high=high)


async def scan_parallel(query_emb, chunk_embs, texts, threshold, clean_median,
                        tau=DEFAULT_TAU, seed=42, cutoff=DEFAULT_COPY_CUTOFF,
                        low=RISK_LOW, high=RISK_HIGH):
    """Both layers at once on worker threads, then fusion. Takes the same
    arguments as scan_sequential()."""
    layer1, coordination = await asyncio.gather(
        asyncio.to_thread(layer1_scan, query_emb, chunk_embs, threshold, tau=tau, seed=seed),
        asyncio.to_thread(coordination_test, texts, cutoff=cutoff),
    )
    return fuse(layer1, coordination, clean_median, low=low, high=high)
