"""In-memory pipeline: fast path, then verifier, then context floor.

This is the whole online decision in one call, given chunks and embeddings that
are already in memory. The store and the retriever wrapper arrive in a later step
and will call this.

Order matters. Fusion runs first, so the verifier only sees what fusion left
ambiguous, and the floor runs last, so it only sees final actions. The verifier
runs only as far as the floor needs.

The verifier runs on a worker thread, because a real one waits on a model for
hundreds of milliseconds and must not block other requests on the event loop.

Known limits: a verifier failure stops the call. When the pool has too few
usable chunks, `low_confidence` is the caller's signal to abstain, and nothing here
abstains on its behalf. The raw silhouette value is not carried through FusionResult.
"""
import asyncio
from dataclasses import dataclass

from .context_floor import DEFAULT_K_MIN, FloorResult, context_floor
from .fast_path import scan_parallel
from .fusion import RISK_HIGH, RISK_LOW, FusionResult
from .layers.clustering import DEFAULT_TAU
from .layers.text_space import DEFAULT_COPY_CUTOFF
from .layers.verifier import DEFAULT_MAX_VERIFIER_CALLS, VerifiedResult, verify_ambiguous


@dataclass
class GuardResult:
    chunks: list              # texts for the generator, in floor order
    low_confidence: bool      # True when fewer than k_min chunks were usable
    fusion: FusionResult      # raw per-chunk scores, kept for calibration
    verified: VerifiedResult  # verdicts and logprobs
    floor: FloorResult        # which chunks were kept and which were fills


async def guard(query, query_emb, chunk_embs, texts, sources, verifier, threshold, clean_median,
                k_min=DEFAULT_K_MIN, tau=DEFAULT_TAU, seed=42, cutoff=DEFAULT_COPY_CUTOFF,
                max_calls=DEFAULT_MAX_VERIFIER_CALLS, low=RISK_LOW, high=RISK_HIGH):
    """Online step. `threshold` and `clean_median` are fitted offline from clean
    data, and `sources` holds one provenance label per chunk."""
    if len(sources) != len(texts):  # checked here so no verifier call is wasted
        raise ValueError("texts and sources must cover the same chunks")

    fusion = await scan_parallel(query_emb, chunk_embs, texts, threshold, clean_median,
                                 tau=tau, seed=seed, cutoff=cutoff, low=low, high=high)
    verified = await asyncio.to_thread(verify_ambiguous, query, texts, fusion, verifier,
                                       k_min=k_min, max_calls=max_calls)
    floor = context_floor(verified.actions, sources, k_min=k_min, seed=seed)
    return GuardResult([texts[i] for i in floor.selected], floor.low_confidence,
                       fusion, verified, floor)
