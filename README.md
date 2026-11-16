# TrustRAG-Guard

Low-latency adversarial-defense middleware for Retrieval-Augmented Generation (RAG).
It sits between the retriever and the generation LLM, scores every retrieved chunk
for "is this poison?" across vector, text, and generation space, and strips malicious
context in well under 150 ms — no retraining of the embedder or the LLM required.

It targets OWASP LLM08 (vector/embedding weaknesses), indirect prompt injection and
PoisonedRAG-style knowledge-base corruption. It does not reliably stop poison reworded to
avoid repeating the question or each other: see the adaptive-attack results in
`docs/EVALUATION_REPORT.md`.

> **Full design & rationale:** see [`DESIGN.md`](DESIGN.md). Read it before contributing —
> it explains the threat model, every layer, the known-fix list, and what makes this
> project original rather than a re-implementation of prior work.

## Status

Stages 0 to 7 are done for their agreed scope. Layer 1, Layer 2's ROUGE-L scoring,
fusion v1, the parallel fast path, the verifier, the context floor and an in-memory pipeline exist as plain
functions, and a LangChain retriever wraps them over Chroma, with conformal calibration on NQ. Detection
was evaluated on NQ, HotpotQA and MS-MARCO, and answers on HotpotQA and MS-MARCO. It met its latency
target, but not its ASR or CCR targets
(`docs/EVALUATION_REPORT.md`). A port of
TrustRAG had lower attack success on the same setup, under static and adaptive attacks, so this
is a research prototype and not ready for use. See `DESIGN.md` section 12 for
what has been measured.

## Quickstart

```bash
# 1. Clone
git clone https://github.com/fidalgovind/trustrag-guard.git
cd trustrag-guard

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 3. Install the exact tested environment
pip install -r requirements.txt

# or install the package itself (dependencies from pyproject.toml)
pip install -e .
```

Layer 3 (the verifier) additionally needs [Ollama](https://ollama.com) running locally
with a small model pulled, e.g. `ollama pull llama3.2:3b`. Ollama runs on your own
machine, not in CI — the rest of the pipeline is testable without it.

## Repository layout

```
trustrag_guard/          # the package
  embeddings.py          # Stage 1: embed + cosine primitives
  config.py              # GuardConfig: all settings in one validated object
  fusion.py              # one risk score per chunk: keep, ambiguous or drop
  fast_path.py           # Layer 1 and Layer 2 in parallel, then fusion
  context_floor.py       # fail-safe floor; never refills by similarity
  pipeline.py            # guard(): fast path, verifier, floor
  retriever.py           # TrustRAGGuard, the LangChain retriever over Chroma
  learned_fusion.py      # optional logistic-regression fusion (not the default)
  layers/
    clustering.py        # L1: k=2 silhouette split test
    over_alignment.py    # L1: signal A (query over-alignment)
    layer1_scan.py       # L1: runs both signals, one result
    fitted.py            # L1: Isolation Forest and Mahalanobis (tested, not in fusion)
    text_space.py        # L2: Numba ROUGE-L, coordinated-payload flag
    verifier.py          # L3: Verifier protocol, mock and Ollama verifier
  calibration/
    build_thresholds.py  # offline conformal calibration
    reference_thresholds.json  # shipped values (MiniLM on BEIR NQ)
  fixtures/
    toy_corpus.py        # tiny clean + poisoned corpus for fast tests
    synthetic_pool.py    # pseudo-word pools for the timing checks
tests/                   # standalone test scripts and benchmarks
docs/                    # evaluation report, literature reports, final report
DESIGN.md                # architecture + rationale (read this first)
pyproject.toml           # package metadata
requirements.txt         # exact tested environment
```

`trustrag_guard/calibration/reference_thresholds.json` holds the shipped thresholds. They
were fitted with `all-MiniLM-L6-v2` on a BEIR NQ subset, so refit them for any other embedder
or corpus. `TrustRAGGuard.calibrated()` refuses a different embedding model.

We add modules one stage at a time.

## Running tests

```bash
python tests/test_vector_space.py     # Stage 1: embeddings and cosine
python tests/test_config.py           # Stage 0: configuration object
python tests/test_clustering.py       # Stage 2: clustering baseline
python tests/test_over_alignment.py   # Stage 2: signal A
python tests/test_layer1_scan.py      # Stage 2: both signals together
python tests/test_text_space.py       # Stage 3: ROUGE-L and the flag
python tests/test_fusion.py           # Stage 3: fusion v1
python tests/test_fast_path.py        # Stage 3: parallel fast path
python tests/test_verifier.py         # Stage 4: verifier interface, mock only
python tests/test_context_floor.py    # Stage 4: fail-safe context floor
python tests/test_pipeline.py         # Stage 4: in-memory pipeline
python tests/test_ollama_verifier.py  # Stage 4: real verifier, needs Ollama for Parts B-D
python tests/bench_verifier.py        # Stage 4: verifier speed, needs Ollama
python tests/bench_verifier_length.py # Stage 4: verifier time against length
python tests/test_retriever.py        # Stage 4: retriever wrapper over Chroma
python tests/build_nq_store.py        # Stage 4: build the NQ store, needs data/nq
python tests/bench_real_pools.py      # Stage 4: real pools and latency, needs the store
python tests/bench_verifier_real.py   # Stage 4: verifier on real ambiguous chunks
python tests/test_calibration.py      # Stage 5: conformal calibration
python tests/bench_calibrated.py      # Stage 5: calibrated values on real pools
python tests/test_fitted.py           # Stage 5: fitted detectors, synthetic checks
python tests/bench_separation.py      # Stage 5: separation test, needs the HotpotQA store
python tests/bench_asr_curve.py       # Stage 5: ASR against CCR, needs Ollama
python tests/test_baseline.py         # Stage 6: TrustRAG baseline port, synthetic checks
python tests/bench_evaluation.py      # Stage 6: main evaluation, needs the MS-MARCO store
python tests/build_attacks.py         # Stage 7: adaptive attack text, evaluation only
python tests/bench_adaptive.py        # Stage 7: robustness table and verifier ablation
python tests/bench_l2_all_pairs.py    # Stage 3: Layer 2 speed check
python tests/bench_fast_path.py       # Stage 3: fast-path speed check
```

Part A of each script uses synthetic vectors. The later parts download
`all-MiniLM-L6-v2` once and skip gracefully if it cannot load.

## Team workflow

- `main` is the shared branch. Do day-to-day work on a feature branch
  (`git checkout -b stage2-vector-space`) and open a PR so changes are reviewed.
- Never commit `.venv/`, `thresholds.json`, or fitted model artifacts — they're in
  `.gitignore` and are regenerated locally.
- Each roadmap stage in `DESIGN.md` is meant to be a small, reviewable chunk.

## Licence

MIT, see [`LICENSE`](LICENSE). `tests/baseline_trustrag.py` ports code from TrustRAG
(Zhou et al.), also under the MIT licence.
