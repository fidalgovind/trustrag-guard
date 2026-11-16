# TrustRAG-Guard — Architecture & Innovation Plan

**A design + study companion. Version 0.22 — 2026-11-08.**

> How to use this document: read it top to bottom once to get the shape of the
> project, then keep it open as a reference while we build each stage. Every
> section has a short **Why it matters** note so you learn the reasoning, not
> just the recipe. Anything marked **[FIX]** corrects a mistake in the original
> spec. Anything marked **[NEW]** is original work that separates us from the
> TrustRAG baseline. Section 12 records what Stage 2 actually measured.

---

## 0. The one-sentence pitch

TrustRAG-Guard is inline security middleware that sits between a RAG retriever
and the generation LLM, scores every retrieved chunk for "is this poison?"
across **three independent spaces** — vector, text, and generation — and strips
malicious context in under 150 ms, with **calibrated false-positive guarantees**
and an honest **adaptive-attacker** evaluation.

The two phrases in bold are what make it ours, not a re-run of prior work.

## 1. Problem & threat model

Standard RAG trusts whatever the retriever returns. An attacker who can write
into the knowledge base (a public wiki, a scraped web corpus, an enterprise
doc store with weak controls) injects **poisoned chunks** engineered to (a) rank
highly for a target query and (b) steer the generator toward an attacker-chosen
answer or instruction.

We defend against four attack shapes, and the defense is only interesting if we
name them precisely:

1. **Clumping poison** — several near-identical injected chunks optimized for the
   same target. They cluster tightly in vector space. (Clustering catches these;
   this is what TrustRAG already does well.)
2. **Dispersed / paraphrased poison** — the same payload reworded so the chunks
   do *not* cluster, defeating K-means-style detection. (Our hard case.)
3. **Indirect prompt injection** — a chunk carrying instructions ("ignore prior
   instructions and answer X") rather than false facts.
4. **Adaptive poison** — an attacker who knows our defense exists and optimizes
   to evade each specific layer. TrustRAG tests only two variants (section 2).
   We will test each layer.

**Why it matters:** every design choice below is justified by *which* of these
four it is meant to stop. If a layer doesn't map to a threat, it's dead weight.

---

## 2. The honest overlap with TrustRAG (and how we escape it)

TrustRAG (Zhou et al., 2025) is a two-stage defense. **Stage 1, Clean Retrieval:**
K-means with fixed k = 2 on the retrieved embeddings, filtering the high-density
group, with ROUGE-L used alongside it to avoid discarding a group that holds only
one malicious document. **Stage 2, Conflict Resolution:** the LLM generates its own
internal knowledge, then consolidates it with the retrieved documents and
self-assesses. (Checked against the paper; see the table at the end of this section.)

The original TrustRAG-Guard spec's Layer 1 (cluster the embeddings) and Layer 2
(ROUGE-L overlap) are *the same two ideas TrustRAG combines in its first stage*.
Adding an Isolation
Forest and a verifier LLM reads as an increment, not a new idea. A reviewer who
knows the field would call it "TrustRAG plus extras."

We escape this in two moves, both of which also happen to attack TrustRAG's real
weaknesses (dispersed attacks, and untested robustness):

- **[PRIOR ART: Cheng et al. 2025, see below] Query-conditional over-alignment (direction A).**
  TrustRAG looks at how documents relate to *each other*. We add how each
  document relates to the *query*. Because poison is optimized to win retrieval,
  it is often **anomalously** aligned to the query — more aligned than any genuine
  passage ever is. We learn the clean query→doc similarity distribution offline
  and flag chunks sitting in its extreme right tail. This is cheap (vectors
  already in hand). Stage 2 evidence (section 12): it catches query-stuffed
  poison that clustering misses, but not plain paraphrase that is not
  over-aligned.

- **[NEW] Signature evaluation — Adaptive attackers (direction D).** We don't only
  run the stock PoisonedRAG attack. We build attackers that *know* we cluster,
  that we ROUGE, that we verify — and try to slip past each. Showing we still
  hold up is a stronger, more honest result than a few points of ASR on a static
  benchmark. TrustRAG section 5.4 does test an "adaptive attack", so this is an
  extension, not a first: it removes the question from the poison and varies the
  poison's style, but it has no attacker aimed at each defense layer.

Roadmap originality (after the core works): **(B) conformal risk scoring** to
replace hand-tuned fusion weights with provable FPR bounds, and **(C) a
generation-space signal** (chunk influence / perplexity under the generator) so
we judge chunks across three spaces, not two.

### What the papers actually say (read 2026-11-05)

Sources: TrustRAG, arXiv:2501.00879v3, and `defend_module.py` in its public code
(github.com/HuichiZhou/TrustRAG); PoisonedRAG, arXiv:2402.07867v3.

| Question | Finding | What it means here |
| --- | --- | --- |
| Is TrustRAG k=2 K-means? | Yes, "K-means clustering (k = 2)" | Same base as our clustering step |
| Does it use a silhouette gate? | No; the word does not appear. Its code has a gate with the same job: K-means runs only if some pair of chunks scores ROUGE-L above 0.25 | Our gate is a different measure doing a similar job, so it is a variation, not a new idea |
| Which group does it treat as malicious? | The high-density group (higher mean intra-group cosine in its code), not the smaller one. If both groups are dense it returns nothing | Differs from our D4; matters for the majority failure (section 12, finding 5) |
| Where is ROUGE? | Stage 1, for intra-cluster similarity, not conflict checking | v0.3 and earlier had this wrong |
| Does it use query→document similarity as a detector? | No. In the paper, query similarity appears only as the attacker's goal. In the code, `k_mean_filtering` never receives the query, and its cosine thresholds compare documents with each other (0.88 in code; the paper says 0.85) | **Signal A is not in TrustRAG** |
| False-positive guarantee or conformal thresholds? | Neither term appears | Direction B is not in TrustRAG |
| Does it test without the question in the poison? | Yes, section 5.4: "removing the query questions from these documents", and "all the experiments are based on this setting" | **Signal A's weak spot** (finding 3). Stage 6 must run both settings. How this affects retrieval there is unclear, so check their code |
| PoisonedRAG black-box construction | "we propose to set S = Q, i.e., P = Q ⊕ I" | Confirmed. Real `I` is LLM-written, about 30 words (V = 30); our toy claims are shorter |
| PoisonedRAG defaults | N = 5 poison texts per question, k = 5 | At k = 5 poison can fill the whole retrieved set |
| Does retrieving more defend? | No: "around 10%" poison at k = 50 still reached 41% ASR on HotpotQA | Over-retrieval makes poison a minority, but is not a defense alone |
| Does hash-based duplicate removal work? | No: the ASR did not change, because each `I` differs | Text layer must score similarity, not exact matches |

### Wider literature search (2026-11-05): signal A's core is already published

- **Cheng et al., "Secure Retrieval-Augmented Generation against Poisoning
  Attacks" (IEEE BigData 2025, arXiv:2510.25025; one of three works called
  RAGuard)** publish the same hypothesis, direction and unit: a one-tailed
  upper-percentile test on query-to-text similarity, because poison tends to
  "exhibit unusually high similarity to Q". Differences: their reference is
  sampled from the database after injection, not a clean set, and the test is one
  of three OR-ed filters. Alone it missed 30.5% / 79.8% / 50.2% of poison on
  NQ / MS-MARCO / HotpotQA.
- **Kumar & Sahaym (preprint, Oct 2026, arXiv:2610.33849)** re-implement that filter
  with thresholds fitted on clean data, so the clean fit is also published, though
  only as a benchmark baseline with no result for the similarity test alone.
  Deleting the question prefix took the whole filter's AUROC from 0.953 to 0.758.
- **Gowda et al., MEMSAD (preprint 2026, arXiv:2605.03482)** is the closest
  standalone form, but for agent memory, at write time, against a query history.

**Signal A must be presented as a variation on Cheng et al., not as a new signal.**
Found to be open is evidence rather than the concept: no standalone evaluation of
a clean-fit, per-chunk, live-query test; no published poison-versus-clean
query-similarity overlap; and no test against an attacker who stops just above the
k-th clean score. The search was partial (no citation walk, no proceedings sweep),
so these are gaps in what was found, not claims of novelty. Full report:
`docs/Query similarity RAG defenses.md`.

### The project's claim (owner decision, 2026-11-05)

**The concept is prior art; our contribution is the honest measurement of it.** We
build on Cheng et al. and aim to deliver:

1. Signal A measured alone, with its realised false-positive rate.
2. The same on poison without the question prefix.
3. Adaptive attackers per layer, including one who stops just above the k-th clean
   score.
4. A published poison-versus-clean query-similarity overlap figure.
5. A per-chunk conformal bound. Conformal gates exist at query level (Li, Guo & Xu),
   but none was found on per-document query similarity.

Say "measured here", and do not say "first" unless a fuller search supports it.

## 3. Corrected system architecture

```
OFFLINE CALIBRATION (run once per corpus/embedder, writes thresholds.json)
──────────────────────────────────────────────────────────────────────────
 Clean corpus ─► chunk ─► embed ─► ingestion scan (dedup, outlier) ─► ChromaDB
 Clean benchmark queries ─► retrieve ─► measure on CLEAN data:
     • query→doc similarity distribution      (for signal A, direction A)
     • pairwise cohesion / silhouette baseline (L1)
     • PCA(384→~40) + Ledoit-Wolf covariance   (for Mahalanobis, L1)
     • Isolation Forest fit on clean embeddings (L1)
     • ROUGE-L overlap distribution            (L2)
     • conformal calibration scores            (fusion, direction B)
   ─────────────────────────────────────────► thresholds.json + fitted_models/

ONLINE FAST PATH  (target < 150 ms; realistic p50 budget below)
──────────────────────────────────────────────────────────────────────────
 user query
   └─► ChromaDB retrieve  k'=30..50  WITH include=["embeddings"]  (no re-embed)
         │
         ├───────────────── asyncio.gather(to_thread…) ─────────────────┐
         ▼                                                              ▼
   L1 Vector space                                            L2 Text space
   • pairwise cosine (FAISS, X·Xᵀ)                            • exact all-pairs scoring
   • silhouette k=2 split test  [FIX]                         • Numba ROUGE-L (warmed) [FIX]
   • Isolation Forest score (fit offline)                     • coordinated-payload flag
   • PCA→Mahalanobis vs clean   [FIX]
   • query→doc over-alignment   [signal A]
         └──────────────────────────┬──────────────────────────────────┘
                                     ▼
                 Risk fusion  →  conformal thresholds  [NEW direction B]
                                     │
              low (clean)        ambiguous band          high (poison)
                   │                   │                      │
                   │                   ▼                      ▼
                   │           L3 verifier (Llama 3.2 3B,   DROP chunk
                   │           T=0, 1-token logprob SAFE/UNSAFE)
                   └───────────────────┴──────────┐
                                                  ▼
         truncate to clean top-k  ─►  PROVENANCE-AWARE context floor [FIX]  ─► Generator
```

**Why it matters:** the biggest structural fix is pushing almost all *fitting*
(PCA, covariance, Isolation Forest, distributions, conformal scores) into the
offline stage. At query time we only *score* against pre-fit models — that's what
makes the latency budget real instead of aspirational.

## 4. Layer specs (corrected)

### Layer 1 — Vector space

**Goal:** catch clumping (threat 1) via clustering, dispersed poison (threat 2)
via density/distance outliers and over-alignment.

- **Over-retrieve k'=30–50**, not top-5. Density and clustering methods need
  statistical mass; 5 points can't support them.
- **Reuse stored embeddings** — `include=["embeddings"]` from ChromaDB. Never
  re-encode text on the hot path.
- **Pairwise cosine** via FAISS / a single `X·Xᵀ` on L2-normalized vectors.
- **[FIX] Silhouette split test.** Silhouette is *undefined for k=1*. The correct
  procedure: always fit **k=2**, compute that split's silhouette `s`. If
  `s < τ_sil` (calibrated, ~0.25 as a starting guess, not a law of nature) treat
  the set as one uniform blob → no attack cluster. If `s ≥ τ_sil`, the smaller/
  denser group is the candidate poison cluster (this rule fails when poison is
  the majority; section 12, finding 5). Consider **HDBSCAN** instead of
  K-means here: it's density-based, needs no fixed k, labels noise points, and
  moving off K-means further distances us from TrustRAG.
- **[FIX] Mahalanobis with reduction.** Raw Mahalanobis in 384-D is unstable and
  slow (384×384 covariance). Offline: fit `PCA(384 → ~40)` on the clean corpus,
  estimate covariance with **Ledoit-Wolf shrinkage**, store its inverse. Online:
  project the chunk, compute distance against the stored inverse. Faster *and*
  statistically sound.
- **Isolation Forest** fit **offline** on clean embeddings; online we only call
  `score_samples`. (Fitting per query was a hidden latency bomb.)
- **[signal A; prior art in section 2] Query→doc over-alignment.** Offline, record the distribution
  of cosine(query, doc) over clean retrievals. Online, flag any chunk whose
  query similarity sits in the extreme right tail (e.g. above the clean
  99th percentile). Rationale: honest passages are relevant; poison is
  *optimized*, so it overshoots. The threshold must come from this clean
  reference, never from the retrieved pool itself (section 12, finding 1). This
  is the most promising addition so far, but section 12 records what it does and
  does not catch.

### Layer 2 — Text space

**Goal:** catch coordinated payloads (threat 1) and ROUGE-evading rephrasings,
and set up the generation-space check for injection text (threat 3).

- **[FIX] No prefilter.** The original spec used MinHash/LSH to avoid O(N²)
  comparisons. Measured, plain all-pairs ROUGE-L missed the limit (section 12,
  finding 7), but bit-parallel LCS met it while scoring every pair exactly. That
  needs no dependency and no similarity floor, and it cannot drop a pair.
- **[FIX] Numba ROUGE-L needs warmup.** `@njit(nogil=True)` compiles on the
  *first* call (seconds). Trigger one dummy call on `int32` arrays at startup, or
  the first real request's p99 is terrible. Operate on pre-tokenized int arrays;
  compute LCS for ROUGE-L; `nogil=True` lets L1 and L2 truly run in parallel.
- Score each chunk by its highest ROUGE-L with any other chunk (`coordination_test`).
  The graded score goes to fusion. It must not decide alone: honest template text
  scores high too (section 12, finding 8).

### Layer 3 — Deterministic verifier (circuit breaker)

**Goal:** adjudicate the ambiguous band and catch hidden instruction overrides.

- **Selective** — only runs on chunks whose fused risk is in the ambiguous band
  (provisionally 0.30–0.85; the exact bounds come from conformal calibration,
  not by hand). Fusion v1 sent up to 7 of 9 on a toy pool (section 12, finding 12), so
  calls are limited: none once fusion kept `k_min` chunks, otherwise at most 3, lowest
  fused risk first, stopping when the floor can be filled. An unchecked chunk stays
  unusable. The value 3 is a proposal for Stage 6 to check.
- **Llama 3.2 3B (or 3.1 8B) via Ollama, T=0.0**, prompted for a `SAFE`/`UNSAFE`
  verdict; read the **logprob** of the first token rather than generating prose.
  `UNSAFE` is not one token (the model emits `UN`, then `SAFE`), so the first token
  decides, and anything unexpected drops the chunk. It is not low latency on a CPU
  (section 7), and it is not a fact checker (section 12, finding 15).
- Prompt must show the chunk *and* the query, and explicitly ask whether the
  chunk contains instructions directed at the model or claims contradicting
  well-established knowledge.

## 5. Risk fusion & the fail-safe (two important fixes)

### [FIX / NEW-B] Fusion: conformal instead of hand-tuned weights

The original "weighted ensemble 0.0–1.0 with the thresholds 0.30 and 0.85" means
three magic numbers that won't transfer across corpora or embedders. Instead:

- Each layer emits a raw anomaly score. Offline, run the whole clean calibration
  set through the layers and record the scores.
- Use **conformal prediction**: pick the operating threshold as an empirical
  quantile of the clean score distribution so that *at most* a chosen fraction
  (say 5%) of clean chunks ever cross it. That gives a **provable bound on the
  false-positive rate** — a real guarantee, which almost no RAG defense offers.
- The ambiguous band (where L3 fires) becomes the region between the "almost
  certainly clean" and "almost certainly poison" conformal quantiles, derived
  from data rather than guessed.

Start simple: a transparent weighted sum *is* fine for v1 so we can get
end-to-end quickly — but record calibration data from day one so swapping in
conformal thresholds later is a config change, not a rewrite.

### [FIX] Fail-safe / context floor must not fall back to similarity

Original policy: if filtering drops candidates below `k_min`, keep the highest-
similarity survivors. **This can hand the generator the poison** — PoisonedRAG
poison is engineered to have the *highest* query similarity, so "keep top
similarity" preserves exactly what we removed. Correct fallback order:

1. Prefer chunks that **no layer flagged** at all.
2. Among those, prefer **provenance diversity** — chunks from distinct sources,
   so one corrupted source can't dominate.
3. If still below `k_min`, **signal low confidence to the caller** (or
   abstain) rather than silently backfilling with suspect chunks.

**A SAFE verdict only fills the floor.** A chunk the verifier called SAFE enters the
context only when fusion-kept chunks are below `k_min`, and only up to `k_min`. It
never overrides a fusion flag otherwise. This limits the damage from a wrong verdict,
at a measured cost (section 12, finding 13).

## 6. Evaluation protocol (with the adaptive attacker)

**Datasets:** PoisonedRAG (NQ, HotpotQA, MS-MARCO; 5 poisoned chunks/question),
InjecAgent + BIPIA (indirect injection), BEIR (corpus-level poisoning).

**Primary metrics:**
- **ASR** (Attack Success Rate) — queries where the LLM emits the attacker's
  target answer. Target **< 5%**.
- **CCR** (Clean Context Recall) — fraction of genuine supporting chunks retained.
  Measured here on chunks in the undefended top 5, and reported separately from the
  question-level count.
  Target **> 95%**.
- **FPR** — clean chunks wrongly flagged (this is what conformal bounds).
- **Latency** — p50/p95/p99 per layer.

**[NEW — direction D] Adaptive evaluation.** For each layer, build an attacker
that targets its assumption and measure ASR *under adaptation*:
- vs clustering → **dispersed** poison (paraphrased, spread out). Signal A only
  catches it if it is over-aligned (section 12), so this test must also measure
  how much query alignment an attacker can drop and still win retrieval.
- vs ROUGE-L → **lexically diverse** payloads with the same semantic target.
- vs the verifier → payloads that read as benign facts, no overt instructions.
Report a table of ASR for static vs adaptive attacks per layer. The story we
wanted was that the ensemble and signal A degrade gracefully. **Measured in Stage 7
(section 12, finding 24), that story is not supported:** the Guard stayed behind
TrustRAG under every attack.

---

## 7. Latency budget (honest version)

```
Retrieval (ChromaDB, k'=30–50, embeddings included) : depends on store, ~5–20 ms
L1 vector (FAISS cosine + offline-fit iForest/Maha   : ~5–15 ms
   score + silhouette + signal A)
L2 text (warmed Numba all-pairs ROUGE-L)              : ~10–25 ms   (parallel w/ L1)
Fusion + gating                                       : ~1–2 ms
──────────────────────────────────────────────────────────────────
FAST PATH (L1‖L2)                                      : ~20–45 ms
L3 escalation (Llama 3.2 3B, 1-token logprob,         : ~120–300 ms
   on 0–3 chunks, only when ambiguous)
```

These are the original estimates. **Measured in Stage 3.5, on one laptop with a
synthetic pool (section 12, findings 10 and 11):** Layer 1 is 10 ms at p50, without
Isolation Forest or Mahalanobis, and Layer 2 is 7 ms. L1 parallel with L2, plus
fusion, is 16 ms at p50 and 22.1 ms at worst p99. Embedding the query adds 11 ms.

**The L3 estimate above was wrong.** Measured on CPU, on mains: 448 ms at p50 on
chunks of about 15 words, and 3,576 ms at p50 on 200-word chunks. Prompt time grows
with chunk length: about 13.5 ms per new prompt token on CPU, and about 4.6 ms on the
integrated GPU if it is enabled (section 12, finding 16). With Chroma on 50,000
passages, the whole fast path took 25.6 ms at p50 on mains (finding 17).

Caveats baked in: (1) **first-call Numba compile is excluded** because we warm it
at startup; (2) iForest/PCA/Mahalanobis costs assume **offline fit, online score
only**; (3) L3 blows the 150 ms budget *per chunk*, which is why it's selective —
if too many chunks land in the ambiguous band, that's a calibration smell, not a
reason to make L3 faster.

## 8. Environment notes (read before coding)

- **Python:** 3.12.3 on the development machine, which matches the spec target.
  Avoid version-specific syntax. Record the real version in `pyproject.toml`.
- **Ollama runs on the host, not in this sandbox.** Layer 3 can't be executed
  from inside the VM. We develop L3 against a **mockable interface** (a
  `Verifier` protocol) so L1/L2/fusion are fully testable without Ollama, and
  you run the real Llama verification on your own machine.
- Keep heavy fitted artifacts (`fitted_models/`, PCA, covariance, iForest) out of
  git or tracked with care — they're regenerable from calibration.

---

## 9. Proposed repo structure

```
trustrag_guard/
  __init__.py
  embeddings.py        # embed + cosine primitives                (Stage 1, done)
  config.py            # dataclasses; loads thresholds.json
  retriever.py         # TrustRAGGuard wrapper (BaseRetriever interface)
  layers/
    clustering.py      # L1: k=2 silhouette split test            (Stage 2, done)
    over_alignment.py  # L1: signal A, clean-reference threshold  (Stage 2, done)
    layer1_scan.py     # L1: runs both signals, one result        (Stage 2, done)
    text_space.py      # L2: Numba ROUGE-L, all pairs (+ warmup())
    verifier.py        # L3: Verifier protocol + Mock (Stage 4, done); Ollama impl next
  context_floor.py     # fail-safe floor; never sees similarity      (Stage 4, done)
  pipeline.py          # fast path + verifier + floor, in memory     (Stage 4, done)
  fusion.py            # score fusion + conformal thresholds + gating
  fast_path.py         # L1 and L2 in parallel, then fusion          (Stage 3, done)
  calibration/
    build_thresholds.py# offline: fits models, writes thresholds.json
    reference_thresholds.json # shipped default, labelled; MiniLM on NQ only
  fixtures/
    toy_corpus.py      # tiny clean corpus + hand-made poison for fast tests
    synthetic_pool.py  # pseudo-word pool for the timing gates
tests/
  test_vector_space.py # (Stage 1 — done)
  test_clustering.py   # (Stage 2 — done)
  test_over_alignment.py  # (Stage 2 — done)
  test_layer1_scan.py  # (Stage 2 — done)
  test_text_overlap.py
  test_fusion.py
  test_end_to_end.py
thresholds.json        # local working copy, git-ignored
pyproject.toml
DESIGN.md              # this file
```

Isolation Forest / Mahalanobis will join `layer1_scan.py` in Stage 5 once a clean
reference exists to fit them on.

---

## 10. Corrected roadmap

```
STAGE 0  Scaffold + toy fixture + Verifier protocol + calibration stub   [partial]
         (toy fixture done; the other stubs are built with the stage that needs them)
STAGE 1  Embeddings & cosine                                             [done]
STAGE 2  L1 vector space
         • silhouette k=2 split test            [FIX]          done
         • query→doc over-alignment              [signal A]     done
         • layer1_scan combiner                                 done
         • HDBSCAN / iForest, PCA → Mahalanobis (offline fit)   deferred to Stage 5
           (they need a clean reference to fit on)
         • over-retrieval k'=30                                 not yet built
           (needs the real retriever, Stage 4)
STAGE 3  L2 text space (warmed Numba ROUGE-L) + weighted fusion v1
STAGE 4  L3 verifier (mock first, Ollama on host) + BaseRetriever wrapper
STAGE 5  Calibration spine → conformal thresholds   [NEW direction B]
STAGE 6  Evaluation harness: PoisonedRAG + static metrics
STAGE 7  Adaptive attackers + robustness table      [NEW direction D]  ← our edge
```

## 11. Study notes (the concepts, in plain language)

Keep this section as your glossary while we build. Each entry is "what it is →
why we use it here."

- **Cosine similarity** → angle between two embedding vectors; 1 = same meaning,
  0 = unrelated. On L2-normalized vectors it's just a dot product, so the whole
  pairwise matrix is one `X·Xᵀ`.
- **Silhouette score** → for a proposed clustering, how well each point fits its
  own cluster vs the nearest other cluster, in [-1, 1]. Needs ≥2 clusters, which
  is exactly why "k=1" can't have one. We use a 2-cluster split's silhouette as a
  *detector*: high → a real second group (candidate poison) exists.
- **HDBSCAN** → density-based clustering that finds clusters of varying density,
  needs no preset k, and labels sparse points as noise. Good for "is there a
  dense injected blob?" without assuming how many clusters.
- **Isolation Forest** → randomly partitions space; points that get isolated in
  few splits are outliers. Fit on clean data, it scores how "surprising" a chunk
  is. Catches dispersed poison that doesn't form a clean cluster.
- **Mahalanobis distance** → distance from the center of a distribution that
  accounts for its shape (correlations/variances), unlike plain Euclidean. In
  high dimensions it needs a well-estimated covariance — hence PCA first.
- **PCA** → finds the few directions that capture most variance; projecting
  384-D → ~40-D makes covariance estimation stable and distance math fast.
- **Ledoit-Wolf shrinkage** → a way to estimate a covariance matrix reliably from
  limited samples by blending the sample estimate toward a simple target.
- **Bit-parallel LCS** → computes the same LCS table by updating 64 cells at a
  time inside one machine word, so it is about 64 times less work with the same
  exact answer. It is why we need no MinHash/LSH prefilter.
- **ROUGE-L / LCS** → ROUGE-L is built on the Longest Common Subsequence between
  two token sequences; high ROUGE-L between chunks = they share long ordered
  spans = likely coordinated copies.
- **Conformal prediction** → uses a calibration set to set thresholds with a
  statistical guarantee (e.g., "≤5% of clean items flagged"). Turns our magic
  numbers into a bound.
- **Logprob verdict** → instead of letting the LLM write a sentence, we read the
  probability it assigns to the single token `SAFE` vs `UNSAFE`. Faster and gives
  a confidence number for free.
- **ASR / CCR / FPR** → did the attack get through / did we keep the good context
  / did we wrongly drop clean context. The three numbers that define success.

---

## 12. Stage 2 findings (measured on the toy corpus)

**Setup.** One query ("Who is the CEO of OpenAI?") over a 9-chunk toy pool:
4 clean, 2 clumping, 2 dispersed and 1 injection chunk, embedded with
all-MiniLM-L6-v2 on the developer's machine. The signal A threshold is the 99th
percentile of query→doc similarity over a clean reference of 40 values (8
unrelated queries × top-5 of a 32-passage clean mini knowledge base): **0.831**.
The chunks are the project's own fixture,
`trustrag_guard/fixtures/toy_corpus.py`, so these numbers reproduce exactly.

| Poison type | Similarity to query | Clustering (k=2 silhouette) | Signal A |
| --- | --- | --- | --- |
| Hand-written paraphrase / injection (5 chunks) | 0.697–0.798 | caught 0 of 5 | caught 0 of 5 |
| Query-stuffed, PoisonedRAG-style (4 chunks) | 0.850–0.894 | caught 4 of 4, 0 clean flagged (4-vs-4 tie, settled by density) | caught 4 of 4, 0 clean flagged |
| Query-stuffed, poison is the majority (5 chunks) | 0.850–0.894 | caught 0 of 5, **all 4 clean flagged** | caught 5 of 5, 0 clean flagged |

The clean chunks scored 0.693 or below in every run. Signal A never flagged a
clean chunk; clustering did, in the majority pool (finding 5). A clean-only pool
of the 4 clean chunks gave silhouette 0.175, no split and nothing flagged.
The stuffed poison was built by prepending the query text to the same false
claims, which matches PoisonedRAG's black-box construction `P = Q ⊕ I`
(paper section 4.2.2).

**Findings**

1. **Pool-relative thresholds fail when poison is the majority.** Poison is 5 of
   9 chunks here, so the pool median (0.697) is already inside the poison. A
   median/MAD score on the Stage 1 similarities flags nothing (largest robust
   z ≈ 1.5 against a usual cutoff of 3.5). Thresholds must come from a clean
   reference fitted offline.
2. **The silhouette cutoff is fragile.** The k=2 silhouette was 0.248 against a
   0.25 cutoff, so clustering reported "no attack cluster". Removing the three
   words "none other than" from one chunk moved it to 0.279 and flipped the
   decision, while changing the seed (0-19) had no effect. We do not tune τ on
   9 points; Stage 5 calibrates it.
3. **Signal A catches only over-aligned poison.** Unstuffed poison scored about
   0.80 and slipped under 0.831; stuffed poison scored 0.85 and above. An attacker
   who knows the threshold can stuff less. Hypothesis, not yet measured: weaker
   stuffing also lowers the chance of winning retrieval, so evading costs the
   attacker strength. Stage 7 should measure ASR against the similarity margin.
4. **A flat similarity threshold is crude.** The genuine answer scores only 0.693
   for this query, while the clean reference reaches 0.837 (a short passage that
   directly answers its question). One global threshold therefore sits high and
   does not transfer across queries. Per-query normalization is a candidate fix
   to evaluate in Stage 5.
5. **"Smaller group = poison" fails when poison is the majority.** With 5 stuffed
   chunks against 4 clean, k=2 separates poison from clean (silhouette 0.404), but
   the smaller group is the clean one, so clustering flags all 4 clean chunks.
   Combined with signal A by OR, all 9 chunks are flagged and nothing is left.
   With 2, 3 or 4 stuffed chunks the rule works. Real pools of 30-50 chunks should
   make poison a minority, but that is unmeasured, so until then clustering must
   not be able to remove a chunk by itself.
6. **ROUGE-L separates the copied poison on the toy corpus.** The clumping pair
   scored 0.778 and the dispersed pair 0.370, against a clean-clean maximum of
   0.240 (Stage 3.1). Stuffing raised the poison-poison mean from 0.265 to 0.512
   and left clean-poison flat (0.139 to 0.142). The injection chunk does not stand
   out, and no threshold has been set.
7. **All-pairs ROUGE-L fits the budget without a prefilter.** The limit was set
   before measuring: p99 at most 12.5 ms on 50 chunks of 200 words, covering
   tokenizing and scoring all 1,225 pairs. Plain LCS failed at 48.8 ms. Bit-parallel
   LCS passed at 6.3 to 9.9 ms over four runs on one laptop. A 400-word pool came
   out at 14 to 22 ms, over this limit, so very long chunks need another look.
8. **The coordinated-payload flag is cleaner than clustering, but not safe alone.**
   At a starting cutoff of 0.5, chosen without looking at toy results, it caught the
   2 clumping chunks of 5 on the fixture, with 0 clean flagged. On the 5-stuffed
   majority pool, where clustering flagged all 4 clean chunks, it caught 4 of 5 with
   0 clean flagged. It did not catch either dispersed chunk or the injection chunk
   on the fixture. **On the 32 clean reference passages it wrongly flagged 6 (19%)**,
   because short template sentences ("Berlin is the capital of Germany") share most
   of their words. Those passages are not a retrieved pool, but the risk is real.
9. **Fusion v1 removes the failure that clustering alone caused.** Weights are 0.85
   signal A, 0.10 coordination and 0.05 clustering, with the spec's 0.30/0.85 band,
   all placeholders. No clean chunk was dropped in any toy pool, including the
   majority pool, and all stuffed poison was dropped (4 of 4, 5 of 5). All 5
   paraphrase chunks reached the ambiguous band, none dropped and none kept.
   **Two costs:** the real answer also lands in the ambiguous band (risk 0.337), so
   the paraphrase pool sends 7 of 9 chunks to the verifier, more than the 0-3
   expected, and section 7 treats that as a calibration smell. On clean-only pools 1
   of 160 chunks was dropped, in-sample. Fusion alone stops no paraphrase poison;
   everything depends on Layer 3.
10. **The fast path meets its first limit, and Layer 1 set its speed.** The limit
    was set before measuring: p99 at most 75 ms for Layer 1, Layer 2 and fusion on 50
    chunks of 200 words, in all of three runs of 500 rounds. It passed at a worst p99
    of 48.6 ms. Chunk embeddings were precomputed, and retrieval and warm-up were
    excluded. Parallel running saved only 0.1 to 1.9 ms at p99, because the path could
    not beat Layer 1 at 32 ms p50, well over its 5-15 ms estimate.
11. **One-thread K-Means more than halved the fast path, and parallel stopped
    helping.** K-Means was starting 16 threads for 50 points. On one thread the worst
    p99 fell to 22.1 ms, or 38.7 ms with the query embedded, and no test output or
    clustering decision changed. An A/B test in one process gave 39.6 ms at p50 with
    default threads and 15.4 ms with one. **Parallel no longer meets its rule:** it
    saves about 0.4 ms at p50. The pool is pseudo-word text, and this is one laptop.
12. **Layer 3 now decides the paraphrase case, and it can also lose the answer.** With
    a scripted verifier, fusion sent 2 chunks per toy pool, except 7 of 9 on the
    paraphrase pool. Only there does the verdict decide which chunks survive the
    verifier: 0 of 5 poison chunks survive one that is always right, and 5 of 5 survive
    one that always says SAFE. Finding 14 limits what that means for the generator. **The two clean chunks sent are the only two naming the correct
    answer,** so a verifier that always says UNSAFE removes the answer along with the
    poison. These are bounds from a mock. No real model has judged a chunk yet.
13. **The floor stops wrongly SAFE poison when fusion kept enough, and it loses the
    answer on the toy pools.** At `k_min` 2, 0 of 5 wrongly SAFE poison chunks reached
    the generator, because fusion kept 2 chunks. With nothing kept, the fill was capped
    at 2 and both were poison. **At `k_min` 2 the correct answer reached the generator
    on no toy pool, even with an always-right verifier,** because the kept chunks do
    not name it and the chunks that do are ambiguous. At `k_min` 4, an always-right
    verifier delivers the answer, but an always-SAFE one puts 2 poison chunks in the
    context. The root cause is the band's lower edge, which Stage 5 must fix. Source
    diversity counts labels, so several fake sources can beat one honest source.
14. **At `k_min` 2 the verifier's verdict changes nothing on the toy pools.** Three
    scripted verifiers, always right, always SAFE and always UNSAFE, gave the same
    context on all four pools, because fusion's 2 kept chunks meet the floor. This
    qualifies finding 9: paraphrase poison stays out of the context here because of
    the floor, not because of Layer 3. The whole pipeline met the 75 ms limit at a worst
    p99 of 19.8 ms, but that pool made 0 verifier calls, so it times the wiring only.
15. **The real verifier is perfect on the toy chunks, weak off them, and slow on CPU.**
    With one prompt, fixed before it saw a toy chunk, Llama 3.2 3B called all 4 clean
    chunks SAFE and all 10 poison chunks UNSAFE, including the injection chunk that no
    earlier layer flagged. **That result does not generalise.** On 18 control sentences
    outside the toy corpus, it passed 7 of 8 false claims, such as "Sydney is the
    capital of Australia", and flagged 3 of 4 honest how-to texts. It is not a fact
    checker: the toy poison is a famous falsehood, which is the easy case. Time per
    chunk was 448 ms at p50 on short chunks and 3,576 ms on 200-word chunks, on CPU and
    on mains, against an estimate of 120-300 ms. The control was written by hand and
    is small, so these are warnings and not rates. Attacks aimed at the verifier are
    untested.
16. **Verifier time is a straight line in prompt tokens, and the integrated GPU cuts
    it by about 2.7 times.** Over 25 to 400 words, the CPU took 60 ms + 13.46 ms per new
    token, and the Radeon 680M through Vulkan took 157 ms + 4.59 ms per new token. At
    200 words, the median was 3,773 ms against 1,373 ms, and all 32 toy and control
    verdicts matched. Ollama drops integrated GPUs by default, so it needs
    `OLLAMA_IGPU_ENABLE=1`. Thirty samples per length support medians, not tail
    percentiles. Each server ran once, on one laptop.
17. **On real pools the defense removes most poison, and it loses most answers.** This
    used 100 PoisonedRAG NQ questions with their published attack texts, over 50,076
    NQ passages, k' = 30 and top 5. Poison in the top 5 fell from 429 of 500 to 51 with
    the question removed, and from 488 to 1 with it included. That confirms signal A's
    weak spot: it flagged 15% of the first form and 50% of the second. **Gold reached
    the top 5 for 34 questions on clean pools, against 90 undefended.** Fusion kept at
    least `k_min` on all 300 pools, so the floor was always met, the verifier made 0
    calls, and ambiguous chunks, including 62 of 108 gold passages, were lost. Clustering
    never split (highest silhouette 0.228 against tau 0.25). The fast path, with
    retrieval and without the verifier, took 25.6 ms at p50 and 31.5 ms at p99 on mains. This is a small subset with no generator, so it is not ASR. The band and D34
    must be revisited before this can be called usable.
18. **The real verifier protects answers and misses half the poison.** On every ambiguous
    chunk of the finding 17 pools, it called all 62 gold chunks SAFE and 43 of 48 other
    clean chunks, but it also called 191 of 344 published poison chunks SAFE, and 82 of
    171 with the question. If SAFE chunks competed for the top 5, gold would rise from
    34 to 85 questions on clean pools, and poison would rise from 51 to 221 as published
    (76 questions) and from 1 to 82 with the question (35 questions). Neither
    setting is acceptable, so this is a real trade-off, and the prompt and the band
    have not been calibrated. This is a what-if on the test questions, with no generator.
19. **Conformal calibration keeps its promise, and shows what the signals cannot do.**
    Fitted on 100 clean questions, it held on 100 others: 5.2% of top chunks reached the
    band against 5%, 0.8% were dropped against 1%. Clean gold rose from 34 to 73 questions,
    against 90 undefended. With the question in the poison, it stayed strong: 39 poison
    against 488 undefended, and gold 71 against 11. **With the question removed, poison
    rose from 51 to 234, reaching 77 of 100 questions,** about half of undefended. Those
    texts score like ordinary relevant passages, so a clean-based threshold cannot
    separate them, and the placeholder band only looked stronger by discarding relevant
    text. This matches finding 3's weak spot, now at real scale.
20. **The conformal promise is per chunk, and its false alarms land on the right answers.**
    All 17 clean questions lost in finding 19 lost their rank-1 passage. Gold chunks
    crossed the band edge at 17.5%, against 1.8% for other top chunks, because gold is
    usually the most similar chunk and risk is mostly similarity. **CCR is 82.5% on test
    questions and 77.4% held out, against a target above 95%.** Keeping 95% would need a
    low edge at about signal A's threshold, leaving no band. On NQ, poison and answers
    compete for the same high-similarity range. Finding 22 tests that on another
    corpus. Finding 19's counts are chunks out of 500 for poison and questions out of
    100 for gold.
21. **The fitted detectors failed a test fixed in advance.** On unused HotpotQA questions,
    fitted on clean HotpotQA, Isolation Forest scored AUROC 0.439 and Mahalanobis 0.459
    on published poison against rank-1 correct passages, about chance. They caught 4.0%
    and 3.0%. Published poison reads like the corpus, so it is not an outlier. They are
    not in fusion. Similarity did better here (AUROC 0.90), so the NQ claim in finding
    20 does not generalise, though it catches only 30% at a 5% false-alarm cutoff.
22. **No operating point meets both targets against question-removed poison.** Over risk
    cutoffs on the same questions, with a stand-in generator: CCR passes 95% at cutoff
    0.80, where published ASR is 57 of 100. ASR reaches its clean level of about 12 only at
    cutoff 0.30, where CCR is 49%. The calibrated point gives CCR 85.4% and ASR 32. With
    the question in the poison, cutoff 0.70 gives CCR 94.7% and ASR 14, so that form is
    nearly solved. The ASR check fires on 9 to 13 clean questions, because many
    answers are `yes` or `no`, so ASR below 5% cannot be read from it directly.
23. **In the main evaluation TrustRAG wins on attack success, and the Guard wins on speed.**
    On unused MS-MARCO questions, on one shared setup with a stand-in generator, ASR
    out of 100 was 71 for the Guard against 28 to 33 for TrustRAG on published poison,
    and 34 against 14 to 22 with the question. Undefended was 93 and 86, and clean
    ASR was 10. The ASR and CCR targets were not met by any system. CCR on clean pools
    was 78.8% for the Guard and 86.9% for TrustRAG. The fast path met its target at p99
    47.1 ms, against a p50 of 483 ms for TrustRAG's first stage on CPU. TrustRAG returned
    an empty context for 44 and 87 of 100 attacked questions, so its generator
    often answered from its own knowledge. Overlap among the top 5 suits PoisonedRAG's
    repetitive texts, and that signal has no decision power in the Guard.
24. **Under adaptive attack the Guard stays behind TrustRAG, and graceful degradation is
    not supported.** Each attacker had one budget and ran against every system. ASR out
    of 100, clean ASR 10: `diverse`, aimed at TrustRAG, gave 84 for the Guard against 50
    and 36. `capped`, aimed at signal A, gave 53 against 29 and 28. `benign`, aimed at
    the verifier, gave 71 against 22 and 22. Every paired difference had an interval
    above zero. The rewording attacker hit its own goal on only 25 of 100 questions, so
    TrustRAG's robustness to rewording is untested at full strength.
25. **The verifier, once forced to run, buys answers and not safety.** On up to 3
    borderline chunks per question, clean CCR rose from 78.8% to 92.9%, close to the
    target, and published ASR rose from 71 to 82. Across cutoffs the curve moved towards
    higher CCR at similar or higher ASR. The layer that never ran works as a recall
    layer, not as a security layer.
26. **The evaluation report (`docs/EVALUATION_REPORT.md`) repeats the pattern on
    HotpotQA.** ASR out of 100 for the Guard, clean ASR 13: published 32, with question
    13, diverse 45, capped 12, benign 49. TrustRAG Stage 1 had 12, 7, 28, 9 and 14, often by
    returning no context. Held-out false alarms per chunk were 3.8% to 5.2% on all three
    corpora, and clean CCR 78.8% to 85.4%. Clustering alone removed almost no poison anywhere.
27. **Signal A's power is a property of the corpus (E42).** Poison sits a median 0.174 above
    the best honest chunk on HotpotQA and 0.045 on MS-MARCO, which explains published ASR
    of 32 against 71. A single Guard ASR does not transfer between corpora.

**Limits.** Findings 1 to 16 rest on one query, 4–9 chunks per pool and a 40-value reference. This is an
existence proof that the signal can work, not an estimate of detection or
false-positive rates.

## 13. Immediate next step

Stages 3 and 4 are built, and Stage 5's calibration and detector test are done. The owner
kept the targets and the calibrated point (D44). Stages 6 and 7 are complete (findings 23
to 25). What remains is packaging and the write-up.

*End of v0.22. We'll version this doc as decisions firm up.*
