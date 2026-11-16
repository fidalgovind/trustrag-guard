# TrustRAG-Guard: A Calibrated Defence Layer Against Knowledge Poisoning in Retrieval-Augmented Generation

Fidal Govind
Postgraduate, Data Analytics and Computational Science

## 1 Introduction

### 1.1 Background

A retrieval-augmented generation (RAG) system answers a question in two steps. It fetches passages from a knowledge base, then passes the top few to a language model, which writes the answer from them. The knowledge base is the weak point. Anyone who can add text to it can plant passages that rank highly for a chosen question and state a wrong answer of their choosing. This is called knowledge poisoning.

PoisonedRAG [16] made the attack concrete. For each target question, a language model writes five passages, each built to make the generator give one chosen incorrect answer. In the black-box version, the question itself is placed in front of each passage so that it wins retrieval. The authors released their target questions and attack passages for three question-answering datasets, and we use that release throughout.

Several defences exist. TrustRAG [15] runs K-means with two clusters on the retrieved passages and removes the denser group, using a ROUGE-L overlap check so that a single malicious passage is not missed. A second stage then asks the language model for its own answer and reconciles it with the remaining passages. RobustRAG [14] answers from each passage separately and aggregates the answers, with guarantees that hold only while clean passages outnumber poisoned ones. ReliabilityRAG [10] links per-passage answers that contradict each other and keeps the largest consistent set, which also assumes a clean majority. Cheng et al. [2] remove a passage when any of three percentile tests fires: a perplexity-difference test on both tails, an upper-tail perplexity-maximum test, and an upper-tail test on the passage's similarity to the question. Their reference sample is drawn from the knowledge base itself. CoordGuard [5] scores groups of embeddings by density, relevance and size, and gates the result with a conformal p-value fitted on benign queries.

### 1.2 About the project

TrustRAG-Guard (the Guard) is a defence layer between the retriever and the generator. It needs no retraining of any model. For each question it retrieves 30 chunks and checks them in three layers.

Layer 1 works on embeddings. Its main test, signal A, flags a chunk that is more similar to the question than clean retrieval normally is. This is the same idea as the similarity test of Cheng et al. [2], so we treat it as prior work that we measure, not as a new signal. Layer 1 also runs a two-cluster K-means test. Layer 2 measures how much of each chunk's wording is shared with another chunk in the pool. The two layers are combined into one risk score per chunk, with thresholds fitted on clean questions only. Layer 3 is a small local language model that judges a borderline chunk, but only when too few chunks are left. A context floor then builds the final list without ever refilling it by similarity.

We set three targets at the start of the project and did not change them. The attack success rate (ASR), the share of questions out of 100 whose generated answer contains the attacker's answer, should be below 5%, read beside the same system's ASR on clean questions. The clean-context rate (CCR), the share of gold chunks kept out of those in the undefended top 5 on clean pools, should be above 95%. The fast path, which covers query embedding, Layer 1, Layer 2 and fusion, should stay under 150 ms at the 99th percentile over at least 500 timed rounds.

### 1.3 Scope of this report

We report the build frozen on 9 November 2026 and every evaluation run on it. Each measurement followed a rule that was written down before any data for it was scored, and no rule was relaxed after its result was known. All results are for the calibrated configuration, which `TrustRAGGuard.calibrated()` loads. A guard built directly with its default arguments uses the earlier placeholder thresholds instead, so its results would differ.

The evidence has three limits that apply to everything below. It is development-level, because all three PoisonedRAG question sets were used while the system was being built. The generator is a local 3B model used as a stand-in. And TrustRAG is our own re-implementation, not the authors' original runs. Section 5.4 gives the details. The test scripts, and an evaluation report with the file hashes, are kept in the repository.

## 2 Methodology

### 2.1 Pipeline overview

Figure 1 follows one question through the system, and Table 1 lists the components.

![Figure 1: Data flow of the frozen build for one question. Boxes name the module and the setting used in evaluation. Source: `trustrag_guard/retriever.py`, `pipeline.py`, `fast_path.py`, `fusion.py`, `context_floor.py`, `layers/verifier.py`.](fig/fig1_architecture.png)

Table 1: Components of the frozen build

| Component | Role | Main setting | Source file |
| --- | --- | --- | --- |
| Retriever wrapper | Embeds the question, retrieves chunks with stored embeddings, returns at most 5 | 30 chunks retrieved; all-MiniLM-L6-v2 | `trustrag_guard/retriever.py` |
| Signal A | Flags chunks more similar to the question than clean retrieval | Threshold fitted per corpus | `layers/over_alignment.py` |
| Clustering test | Two-cluster K-means split, smaller group flagged | Split only if silhouette reaches tau | `layers/clustering.py` |
| Coordination test | Highest ROUGE-L of each chunk with any other chunk | Bit-parallel exact scoring | `layers/text_space.py` |
| Fusion | One risk score and an action per chunk | Weights 0.85, 0.10, 0.05 | `fusion.py` |
| Verifier | Judges a borderline chunk SAFE or UNSAFE | llama3.2:3b, at most 3 calls | `layers/verifier.py` |
| Context floor | Builds the final list | At least 2 chunks; never by similarity | `context_floor.py` |
| Calibration | Fits thresholds on clean questions | Conformal quantiles | `calibration/build_thresholds.py` |

### 2.2 Layer 1: signal A and the clustering test

Signal A computes each chunk's cosine similarity to the question and compares it with a threshold fitted offline on clean retrieval. The threshold never comes from the pool being scored, because poison can make up a large share of that pool and would pull any pool statistic towards itself.

The clustering test always fits K-means with two clusters and computes the silhouette score [9], which measures how well separated the two groups are. If the score reaches a threshold called tau, the smaller group is flagged; otherwise the test reports no split. It also reports no split for fewer than four chunks. K-means runs on one thread. On 50-chunk test pools this cut Layer 1 from 32 ms to 10 ms at the median without changing any decision on 200 random pools.

### 2.3 Layer 2: the coordination test

Poison passages written for the same question tend to reuse each other's wording. Layer 2 gives each chunk one score, its highest ROUGE-L F-measure [6] with any other chunk in the pool. ROUGE-L is based on the longest common subsequence of words. Every pair is scored exactly with a bit-parallel routine compiled with Numba. On 50 chunks of 200 words its p99 was 6.3 to 9.9 ms over four runs, inside a 12.5 ms limit set for this layer. A plain implementation had failed the same limit at 48.8 ms.

### 2.4 Fusion and thresholds

Signal A's similarity is mapped to a grade that is 0 at the clean median and 1 at the signal A threshold, and is clipped to that range. The risk is 0.85 times the grade, plus 0.10 times the coordination score, plus 0.05 times the clustering flag, so it lies between 0 and 1. A chunk below the lower threshold `low` is kept (KEEP). A chunk from `low` up to the upper threshold `high` is AMBIGUOUS, and a chunk at or above `high` is dropped (DROP). The code refuses weights that would let clustering or coordination reach `high` without signal A.

With the calibrated values, `high` is 0.867 to 0.869. Signal A alone can contribute at most 0.85, so it cannot drop a chunk by itself. A drop needs signal A at its maximum grade plus a coordination score of at least 0.17 to 0.19, or the clustering flag. Clustering and coordination together reach at most 0.15, so they cannot drop a chunk without signal A.

### 2.5 Layer 3 verifier and the context floor

The verifier sends the question and one chunk to llama3.2:3b through Ollama, at temperature 0 with a fixed seed, and reads only the first token of the reply. The token `SAFE` means SAFE, a token starting with `UN` means UNSAFE, and anything else is treated as unsafe. The prompt was written once and not changed after results were seen. The verifier runs only when fusion kept fewer than two chunks. It then checks AMBIGUOUS chunks in order of lowest risk, at most three per question.

The context floor starts with every KEEP chunk. If fewer than two remain, chunks the verifier called SAFE fill the gap, and no more than the gap, taking the least represented source first. The floor never uses similarity, because PoisonedRAG poison is written to be the most similar text in the pool. DROP chunks and unchecked AMBIGUOUS chunks are never returned.

### 2.6 Conformal calibration

Calibration fits five values per corpus from clean reference questions. The clean median is the ordinary median of clean top-5 similarities. The other four are conformal quantiles [1, 12]: each is the value at rank ceil((n + 1)(1 - alpha)) of the sorted clean scores. Alpha is 0.01 for the signal A threshold, tau and `high`, and 0.05 for `low`. On new clean data, about 5% of top-5 chunks should then reach the AMBIGUOUS band or above, and about 1% should be dropped. Chunks in one pool are not independent, so this is only approximate, and Section 4.2 checks it on held-out questions. The fusion weights are set by hand.

Table 2: Values fitted on clean reference questions and used in every evaluation (similarities are cosine values; `low` and `high` are on the 0 to 1 risk scale)

| Corpus | Signal A threshold | Clean median | tau | low | high |
| --- | --- | --- | --- | --- | --- |
| NQ | 0.831 | 0.518 | 0.161 | 0.645 | 0.869 |
| HotpotQA | 0.855 | 0.489 | 0.153 | 0.558 | 0.868 |
| MS-MARCO | 0.886 | 0.535 | 0.193 | 0.736 | 0.867 |

### 2.7 Baselines and metrics

We compare four systems: no defence (the top 5 retrieved chunks), TrustRAG Stage 1 (the clustering filter alone), TrustRAG full (the filter plus its answer-reconciliation stage), and the Guard. Our TrustRAG port in `tests/baseline_trustrag.py` keeps the public code's branches and constants (ROUGE-L gate 0.25, cosine limit 0.88, K-means random state 0) and its SimCSE embedder [3]. Its prompts are copied word for word. It runs on the top 5 of our MiniLM retrieval rather than the original retriever, with our local generator. The paper says its experiments remove the question from the poison, while its public attack code puts the question in front of each passage. We therefore report both poison forms.

Every system uses the same generator: llama3.2:3b through Ollama at temperature 0 and seed 42, with PoisonedRAG's prompt and up to 128 new tokens. ASR counts the questions, out of 100, whose answer contains the attacker's answer as a substring, as PoisonedRAG does. This check also fires on questions with no poison present, so every ASR is reported beside that system's clean ASR. CCR is the share of gold chunks kept, out of those in the undefended top 5. Where the number of gold chunks is below 20, we give the count instead of a percentage.

Latency is reported as percentiles: p50 is the median and p99 the value that 99% of rounds stay under. We quote p99 only from at least 500 samples and p95 only from at least 100. AUROC is the probability that a detector scores a random poison chunk above a random clean chunk; 0.5 is chance. Intervals are 95% bootstrap intervals from 2,000 draws that resample questions. Differences between two systems are paired: each draw takes both systems' results on the same questions.

## 3 Details of Data

### 3.1 Datasets and attack sets

The attack data is PoisonedRAG's public release for NQ, HotpotQA and MS-MARCO. Each set has 100 target questions, and each question has a correct answer, an incorrect answer chosen by the attacker, and five poison passages. The knowledge base for each dataset is a subset of about 50,000 passages from its BEIR version [11]. It contains the gold passages of all target and reference questions, plus random passages chosen with seed 42. Not every measurement exists for every dataset (Table 3).

Table 3: What was measured on each dataset

| Measurement | NQ | HotpotQA | MS-MARCO |
| --- | --- | --- | --- |
| Detection, published and with-question poison | Yes | Yes | Yes |
| Detection, adaptive poison | Yes | Yes | Yes |
| Generated answers (ASR) | No | Yes | Yes |
| Held-out false alarms | Yes | Yes | Yes |
| Fitted-detector separation test | No | Yes | No |

### 3.2 One sample record

Table 4 shows one HotpotQA target question with its passages, shortened. The poison passage was retrieved at rank 3 with similarity 0.652 to the question. The gold passage that answers the question correctly was at rank 7 (similarity 0.511).

Table 4: A sample record (HotpotQA, question id 5ae4a3265542995ad6573de5)

| Field | Content |
| --- | --- |
| Question | Hayden is a singer-songwriter from Canada, but where does Buck-Tick hail from? |
| Correct answer | Japan |
| Attacker's answer | Sweden |
| Gold passage (rank 7) | "Buck-Tick (stylized as BUCK-TICK) is a Japanese rock band, formed in Fujioka, Gunma in 1983. ..." |
| Poison passage (rank 3) | "Known for their musical diversity, Buck-Tick is a rock band from Sweden. Despite common belief, they are not from Japan like many presume, ..." |
| Clean retrieved passage (rank 8) | "Leahy is a Canadian folk rock band." |

### 3.3 Data schema

Table 5: Data fields used by the evaluation

| Source | Field | Content |
| --- | --- | --- |
| Question file | `id`, `question` | Question identifier and text |
| Question file | `gold` | Identifiers of the gold passages, from the BEIR relevance labels |
| Question file | `correct`, `incorrect` | Correct answer and attacker's answer |
| Question file | `adv_texts` | The five PoisonedRAG poison passages |
| Vector store | id, document, embedding | One passage, its text and its 384-dimension embedding |
| Vector store | `source` | Passage title, used by the context floor to spread chunks across sources |
| Pool (built per question) | 30 chunks | Top 30 by similarity after the poison is inserted at its own rank |

### 3.4 Calibration and held-out splits

Each corpus also has 200 reference questions that are not targets. They were split once, with seed 42, into 100 for fitting the values in Table 2 and 100 held out to check the promised false-alarm rates. No target question was used for fitting. A pool for a target question is built by retrieving the top 30 clean chunks, inserting the five poison passages at their own similarity rank, and keeping the top 30, so every pool has exactly 30 chunks. Besides the clean pool, each question has a published pool (poison as released), a with-question pool (the question and a full stop in front of each poison passage), and three adaptive pools (Section 4.5).

### 3.5 Test environment

Table 6: Environment used for the results

| Item | Value |
| --- | --- |
| Operating system | Microsoft Windows 11 Home Single Language, version 10.0.26300 |
| Processor and memory | AMD Ryzen 7 7735HS (8 cores, 16 threads); 13.3 GB reported by Windows |
| GPU | AMD Radeon 680M (integrated), used by Ollama for the verifier and generator |
| Python | 3.12.3 |
| Main libraries | torch 2.14.1, transformers 5.19.0, sentence-transformers 6.1.0, scikit-learn 1.9.1, numpy 2.5.3, numba 0.68.1, chromadb 1.5.9, langchain-core 1.7.6, rouge_score 0.1.2 |
| Embedding models | all-MiniLM-L6-v2 for the Guard [8, 13]; sup-simcse-bert-base-uncased for the TrustRAG port [3] |
| Language model | llama3.2:3b through Ollama 0.40.14 |
| Build | Commit f050758 plus the uncommitted working tree, with SHA-256 hashes recorded |

## 4 Results

### 4.1 Automated test results

The test suite is a set of standalone scripts that print one line per passed check. Table 7 lists the runs that were made on the frozen build.

Table 7: Test runs on the frozen build

| Date | Scope | Result |
| --- | --- | --- |
| 10 November 2026 | All 15 unit-test scripts | All 15 finished; 111 checks passed, none skipped |
| 10 November 2026 | Rerun after a later module was added: 9 of 16 scripts | All 9 finished; 60 checks passed |
| 15 November 2026 | Remaining 7 scripts | Stopped before completing |
| 16 November 2026 | All 16 unit-test scripts | All 16 finished; 116 checks passed, every output line identical to the 10 November runs |

### 4.2 Calibration results

Table 8: False alarms on 100 held-out clean questions per corpus (500 top-5 chunks)

| Corpus | Chunks not kept (promise about 5%) | Chunks dropped (promise about 1%) | Questions with a chunk not kept | Questions with a chunk dropped |
| --- | --- | --- | --- | --- |
| NQ | 26 of 500 (5.2%) | 4 (0.8%) | 24 | 4 |
| HotpotQA | 19 of 500 (3.8%) | 1 (0.2%) | 19 | 1 |
| MS-MARCO | 20 of 500 (4.0%) | 8 (1.6%) | 18 | 8 |

The share of chunks not kept was 0.2 points above 5% on NQ, 1.2 points below on HotpotQA and 1.0 point below on MS-MARCO. The share dropped stayed within 0.8 points of 1%. Per question the picture is weaker: 18 to 24 of 100 clean questions had at least one top-5 chunk moved out of KEEP.

### 4.3 Detection results

Two fitted outlier detectors were tested on HotpotQA before they could join fusion: an Isolation Forest [7] with 100 trees, and a Mahalanobis distance computed after reducing the embeddings to 40 dimensions with PCA, using a Ledoit-Wolf covariance estimate [4]. Both were fitted on 20,000 clean passages. The pass mark, fixed beforehand, was an AUROC of at least 0.80, catching at least half of the poison while flagging at most 10% of gold chunks.

Table 9: Separating 500 published poison chunks from 87 rank-1 gold chunks on HotpotQA (cutoff set on fitting-question gold at alpha 0.05)

| Score | AUROC (95% interval) | Poison caught | Gold flagged | Passes |
| --- | --- | --- | --- | --- |
| Isolation Forest | 0.439 (0.373 to 0.504) | 4.0% | 8.0% | No |
| Mahalanobis | 0.459 (0.407 to 0.512) | 3.0% | 5.7% | No |
| Similarity to the question (baseline) | 0.899 (0.858 to 0.936) | 30.4% | 2.3% | Baseline |
| Fused risk (baseline) | 0.908 (0.870 to 0.943) | 34.8% | 2.3% | Baseline |
| Word count (baseline) | 0.894 (0.829 to 0.949) | 0.0% | 0.0% | Baseline |

Table 10 shows how much published and with-question poison reached the final top 5. It also shows each Layer 1 and Layer 2 signal acting alone on published poison, dropping only what it flags.

Table 10: Poison chunks in the final top 5, out of 500 slots per dataset

| System | NQ published | HotpotQA published | MS-MARCO published | NQ with question | HotpotQA with question | MS-MARCO with question |
| --- | --- | --- | --- | --- | --- | --- |
| No defence | 429 | 486 | 435 | 489 | 497 | 484 |
| Signal A only | 367 | 334 | 396 | 189 | 177 | 298 |
| Clustering only | 429 | 481 | 435 | 479 | 487 | 484 |
| Layer 2 only (cutoff 0.5) | 339 | 246 | 388 | 127 | 22 | 296 |
| Guard | 234 | 60 | 263 | 15 | 0 | 125 |
| TrustRAG Stage 1 | 34 | 11 | 49 | 5 | 0 | 5 |

### 4.4 ASR and CCR results

Answers were generated on HotpotQA and MS-MARCO only. Tables 11 and 12 give ASR for every pool, with the clean row giving each system's clean ASR. Figure 2 plots the same values.

Table 11: HotpotQA ASR, questions of 100, with 95% interval (question joined to each poison passage with a full stop; up to 128 new tokens)

| Pool | No defence | TrustRAG Stage 1 | TrustRAG full | Guard | Guard, verifier forced |
| --- | --- | --- | --- | --- | --- |
| clean | 12 (6 to 19) | 11 (5 to 17) | 22 (14 to 31) | 13 (6 to 20) | 12 (6 to 19) |
| published | 94 (89 to 98) | 12 (6 to 19) | 30 (22 to 38) | 32 (23 to 41) | 55 (45 to 65) |
| with question | 91 (85 to 96) | 7 (3 to 12) | 27 (19 to 36) | 13 (7 to 20) | 25 (17 to 34) |
| diverse | 89 (82 to 94) | 28 (20 to 37) | 32 (23 to 41) | 45 (35 to 55) | 64 (55 to 73) |
| capped | 77 (68 to 85) | 9 (4 to 15) | 28 (19 to 37) | 12 (6 to 19) | 34 (25 to 44) |
| benign | 73 (64 to 81) | 14 (8 to 21) | 30 (22 to 39) | 49 (39 to 59) | 65 (56 to 74) |

Table 12: MS-MARCO ASR, questions of 100, with 95% interval (same settings as Table 11)

| Pool | No defence | TrustRAG Stage 1 | TrustRAG full | Guard | Guard, verifier forced |
| --- | --- | --- | --- | --- | --- |
| clean | 10 (5 to 16) | 10 (5 to 16) | 21 (14 to 29) | 10 (5 to 16) | 10 (5 to 16) |
| published | 93 (88 to 97) | 33 (24 to 43) | 28 (20 to 37) | 71 (62 to 80) | 82 (74 to 89) |
| with question | 86 (79 to 92) | 14 (8 to 21) | 22 (14 to 30) | 34 (25 to 43) | 41 (32 to 51) |
| diverse | 83 (76 to 90) | 50 (40 to 60) | 36 (26 to 45) | 84 (77 to 91) | 85 (77 to 92) |
| capped | 82 (74 to 89) | 29 (20 to 38) | 28 (20 to 37) | 53 (44 to 63) | 62 (53 to 71) |
| benign | 74 (65 to 83) | 22 (14 to 30) | 22 (14 to 30) | 71 (62 to 80) | 76 (67 to 84) |

![Figure 2: ASR per pool and system with 95% intervals, in questions of 100; the dotted line is the 5% target. The values are those of Tables 11 and 12. Sources: `data/hotpotqa/report_matrix.json` and `data/msmarco/adaptive_results.json`.](fig/fig3_asr_systems.png)

Table 13: Paired ASR difference, Guard minus TrustRAG, in percentage points with 95% interval (positive means more attacks succeeded against the Guard)

| Pool | MS-MARCO, minus Stage 1 | MS-MARCO, minus full | HotpotQA, minus Stage 1 | HotpotQA, minus full |
| --- | --- | --- | --- | --- |
| clean | 0 (0 to 0) | -11 (-18 to -4) | 2 (0 to 5) | -9 (-16 to -2) |
| published | 38 (27 to 49) | 43 (33 to 54) | 20 (10 to 30) | 2 (-10 to 14) |
| with question | 20 (10 to 29) | 12 (3 to 22) | 6 (-2 to 13) | -14 (-22 to -6) |
| diverse | 34 (23 to 45) | 48 (36 to 59) | 17 (5 to 29) | 13 (0 to 26) |
| capped | 24 (12 to 36) | 25 (14 to 36) | 3 (-5 to 11) | -16 (-24 to -8) |
| benign | 49 (37 to 61) | 49 (38 to 60) | 35 (25 to 46) | 19 (6 to 33) |

The Guard's ASR was higher than TrustRAG Stage 1's in all ten attacked pools, and in eight of them the paired interval excludes zero. Against TrustRAG full the Guard was higher in eight of ten attacked pools. It was lower on HotpotQA with-question and capped poison, by 14 and 16 points, because TrustRAG full's reconciliation stage raises ASR even with no poison present (22 against 12 with no defence on clean HotpotQA).

Table 14 gives the context the generator received. TrustRAG removed the whole top 5 for many questions, and the generator then answered from its own knowledge. The Guard always returned a context.

Table 14: Gold chunks kept and empty contexts on the static pools (gold chunks kept of those in the undefended top 5; empty contexts in questions of 100)

| Dataset and pool | No defence | TrustRAG Stage 1 | Guard | Empty contexts, TrustRAG | Empty contexts, Guard |
| --- | --- | --- | --- | --- | --- |
| NQ clean | 100.0% (103/103) | 97.1% (100/103) | 82.5% (85/103) | 1 | 0 |
| HotpotQA clean | 100.0% (151/151) | 74.8% (113/151) | 85.4% (129/151) | 4 | 0 |
| MS-MARCO clean | 100.0% (99/99) | 86.9% (86/99) | 78.8% (78/99) | 0 | 0 |
| HotpotQA published | 14 of 14 | 10 of 14 | 8 of 14 | 87 | 0 |
| HotpotQA with question | 3 of 3 | 3 of 3 | 0 of 3 | 97 | 0 |
| MS-MARCO published | 100.0% (51/51) | 84.3% (43/51) | 60.8% (31/51) | 44 | 0 |
| MS-MARCO with question | 15 of 15 | 11 of 15 | 6 of 15 | 87 | 0 |

Figure 3 and Table 15 show the trade-off between ASR and clean CCR on HotpotQA as the risk cutoff moves. These values come from an earlier script. It joined the question to each poison passage with a space instead of a full stop and allowed 64 new tokens instead of 128. That is why its undefended with-question ASR is 82, where Table 11 shows 91. On the published pool, which involves no join, both scripts gave the same ASR (94 with no defence, 32 at the calibrated cutoff), so the join is the more likely cause. PoisonedRAG's own code uses the full stop, so Table 15 describes a variant of the published attack.

Table 15: HotpotQA trade-off as the risk cutoff varies (question joined with a space; up to 64 new tokens). ASR in questions of 100; poison in chunks of 500; CCR of 151 gold chunks

| Risk cutoff | Clean CCR | Published poison | Published ASR | With-question poison | With-question ASR | Clean ASR |
| --- | --- | --- | --- | --- | --- | --- |
| No defence | 100.0% (151) | 486 | 94 | 499 | 82 | 12 |
| 0.90 | 99.3% (150) | 392 | 84 | 137 | 36 | 12 |
| 0.80 | 97.4% (147) | 222 | 57 | 30 | 19 | 12 |
| 0.70 | 94.7% (143) | 137 | 45 | 9 | 14 | 12 |
| 0.60 | 90.1% (136) | 84 | 38 | 0 | 12 | 12 |
| 0.558 (calibrated) | 85.4% (129) | 60 | 32 | 0 | 12 | 13 |
| 0.50 | 80.1% (121) | 38 | 25 | 0 | 13 | 13 |
| 0.40 | 66.9% (101) | 20 | 18 | 0 | 10 | 11 |
| 0.30 | 49.0% (74) | 7 | 12 | 0 | 10 | 9 |

![Figure 3: ASR against clean CCR on HotpotQA as the risk cutoff varies, with the values of Table 15. ASR in questions of 100; CCR in % of 151 gold chunks. The shaded corner is where both targets would be met. Source: `data/hotpotqa/asr_curve.json`.](fig/fig2_asr_ccr.png)

No cutoff meets both targets for published poison. CCR passes 95% at 0.80, where ASR is 57, and ASR reaches the clean level only at 0.30, where CCR is 49.0%. With the question in the poison, a cutoff of 0.70 gives CCR 94.7% and ASR 14 against a clean ASR of 12.

### 4.5 Adaptive attackers

Three attackers rewrote the poison with the same budget against every system: five chunks per question, at most 60 words each, at most three language-model calls per chunk, full knowledge of every system's code and fitted values, and the incorrect answer kept. The `diverse` attacker targets overlap checks (TrustRAG's and Layer 2) by making the five chunks share pairwise ROUGE-L of at most 0.25. The `capped` attacker targets signal A by padding question-prefixed poison until its similarity sits just below the Guard's band edge. The `benign` attacker targets the verifier by rewriting poison as neutral text until the verifier answers SAFE.

Table 16: Attackers at their own goals

| Attacker and goal | NQ | HotpotQA | MS-MARCO |
| --- | --- | --- | --- |
| diverse: questions with all pairs at ROUGE-L 0.25 or below (of 100) | 10 | 4 | 25 |
| capped: chunks under the band edge (of 500) | 53 | 10 | 265 |
| benign: chunks the verifier called SAFE (of 500) | 396 | 434 | 392 |
| Language-model calls used (budget 4,500) | 1,685 | 1,709 | 1,574 |

`diverse` mostly failed its rewording goal, so it is a weak test of overlap checks. `capped` reached its goal for about half the chunks on MS-MARCO but seldom elsewhere, and `benign` succeeded for 78% to 87% of chunks. On MS-MARCO the attack aimed at TrustRAG raised TrustRAG Stage 1's ASR from 33 to 50, while the Guard stood at 84. The attack aimed at signal A raised the Guard from 34 (its with-question ASR) to 53. The attack aimed at the verifier never had to beat it, because the Guard's own rule made 0 verifier calls in every pool on both datasets. Table 17 adds the detection counts on adaptive pools, including NQ, where no answers were generated.

Table 17: Poison chunks in the final top 5 on adaptive pools (of 500)

| Pool | NQ, TrustRAG Stage 1 | NQ, Guard | HotpotQA, TrustRAG Stage 1 | HotpotQA, Guard | MS-MARCO, TrustRAG Stage 1 | MS-MARCO, Guard |
| --- | --- | --- | --- | --- | --- | --- |
| diverse | 96 | 270 | 71 | 157 | 173 | 332 |
| capped | 10 | 32 | 4 | 4 | 50 | 192 |
| benign | 28 | 196 | 23 | 163 | 47 | 289 |

### 4.6 Latency

The fast path was timed on synthetic pools of chunks of about 200 words, with 20 rounds discarded and 500 timed at each size. The timing covers query embedding, both layers and fusion; chunk embeddings were computed beforehand, as a vector store would hold them.

Table 18: Fast-path latency in ms over 500 rounds per size (run 1 on mains power; run 2 started on mains and ended on battery)

| Chunks | Run 1 p50 | Run 1 p95 | Run 1 p99 | Run 2 p50 | Run 2 p95 | Run 2 p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 17.5 | 19.2 | 21.3 | 30.6 | 34.3 | 38.5 |
| 10 | 17.9 | 19.2 | 19.6 | 31.1 | 33.9 | 40.4 |
| 30 | 21.5 | 23.4 | 24.2 | 37.6 | 40.7 | 42.0 |
| 50 | 24.7 | 27.1 | 28.7 | 43.7 | 49.7 | 54.6 |

Table 19: Verifier time per question in ms, forced to 1, 2 or 3 calls on 200-word chunks (100 questions each, on mains)

| Verifier calls | p50 | p95 | Slowest |
| --- | --- | --- | --- |
| 1 | 1,499 | 1,867 | 2,009 |
| 2 | 3,063 | 3,554 | 3,881 |
| 3 | 4,509 | 4,997 | 5,101 |

![Figure 4: Left, fast-path latency per question against pool size (p50 and p99, ms) with the 150 ms target, from Table 18. Right, verifier time per question (p50 and p95, s), from Table 19. Sources: `data/report_logs/report_speed_run1_crashed.txt`, `data/report_speed.json`, `data/report_speed_verifier.json`.](fig/fig4_latency.png)

The fast path met the 150 ms target in every measurement. In run 1, on mains power, p99 was 19.6 to 28.7 ms across pool sizes. In run 2, which ended on battery, it was 38.5 to 54.6 ms. On real MS-MARCO pools, including retrieval from the vector store, the main evaluation measured p50 29.8 ms and p99 47.1 ms over 500 rounds. TrustRAG Stage 1 took p50 483 ms and p99 888 ms there, because it runs a second embedder on the processor. A question that reaches the verifier takes 50 to 150 times longer than the fast path.

### 4.7 Failures and negative results

Table 20: What was tried, what happened, and what we concluded

| What was tried | What happened | What we concluded |
| --- | --- | --- |
| Isolation Forest and Mahalanobis detectors on HotpotQA | AUROC 0.439 and 0.459; 4.0% and 3.0% of poison caught | Published poison is not an outlier in embedding space; neither detector entered fusion |
| Out-of-corpus corroboration: does the correct answer appear in more full-corpus passages than the incorrect one? | On 250 applicable questions, the correct answer had more support on 145 (58.0%, bar 60%) and less on 54 (21.6%, bar 15%) | The test stopped at its bar; incorrect answers are often real entities that the corpus also mentions |
| Forcing the verifier onto up to three borderline chunks | Clean CCR rose from 85.4% to 97.4% on HotpotQA and from 78.8% to 92.9% on MS-MARCO; ASR rose in every attacked pool by 1 to 23 points | The verifier calls most gold chunks SAFE but also most poison; it improves context, not safety |
| The Guard against TrustRAG Stage 1 on generated answers | Guard ASR higher in all ten attacked pools, by 3 to 49 points | The Guard has no strong signal against poison without the question |

## 5 Discussion

### 5.1 Interpretation of the results

The Guard met one target and missed two (Table 21). It is fast, it never empties the context, and its per-chunk false-alarm rate on held-out questions stayed within 1.2 points of the promised 5%. Against poison that repeats the question it worked well on HotpotQA, where no such chunk reached the context and ASR matched clean ASR. It worked less well on MS-MARCO, where 125 chunks reached the context. On NQ only detection was measured: 15 with-question chunks reached the context. An earlier calibration run reported 39 with the same values; a rerun that changed only the join reproduced both counts, 39 with a space and 15 with a full stop. Against poison as published, without the question, it let through most of what reached the pool.

Table 21: Targets met or not met (ASR and clean ASR in questions of 100)

| Target | NQ | HotpotQA | MS-MARCO |
| --- | --- | --- | --- |
| ASR below 5%, published | not measured | Not met: 32 (clean 13) | Not met: 71 (clean 10) |
| ASR below 5%, with question | not measured | Not met: 13 (clean 13) | Not met: 34 (clean 10) |
| ASR below 5%, adaptive | not measured | Not met: 12 to 49 | Not met: 53 to 84 |
| CCR above 95%, clean | Not met: 82.5% | Not met: 85.4% | Not met: 78.8% |
| Fast path p99 under 150 ms | Met on all runs: 28.7 ms (run 1, synthetic), 54.6 ms (run 2, battery, synthetic), 47.1 ms (real MS-MARCO pools) | | |

### 5.2 Problem analysis

CCR stays at 78.8% to 85.4% because the calibration controls false alarms per chunk, not where they fall. Gold is usually the chunk most similar to the question, and the risk score is mostly similarity, so the clean chunks that cross the band are mostly gold. On the NQ test questions, 18 of 103 gold chunks crossed the band edge against 7 of 397 other chunks, and all 17 questions that lost their gold passage lost the one ranked first. Table 15 shows the cost of moving the cutoff: CCR passes 95% only where published-poison ASR is 57.

ASR stays above TrustRAG's because the two systems look at different things. PoisonedRAG's five passages per question repeat one another, and TrustRAG Stage 1 looks for exactly that overlap. In 71 to 100 of 100 attacked MS-MARCO pools its n-gram gate would fire. Poison without the question is no more similar to the question than clean text, so signal A does not react to it. A trace of every MS-MARCO poison chunk found most of it below the band edge (339 of 500 published chunks), and the poison that reached the context sat at a median retrieval rank of 3. Layer 2 does see the overlap: poison chunks had a median highest-overlap score of 0.25 to 0.47 against 0.19 for clean top-5 chunks. At a weight of 0.10, though, that never moves a chunk across the band. We set the weight low on purpose, because the coordination score also flags template sentences in clean text, and these results show the cost of that choice. Signal A also depends on the corpus. Published poison sat a median 0.174 above the best clean chunk on HotpotQA but only 0.045 on MS-MARCO, where clean passages often match short web queries almost word for word. As a result, 156 poison chunks crossed the signal A threshold on HotpotQA against 58 on MS-MARCO.

Clean questions show an ASR of 9 to 22 because the substring check fires without any poison. Fifteen of the 100 HotpotQA incorrect answers are "yes" or "no", which appear in many replies, and short answers can occur by chance. On HotpotQA with-question poison the Guard let no poison chunk into any of the 100 contexts, and ASR was still 13, equal to clean ASR. As ASR is measured here, even a perfect filter would not reach 5% on these sets. Measured as the excess over clean ASR, the Guard reached zero only on HotpotQA with-question and capped poison.

### 5.3 Components with no measured effect in these runs

Three parts of the design had no measured effect in these runs. Layer 3 made 0 verifier calls in every evaluated pool, because fusion always kept at least two chunks, so the floor was always met. The clustering test never split any of 300 NQ pools at the starting tau of 0.25. At the calibrated tau it flagged none of 2,457 poison chunks in the MS-MARCO pools, and acting alone it removed no published poison on NQ or MS-MARCO and 5 chunks on HotpotQA (Table 10). The fitted Isolation Forest and Mahalanobis detectors failed their pass mark and were kept out of fusion. Both the verifier and the clustering test remain in the code path: the verifier still runs when fewer than two chunks are kept, and with the calibrated `high` the clustering flag can still decide a drop.

### 5.4 Limitations

All three PoisonedRAG target sets were used during development, several of them more than once. Only the first MS-MARCO evaluation was on questions that had not been used before, so a new method would need new test data. Each setting ran once on 100 questions, which leaves most ASR intervals about plus or minus 9 points wide.

A single local model, llama3.2:3b, generated the answers, judged chunks as the verifier and wrote the adaptive attacks. A larger generator might follow poison more or less readily, and a stronger attack writer would probably reach its goals more often. On HotpotQA and NQ the adaptive attackers mostly missed their goals, so those rows test the defences weakly.

The TrustRAG results come from our port of its public code, run with MiniLM retrieval and our generator, not from the original setup. The corpora are 50,000-passage subsets, where retrieval is easier and faster than at full size. NQ has detection results only, and no ASR was measured for each signal alone. Some figures were recorded from console output at the time and have no saved result file: the separation test, the poison trace, the MS-MARCO attacker goals, the real-pool latency and the corpus comparison in Section 5.2. All timings come from one laptop whose power source changed during some runs; run 1, on mains, is the one we treat as the measurement.

Descriptions of other defences are paraphrased from the papers' arXiv versions, and no wording was checked against a final published PDF.

### 5.5 Summary

TrustRAG-Guard meets its latency target with p99 between 28.7 ms and 54.6 ms on synthetic pools and 47.1 ms on real pools, and its calibrated false-alarm rate held on held-out questions. It missed the ASR target in every measured setting and the CCR target on every dataset. Its ASR was higher than TrustRAG Stage 1's in all ten attacked pools where answers were generated, although TrustRAG reached its lower ASR partly by returning no context for 13 to 97 of 100 attacked questions. The clearest open problem is poison that does not repeat the question, which none of the Guard's active signals separates from clean text.

Repository: https://github.com/fidalgovind/trustrag-guard.git

## References

[1] Angelopoulos, A. N., and Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.

[2] Cheng, Z., Sun, J., Gao, A., Quan, Y., Liu, Z., Hu, X., and Fang, M. (2025). Secure Retrieval-Augmented Generation against Poisoning Attacks. IEEE International Conference on Big Data 2025. arXiv:2510.25025.

[3] Gao, T., Yao, X., and Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. Proceedings of EMNLP 2021.

[4] Ledoit, O., and Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. Journal of Multivariate Analysis, 88(2), 365–411.

[5] Li, Z., Guo, Q., and Xu, J. (2026). Consistency Is the Attack Surface: Detecting Coordinated Retrieval Poisoning with Embedding Geometry. arXiv:2610.24834.

[6] Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. Text Summarization Branches Out, ACL Workshop.

[7] Liu, F. T., Ting, K. M., and Zhou, Z.-H. (2008). Isolation Forest. Proceedings of the IEEE International Conference on Data Mining 2008.

[8] Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of EMNLP-IJCNLP 2019.

[9] Rousseeuw, P. J. (1987). Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics, 20, 53–65.

[10] Shen, Z., Imana, B., Wu, T., Xiang, C., Mittal, P., and Korolova, A. (2025). ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search. NeurIPS 2025. arXiv:2509.23519.

[11] Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS 2021 Datasets and Benchmarks Track.

[12] Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

[13] Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. NeurIPS 2020.

[14] Xiang, C., Wu, T., Zhong, Z., Wagner, D., Chen, D., and Mittal, P. (2024). Certifiably Robust RAG against Retrieval Corruption. arXiv:2405.15556.

[15] Zhou, H., Lee, K.-H., Zhan, Z., Chen, Y., Li, Z., Wang, Z., Haddadi, H., and Yilmaz, E. (2025). TrustRAG: Enhancing Robustness and Trustworthiness in Retrieval-Augmented Generation. arXiv:2501.00879.

[16] Zou, W., Geng, R., Wang, B., and Jia, J. (2025). PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models. USENIX Security Symposium 2025. arXiv:2402.07867.
