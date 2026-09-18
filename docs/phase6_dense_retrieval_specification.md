# Phase 6 — Dense Retrieval Specification

> **Status:** LOCKED DESIGN SPECIFICATION for implementation  
> **Project:** PROJECT 2 — Vietnamese Legal Document Intelligence / Vietnamese Legal Retrieval System  
> **Scope:** Phase 6 — Baseline 2: Dense Retrieval + BM25 comparison  
> **Implementation mode:** Pretrained zero-shot dense retrieval; no fine-tuning

---

## 1. Objective

### 1.1 Primary objective

Implement and evaluate a **dense bi-encoder retrieval baseline** for the existing Vietnamese legal article-level retrieval task.

The baseline must:

1. use the existing 59,636-article corpus;
2. preserve `global_article_id` exactly;
3. produce one dense vector per article;
4. encode each query with the same pretrained embedding model;
5. retrieve articles using exact similarity search;
6. reuse the existing retrieval result contract and evaluator;
7. compare Dense Retrieval against the official Phase 5 BM25 baseline under the same benchmark conditions;
8. explicitly analyze whether Dense Retrieval rescues queries for which BM25 has no relevant article in Top-10.

### 1.2 Research question

> **Does pretrained dense retrieval improve retrieval for queries where BM25 suffers from lexical mismatch or paraphrase?**

This is an experimental question. The specification must not assume the answer.

### 1.3 Classification

- **FACT:** Phase 6 is a zero-shot/pretrained dense retrieval phase; no model fine-tuning is in scope.
- **FACT:** The retrieval unit remains the article.
- **DECISION:** The official Phase 6 baseline is dense-only, one vector per article, exact search.
- **EXPERIMENTAL:** Whether Dense Retrieval improves over BM25 is determined only by the benchmark results.

---

## 2. Scope

### 2.1 Included

- pretrained multilingual dense embedding model;
- article encoding;
- query encoding;
- L2 normalization;
- exact inner-product search;
- persistent document embeddings;
- persistent exact vector index;
- dense `BaseRetriever` implementation;
- evaluation with the existing retrieval evaluator;
- BM25-vs-Dense comparison;
- latency/QPS measurement;
- BM25 hard-failure analysis;
- reproducibility metadata;
- unit/integration/validation tests.

### 2.2 Explicitly excluded

The implementation MUST NOT include:

- embedding-model fine-tuning;
- contrastive training;
- hard-negative mining;
- hybrid retrieval;
- BM25 + Dense score fusion;
- reranking/cross-encoder;
- ColBERT/multi-vector retrieval;
- BGE-M3 sparse/lexical retrieval output;
- chunking;
- RAG;
- LLM answer generation;
- query rewriting;
- production API;
- web application;
- replacement or redesign of Phase 5.

The roadmap explicitly places chunking after Dense Retrieval, and hybrid/reranking after that; therefore these mechanisms remain outside Phase 6.

- **FACT:** The project roadmap separates Dense Retrieval, chunking, vector-database work, RAG, hybrid retrieval, and reranking into later phases. [PROJECT2 roadmap]
- **DECISION:** No later-phase mechanism may be pulled forward merely because the embedding library supports it.

---

## 3. Existing Project Constraints

### 3.1 Repository architecture

The repository uses a **thin reusable library + phase scripts** architecture rather than a packaged service.

Relevant documented structure:

```text
src/
├── retrieval/
└── evaluation/

scripts/
├── retrieval/
└── evaluation/

tests/

data/
├── processed/
├── benchmarks/
└── indices/

results/
└── benchmarks/
```

- **FACT:** `src/retrieval/base.py` defines `RetrievalResult(article_id: int, score: float, rank: int)` with **1-indexed rank** and a `BaseRetriever` contract containing `fit`, `search`, `batch_search`, `save`, and `load`.
- **FACT:** Existing evaluation consumes article IDs in rank order and uses the established retrieval metric API.
- **FACT:** New retrievers are required to subclass `BaseRetriever`.
- **FACT:** Existing project guidance says not to refactor or redesign the library/script split.
- **SOURCE STATUS:** The current library audit available to this specification verified the contract through `AGENTS.md`, but the exact source text of `src/retrieval/base.py`, `bm25_retriever.py`, `tfidf_retriever.py`, `retrieval_metrics.py`, and `tests/test_retrieval_metrics.py` was not independently surfaced in the file index during this design pass.
- **UNKNOWN / NOT VERIFIED:** Exact current Python signatures and internal implementation details of those source modules must be inspected by the coding agent before editing. The specification intentionally does not invent signatures.

### 3.2 Corpus contract

Production retrieval corpus:

```text
data/processed/legal_retrieval/corpus_articles_preprocessed.parquet
```

Documented production fields include:

```text
global_article_id
doc_id
law_id
aid
text_raw
text
char_count_preprocessed
word_count_preprocessed
is_empty_preprocessed
created_at
```

- **FACT:** There are **59,636** retrieval articles.
- **FACT:** `global_article_id` is the system-wide retrieval key and spans `[0, 59635]`.
- **FACT:** The ID is tied to flattening order and must not be compacted or reordered.
- **FACT:** Exactly 8 articles are empty/whitespace and are flagged with `is_empty_preprocessed=True`; they are not removed.
- **FACT:** Existing Phase 5 retrieval builders operate over the same article ID space.
- **DECISION:** Dense indexing must represent the same 59,636 article positions in canonical corpus order.

### 3.3 Benchmark contract

```text
data/benchmarks/retrieval_train.json
data/benchmarks/retrieval_test.json
```

Schema:

```text
query_id
query_text
relevant_article_ids
```

- **FACT:** `retrieval_train.json` has 1,752 queries.
- **FACT:** `retrieval_test.json` has 438 queries.
- **FACT:** `relevant_article_ids` are sorted unique integers in `[0, 59635]`.
- **FACT:** The split is documented as 80/20 and was generated with `random_state=42`.
- **FACT:** `retrieval_test.json` is a frozen hold-out.
- **DECISION:** No query text, qrels, split definition, or benchmark file may be modified by Phase 6.

### 3.4 Existing evaluation contract

Current metric keys:

```text
P@1
P@5
P@10
P@20
P@50
P@100
R@1
R@5
R@10
R@20
R@50
R@100
MRR@10
MAP@10
MAP@100
nDCG@10
nDCG@100
```

- **FACT:** Metrics are computed per query and macro-averaged.
- **FACT:** Relevance is binary.
- **FACT:** Existing implementation de-duplicates duplicate retrieved IDs.
- **DECISION:** Existing metric key names MUST NOT change.
- **DECISION:** No second implementation of the existing retrieval metrics is allowed.

### 3.5 Official Phase 5 reference

Documented official lexical baseline:

```text
Method: BM25 tuned
k1 = 1.80
b  = 0.85

Reference hold-out metrics:
nDCG@10   = 0.4866
R@10      = 0.6671
MRR@10    = 0.4610
R@100     = 0.8727
latency   ≈ 3.26 ms/query
```

- **FACT:** These are documented Phase 5 reference values, not values to be regenerated or altered in Phase 6.
- **DECISION:** Dense is compared against this official BM25 baseline, not against a newly tuned BM25 configuration.

### 3.6 Existing failure signal

- **FACT:** 99 of 438 frozen-test queries (22.60%) have `R@10 = 0` for both TF-IDF and BM25.
- **DECISION:** Phase 6 must explicitly test whether Dense rescues this failure set.

---

## 4. Research Findings

### 4.1 Candidate model set

Only a small number of genuinely relevant candidates are retained.

| Candidate | Evidence | Important properties | Project relevance | Phase 6 status |
|---|---|---|---|---|
| `BAAI/bge-m3` | Official BGE model card/docs + Vietnamese legal retrieval research | multilingual, 1024-d, up to 8192 tokens; dense, sparse and multi-vector capabilities | Strong fit for long Vietnamese legal articles and direct dense-only use | **DECISION: official baseline** |
| `intfloat/multilingual-e5-base` | Official E5 model card/report | multilingual, 768-d; 512-token limit; retrieval requires `query:` / `passage:` prefixes | Useful lightweight comparison candidate; shorter context is a limitation for article-level legal text | **RECOMMENDATION / candidate audit** |
| `intfloat/multilingual-e5-large` | Official E5 model card/report | multilingual, 1024-d; 512-token limit; 24 layers | Higher-capacity E5 candidate but same hard context limit | **RECOMMENDATION / candidate audit if resources allow** |

Vietnamese legal retrieval evidence does **not** prove a universal winner:

A published Vietnamese legal retrieval study reports different behavior across TVPL and Legal Zalo 21. In that study, BGE-M3 and mE5-base were both usable zero-shot models, but their relative performance changed by benchmark; therefore benchmark-specific evaluation remains necessary. [Vietnamese Legal Retrieval study]

- **EVIDENCE:** BGE-M3 is multilingual, has 1024-dimensional dense vectors and supports up to 8192-token sequences. [BGE-M3 model card]
- **EVIDENCE:** BGE-M3's official dense-retrieval documentation uses normalized `[CLS]` representations and inner-product-style scoring. [BGE-M3 documentation]
- **EVIDENCE:** Multilingual-E5-base supports 100 languages but truncates long input to 512 tokens and requires `query:` / `passage:` prefixes for asymmetric retrieval. [E5-base model card]
- **EVIDENCE:** Multilingual-E5-large has 1024 dimensions, 24 layers, supports multilingual inputs, uses `query:` / `passage:` prefixes, and uses a 512-token maximum in the official example/model card. [E5-large model card]
- **EVIDENCE:** Vietnamese legal retrieval research has evaluated BGE-M3 and mE5 on Vietnamese legal benchmarks, showing that generic multilingual embeddings can behave differently across legal datasets. [arXiv:2412.00657]
- **UNKNOWN / NOT VERIFIED:** Exact benchmark contamination/provenance for every possible Vietnamese-specific checkpoint was not established; such checkpoints are not part of the official Phase 6 baseline.

### 4.2 Final model decision

**DECISION**

```text
Preselected baseline model: BAAI/bge-m3
This model is the initial baseline selected from the research stage. Before implementation, verify the official model documentation, required dependencies, dense-embedding interface, model revision, and compatibility with the project environment. Do not silently replace the model. If the verification reveals a blocking incompatibility, STOP and report the exact conflict.
Mode: dense-only
Representation: native dense CLS representation
Dimension: 1024
```

Rationale:

1. It is explicitly multilingual.
2. It supports substantially longer input than the E5 candidates.
3. Official documentation directly specifies dense CLS retrieval.
4. It has direct evidence of use in Vietnamese legal retrieval research.
5. Long context matters because Phase 6 intentionally uses **one vector per full article** and does not yet use chunking.
6. The corpus size is small enough that the extra vector dimension is operationally manageable.

This decision does **not** assert that BGE-M3 is globally or permanently superior. It establishes a defensible Phase 6 baseline.

### 4.3 Pooling

**DECISION:**

```text
pooling = native BGE-M3 CLS dense representation
```

The implementation should use the official BGE-M3 pathway rather than invent a custom mean-pooling head.

- **EVIDENCE:** Official BGE documentation defines the dense representation as the normalized hidden state of the special `[CLS]` token.
- **DECISION:** Mean pooling must not be substituted into the official baseline.

### 4.4 Query/document encoding behavior

**DECISION:**

- Use the same BGE-M3 model revision for both queries and articles.
- Use the model's dedicated dense encoding path.
- Do not add E5-style `query:`/`passage:` prefixes to BGE-M3.
- Do not add an ad-hoc query instruction unless the pinned model documentation explicitly requires it.
- Use the model's official corpus/passage encoding path for documents where available.

### 4.5 Normalization and similarity

**DECISION:**

```text
L2-normalize query vectors
L2-normalize article vectors
FAISS metric = inner product
```

For normalized vectors:

\[
\hat q = \frac{q}{\|q\|_2},
\qquad
\hat d = \frac{d}{\|d\|_2}
\]

then:

\[
\hat q^\top \hat d = \cos(q,d)
\]

Therefore normalized inner product produces the same ordering as cosine similarity.

- **EVIDENCE:** FAISS documents `IndexFlatIP` as exact inner-product search and explicitly notes that it can be used for cosine search after normalization.
- **DECISION:** Inner product is the implementation score; its interpretation is cosine similarity because vectors are normalized.
- **NOT SELECTED:** Raw L2 distance is not used because it provides no advantage for the normalized dense baseline.

### 4.6 Exact search vs ANN

Corpus size:

```text
N = 59,636
D = 1,024
```

FP32 dense storage:

```text
59,636 × 1,024 × 4 bytes
≈ 244,269,056 bytes
≈ 233 MiB
```

Exact-search computational work per query is on the order of:

```text
N × D ≈ 61 million multiply-accumulate-scale operations
```

This is practical for a baseline, especially because the evaluation workload is only 438 frozen-test queries and document embeddings are precomputed.

**DECISION:**

```text
Index: faiss.IndexFlatIP
Search: exact
ANN: not required for Phase 6
```

Rationale:

- exact ranking;
- no recall-vs-speed approximation trade-off;
- no `nprobe`, `efSearch`, cluster count, or graph hyperparameters;
- easier debugging;
- simpler reproducibility;
- approximately 233 MiB for FP32 vectors;
- corpus is only ~59.6K vectors.

- **EVIDENCE:** FAISS identifies `IndexFlatIP` as exhaustive exact inner-product search and as suitable for cosine retrieval after normalization.
- **EVIDENCE:** FAISS's current index-selection guidance says a `Flat` index is appropriate when exact results are desired; direct computation can also be efficient for relatively small workloads.
- **DECISION:** HNSW/IVF/PQ are explicitly deferred.

### 4.7 Precision

**DECISION:**

```text
Official stored embeddings: float32
Official FAISS vectors: float32
```

Inference precision:

```text
Preferred official baseline: FP32
Optional implementation optimization: FP16 only as a separately documented run
```

Reason:

- FP16 can improve speed but the BGE documentation notes slight performance degradation.
- Phase 6's primary purpose is to establish a quality baseline, not optimize serving throughput.
- Persisting FP32 makes artifacts stable and avoids unnecessary score drift from storage compression.

- **EVIDENCE:** Official FlagEmbedding documentation states that FP16 speeds computation with a slight performance degradation.
- **DECISION:** Do not silently mix FP16 and FP32 results under the same experimental run ID.

---

## 5. Initial Design Audit

| Initial decision | Status | Audit | Final decision |
|---|---|---|---|
| BGE-M3 candidate | **VERIFIED** | Strong multilingual + long-context + direct Vietnamese legal evidence | Keep as official baseline |
| pretrained bi-encoder | **VERIFIED** | Fits Phase 6 scope | Keep |
| normalized embeddings | **VERIFIED** | Correct for cosine-equivalent inner product | Keep |
| cosine / inner product | **VERIFIED** | Equivalent after L2 normalization | Use normalized inner product |
| FAISS | **VERIFIED, subject to dependency audit** | Appropriate implementation for exact vector search | Keep |
| `IndexFlatIP` | **VERIFIED** | Exact, simple, reproducible for 59.6K vectors | Keep |
| article max length = 2048 | **NOT YET VERIFIED** | Original value was an unsupported fixed assumption | Determine from token-length audit before lock |
| query max length = 512 | **PARTIALLY VERIFIED** | Supported by BGE API default, but truncation must be measured on project queries | Lock through train-side audit |
| batch size = 8/32 | **IMPLEMENTATION DEFAULT** | Not a quality hyperparameter; may be resource constrained | Start with safe defaults and record actual values |
| top-k = 100 | **VERIFIED** | Matches existing evaluation depth and Phase 5 conventions | Keep Top-100 |
| FP16 | **NEEDS DESIGN CORRECTION** | Faster but can slightly change quality/numerics | Official baseline stores/uses FP32; FP16 is separate |
| precomputed embeddings | **VERIFIED** | Necessary to avoid re-encoding 59.6K articles per query | Keep |
| artifact structure | **RECOMMENDATION** | Useful but must follow repository conventions | Add minimal dense-specific artifacts |
| train/dev protocol | **NEEDS DESIGN CORRECTION** | A model-selection split is unnecessary if BGE-M3 is locked as the baseline | Use retrieval_train for non-test validation; optional predeclared candidate check only |
| frozen test | **VERIFIED** | Required by project protocol | Keep strictly frozen |
| hard-failure analysis | **VERIFIED** | Directly answers Phase 6 research question | Keep and formalize |

---

## 6. Final Design Decisions

### 6.1 Locked baseline

```text
model = BAAI/bge-m3
mode = dense-only
pooling = native CLS
embedding_dimension = 1024
normalize_embeddings = true
similarity = inner_product_on_normalized_vectors
index = faiss.IndexFlatIP
Retrieval search depth: 100
This is selected because the existing Phase 5 evaluator reports retrieval quality at cutoffs 1, 5, 10, 20, 50, 100. The dense retriever should therefore retrieve the top 100 candidates so that all existing evaluation cutoffs can be computed from the same ranking. This is not a tunable parameter for the frozen test.
stored_embedding_dtype = float32
retrieval_unit = article
corpus_size = 59,636
```

### 6.2 Configuration classification

Every important configuration must be classified.

#### LOCKED BY PROJECT REQUIREMENT

- retrieval unit = article;
- corpus = existing 59,636 articles;
- key = `global_article_id`;
- frozen test = existing `retrieval_test.json`;
- existing metric names;
- no fine-tuning;
- no chunking/hybrid/reranking/RAG.

#### LOCKED BY EXTERNAL EVIDENCE + DESIGN DECISION

- model = `BAAI/bge-m3`;
- dense-only output;
- native CLS representation;
- dimension = 1024;
- normalized inner product;
- exact `IndexFlatIP`.

#### DATA-/EVIDENCE-BASED AND LOCKED BEFORE FROZEN TEST

- article `max_length`;
- query `max_length`.

#### IMPLEMENTATION DEFAULT

- initial batch sizes;
- worker/thread settings;
- cache directory;
- device auto-selection.

These implementation defaults may be changed for resource reasons without changing retrieval semantics, but the actual values used must be recorded.

#### EXPERIMENTAL

- whether an alternate candidate is tested on train;
- any paired bootstrap confidence intervals;
- observed truncation-rate thresholds if they are predeclared and then validated.

---

## 7. Dense Retrieval Algorithm

### 7.1 Document indexing

```text
Existing processed corpus
        ↓
Read canonical article order
        ↓
Extract existing article text
        ↓
Validate global_article_id = expected ID space
        ↓
BGE-M3 tokenizer
        ↓
Deterministic truncation to locked article max_length
        ↓
BGE-M3 dense CLS representation
        ↓
L2 normalization
        ↓
FP32 embedding matrix [59,636, 1,024]
        ↓
FAISS IndexFlatIP
        ↓
Persist index + embedding metadata + corpus fingerprint
```

### 7.2 Query retrieval

```text
query_text
    ↓
Existing query normalization only
    ↓
BGE-M3 query encoder
    ↓
Deterministic truncation to locked query max_length
    ↓
BGE-M3 dense CLS representation
    ↓
L2 normalization
    ↓
IndexFlatIP.search(top_k=100)
    ↓
FAISS row position
    ↓
global_article_id mapping
    ↓
score + rank
    ↓
list[RetrievalResult]
```

### 7.3 Required invariants

- FAISS row 0 must always refer to the first canonical article.
- FAISS row `i` must map to the `global_article_id` stored alongside that row.
- No sorting of articles by text, ID, title, or metadata may occur before indexing.
- No articles may be removed, including the 8 empty articles.
- Dense scoring must not be mixed with BM25 scoring in the baseline.

---

## 8. Model Specification

### 8.1 Official model

```text
Model identifier: BAAI/bge-m3
Model revision: exact immutable revision/hash recorded at build time
Representation: dense
Use the official BGE-M3 dense embedding interface for generating dense vectors. Do not manually implement an alternative pooling strategy unless the official interface is unavailable. If manual pooling becomes necessary, STOP and report the deviation before implementation.
Dimension: 1024
```

### 8.2 Model revision

The coding agent MUST resolve and record the exact model revision/commit used for the experiment.

Do not rely on a floating `main` state for reproducibility.

If the model revision cannot be resolved or differs between index build and query time:

```text
FAIL FAST
```

### 8.3 Library

The BGE-M3 implementation should use an officially supported inference route, preferably FlagEmbedding for the dedicated BGE-M3 API.

The coding agent must first inspect dependency support in the repository environment.

Required new dependencies are expected to include some combination of:

```text
torch
FlagEmbedding
faiss-cpu or an equivalent FAISS package
numpy
```

Exact package/version pins must be determined from the actual project environment and compatibility constraints, then recorded in the Phase 6 reproducibility manifest.

- **UNKNOWN / NOT VERIFIED:** The exact current production environment and installability of FAISS/FlagEmbedding on the project's target Windows environment were not validated in this design-only pass.
- **DECISION:** Do not invent a package version. Validate installation before implementation is considered complete.

---

## 9. Input / Text Processing

### 9.1 Article text source

Dense Retrieval must use the **same existing processed article text** already established for retrieval.

The agent must inspect the actual corpus schema and Phase 5 retriever input path before coding.

The frozen-test fingerprint is integrity metadata only. It must never be used as input for model selection, hyperparameter tuning, configuration selection, or error analysis before the frozen evaluation stage.

Preferred rule:

```text
article_text = existing production retrieval text field
```

Do not silently add:

- title;
- law metadata;
- chapter labels;
- issuing authority;
- citation graph;
- duplicated metadata.

Adding such fields would change the experimental input representation and make BM25-vs-Dense comparisons less controlled.

### 9.2 Query processing

Use `query_text` from the benchmark unchanged except for the normalization already guaranteed by the benchmark contract.

Do not add:

- query rewriting;
- legal keyword expansion;
- manual synonym substitution;
- LLM paraphrasing;
- metadata injection.

### 9.3 Empty articles

The 8 empty articles remain in the corpus ID space.

For their dense representation:

- the encoder must receive the exact existing empty/whitespace text if the preprocessing contract permits it;
- if the embedding library rejects an empty string, the implementation must handle the failure explicitly and preserve the ID slot rather than deleting it;
- any fallback representation must be documented and tested before frozen evaluation.

Do not invent a semantic placeholder without documenting its retrieval implications.

---

## 10. Embedding Pipeline

### 10.1 Normalization

Apply L2 normalization after dense extraction and before indexing:

\[
v' = \frac{v}{\|v\|_2}
\]

Reject a zero vector.

### 10.2 Batch encoding

Batching is a resource parameter, not a retrieval-quality parameter.

Initial implementation may use:

```text
article_batch_size = implementation default
query_batch_size = implementation default
```

The agent must:

1. attempt a documented safe batch size;
2. reduce it if OOM occurs;
3. never change max_length solely because of test results;
4. record the actual batch sizes used.

### 10.3 Determinism

Use deterministic settings where the framework supports them.

Record:

- Python version;
- PyTorch version;
- transformers/FlagEmbedding version;
- FAISS version;
- model revision;
- device;
- inference dtype;
- tokenizer configuration;
- max lengths;
- batch sizes.

Do not promise bitwise equality across different hardware.

Required reproducibility level:

> same environment + same model revision + same corpus + same config ⇒ same ranking, with only negligible floating-point differences permitted.

---

## 11. Similarity and Vector Index

### 11.1 Exact index

Official:

```python
faiss.IndexFlatIP(1024)
```

The exact implementation must not introduce:

- IVF;
- HNSW;
- PQ;
- SQ;
- ANN parameters.

### 11.2 Why not NumPy-only

A pure matrix-multiplication backend is mathematically sufficient and may be used as a **small-fixture test oracle**, but FAISS remains the official project backend.

Rationale:

- persistent index artifact;
- established vector-search implementation;
- exactness;
- efficient CPU vector operations;
- simple ID mapping;
- easier future transition to another index without changing retrieval contract.

### 11.3 Exactness test

For a small fixture:

```text
FAISS IndexFlatIP ranking
==
NumPy normalized-dot-product ranking
```

up to expected floating-point tolerance.

---

## 12. Retriever Architecture

### 12.1 Architectural rule

Add Dense Retrieval **beside** existing TF-IDF/BM25.

Do not modify their behavior.

### 12.2 Suggested minimal module layout

The exact path must follow repository inspection.

A likely structure is:

```text
src/retrieval/
    dense/
        __init__.py
        encoder.py
        index.py
        artifacts.py
        retriever.py
```

Parallel scripts:

```text
scripts/retrieval/
    build_dense_index.py
    run_dense_retrieval.py

scripts/evaluation/
    evaluate_dense.py
    analyze_dense_vs_bm25.py
```

Tests:

```text
tests/
    retrieval/
        test_dense_encoder.py
        test_dense_index.py
        test_dense_retriever.py
        test_dense_artifacts.py
```

This is a **RECOMMENDATION**, not a mandate to rename or move existing files.

### 12.3 Module responsibilities

#### `encoder`

Responsibility:

- load exact model revision;
- tokenize;
- encode articles and queries;
- apply locked max lengths;
- extract native CLS dense embeddings;
- normalize;
- validate shape/dimension;
- return contiguous FP32 arrays.

#### `index`

Responsibility:

- create `IndexFlatIP`;
- add vectors in canonical order;
- search Top-k;
- expose row-position mapping;
- save/load FAISS index;
- validate dimension and row count.

#### `artifacts`

Responsibility:

- corpus/config fingerprint;
- model revision metadata;
- atomic persistence;
- compatibility checks;
- fail-fast mismatch handling.

#### `retriever`

Responsibility:

- adapt dense encoder + index to the existing `BaseRetriever` contract;
- return `list[RetrievalResult]`;
- preserve rank and score semantics.

#### build script

Responsibility:

- load production corpus;
- perform pre-index validation;
- run token-length audit if required;
- encode once;
- persist artifacts.

#### retrieval script

Responsibility:

- load a locked index;
- encode queries;
- retrieve Top-100;
- write prediction/run output.

#### evaluation script

Responsibility:

- call the existing evaluator;
- emit the same metric keys;
- write per-query results required for analysis.

#### analysis script

Responsibility:

- compare BM25 and Dense per query;
- identify BM25 hard failures;
- classify rescue/both-fail cases;
- emit reproducible reports.

---

## 13. Configuration

The implementation must have one serializable configuration object/file.

Illustrative schema:

```yaml
name: dense_bge_m3

model:
  name: BAAI/bge-m3
  revision: "<immutable revision>"
  mode: dense_only
  pooling: cls
  embedding_dimension: 1024

text:
  article_max_length: "<locked after corpus token-length audit>"
  query_max_length: "<locked after train-query token-length audit>"
  truncation: true

embedding:
  normalize: true
  storage_dtype: float32
  inference_dtype: float32

index:
  type: IndexFlatIP
  similarity: inner_product_on_normalized_vectors
  top_k: 100

runtime:
  article_batch_size: "<record actual value>"
  query_batch_size: "<record actual value>"
  device: "<record actual value>"

benchmark:
  evaluator: existing_project_evaluator
  frozen_test: retrieval_test.json
```

### 13.1 Configuration classification in implementation

The config must clearly distinguish:

```text
locked requirement
evidence-based decision
data-driven lock
implementation default
experimental value
```

Do not encode an implementation default as if it were a scientifically tuned hyperparameter.

---

## 14. Artifact Specification

Artifacts should be generated under the repository's existing artifact conventions after inspection.

A suitable minimal dense-specific structure is:

```text
<dense-artifact-root>/
    config.json
    corpus_ids.json
    corpus_fingerprint.json
    embeddings.f32.npy
    index.faiss
    build_report.json
```

### 14.1 `config.json`

Purpose:

- complete model/inference/index configuration.

Producer:

- build step.

Consumer:

- retriever, evaluator, reproducibility audit.

### 14.2 `corpus_ids.json`

Purpose:

- ordered mapping from FAISS row position to `global_article_id`.

Invariant:

```text
len(corpus_ids) == 59636
set(corpus_ids) == {0, ..., 59635}
```

Prefer integer IDs, not strings, unless the actual contract requires otherwise.

### 14.3 `corpus_fingerprint.json`

Must include:

- ordered article ID fingerprint;
- ordered article-text fingerprint;
- corpus row count;
- model name;
- model revision;
- tokenizer/model library versions;
- article max length;
- query max length;
- pooling;
- normalization;
- dtype;
- device;
- index type;
- package versions;
- build timestamp;
- token-length statistics;
- truncation count/rate.

### 14.4 `embeddings.f32.npy`

Purpose:

- reproducible dense representation cache.

Properties:

```text
shape = [59636, 1024]
dtype = float32
row order = canonical corpus order
```

### 14.5 `index.faiss`

Purpose:

- exact vector search index.

Invariant:

```text
index.d == 1024
index.ntotal == 59636
```

### 14.6 `build_report.json`

Must record:

- start/end time;
- elapsed build time;
- articles encoded;
- failures;
- batch size;
- device;
- dtype;
- max lengths;
- token statistics;
- truncation statistics;
- embedding shape;
- index size;
- fingerprints;
- software versions.

### 14.7 Git policy

- **FACT:** `data/raw/`, `data/processed/`, and `data/indices/` are currently documented as gitignored.
- **DECISION:** Large generated vectors/indexes should not be committed by default.
- Config/spec/source code belongs in Git.
- Large generated embeddings/indexes belong outside Git unless repository policy explicitly changes.
- Exact artifact storage behavior must be aligned with the actual `.gitignore` and repository conventions before implementation.

---

## 15. Experiment Protocol

### 15.1 Core principle

Only the retrieval method changes.

```text
Same corpus
Same query IDs
Same qrels
Same evaluator
Same metric implementation
Same cutoffs

          ┌─────────────┐
          │             │
        BM25          Dense
          │             │
          └──────┬──────┘
                 ↓
          Same evaluation
```

### 15.2 Development phase

Because this Phase 6 baseline uses a fixed pretrained model rather than training a model, a new train/dev split is **not required for the final official baseline**.

Use `retrieval_train.json` for:

- schema validation;
- token-length analysis;
- smoke tests;
- implementation validation;
- optional predeclared candidate comparison.

Do not inspect frozen-test query text for tuning.

### 15.3 Optional candidate sanity experiment

If the project wants empirical model selection rather than an evidence-locked BGE-M3 baseline, define the candidate set **before any results are read**:

```text
BAAI/bge-m3
intfloat/multilingual-e5-base
intfloat/multilingual-e5-large   # only if resources allow
```

Use only `retrieval_train.json`.

Selection metric:

```text
primary   = nDCG@10
secondary = R@10
```

Any selection rule must be frozen before the run.

However:

- **DECISION:** Official Phase 6 does not require this candidate tournament.
- **DECISION:** The BGE-M3 baseline may be locked directly based on external evidence and the project's long-context/no-chunking requirement.
- This avoids turning Phase 6 into an unbounded model-hunting exercise.

### 15.4 Frozen-test lock

Before any test retrieval:

1. finalize model revision;
2. finalize text representation;
3. finalize max lengths;
4. finalize normalization;
5. finalize similarity/index;
6. finalize Top-k;
7. finalize dependency/runtime configuration;
8. compute corpus fingerprint;
9. compute benchmark fingerprints;
10. write immutable experiment manifest;
11. record code commit;
12. verify working tree state according to project policy.

Only then may `retrieval_test.json` be loaded for evaluation.

### 15.5 Frozen test

Use:

```text
data/benchmarks/retrieval_test.json
```

exactly as it exists.

Do not:

- regenerate;
- reorder;
- relabel;
- remove queries;
- add queries;
- modify qrels;
- tune on its results.

Treat repeated test evaluations with different configurations as separate experiments that consume the hold-out.

---

## 16. BM25 vs Dense Evaluation

### 16.1 Metrics

Report exactly the existing keys:

```text
P@1 P@5 P@10 P@20 P@50 P@100
R@1 R@5 R@10 R@20 R@50 R@100
MRR@10
MAP@10
MAP@100
nDCG@10
nDCG@100
```

No renaming.

### 16.2 Primary interpretation metrics

For the Phase 6 research question, emphasize:

```text
nDCG@10
R@10
MRR@10
R@100
```

These are interpretations, not replacements for the full evaluator output.

### 16.3 Required comparison table

Final experiment report should include:

| Metric | BM25 tuned | Dense BGE-M3 | Dense − BM25 |
|---|---:|---:|---:|
| P@1 | report | report | report |
| P@5 | report | report | report |
| P@10 | report | report | report |
| R@10 | report | report | report |
| R@100 | report | report | report |
| MRR@10 | report | report | report |
| MAP@10 | report | report | report |
| MAP@100 | report | report | report |
| nDCG@10 | report | report | report |
| nDCG@100 | report | report | report |

Do not infer a conclusion before the actual experiment.

### 16.4 Statistical description

After the baseline configuration is locked, paired per-query bootstrap may be used to describe uncertainty:

```text
resamples = 10,000
unit = query
seed = fixed
statistics:
  delta nDCG@10
  delta R@10
CI = 95%
```

This is descriptive analysis, not a model-selection mechanism.

---

## 17. Hard-Failure Analysis

### 17.1 Definition

Define programmatically:

```text
BM25 hard failure
=
no relevant article in BM25 Top-10
```

Equivalent:

```text
BM25 R@10 == 0
```

Do not hard-code a list of 99 query IDs.

Recompute the set from the actual frozen run.

### 17.2 Required four-way matrix

For each test query:

| BM25 | Dense | Category |
|---|---|---|
| succeeds | succeeds | both succeed |
| succeeds | fails | BM25-only success |
| fails | succeeds | **Dense rescue** |
| fails | fails | **both fail** |

### 17.3 Rescue definition

```text
Dense rescue
=
BM25 R@10 == 0
AND
Dense R@10 > 0
```

Report:

- rescue count;
- rescue rate among BM25 failures;
- first relevant rank in Dense;
- Dense R@100;
- score/rank distributions;
- example top-10 rankings.

### 17.4 Failure analysis dimensions

For BM25-failure queries, record:

- query token length;
- relevant article token length;
- relevant article truncation status;
- lexical overlap between query and relevant text;
- rank of first relevant document;
- whether Dense retrieves the relevant article;
- whether Dense retrieves semantically similar but legally different articles.

### 17.5 Failure categories

Use a controlled label set:

```text
paraphrase / synonym mismatch
scenario-language vs legal-terminology mismatch
exact entity/number/citation retrieval
long article truncation
semantic confusion among related provisions
insufficient representation / both systems fail
benchmark or ID anomaly
```

These labels are for analysis, not for modifying qrels.

### 17.6 Manual sample policy

To avoid cherry-picking:

- select up to 20 rescue cases;
- select up to 20 both-fail cases;
- choose by deterministic hash of `query_id`;
- do not select only visually impressive examples.

---

## 18. Latency / Throughput Evaluation

### 18.1 Separate offline and online measurements

Report separately:

```text
A. Model loading time
B. Corpus encoding/index-build time
C. Query encoding time
D. Index search time
E. End-to-end query latency
F. QPS
```

Do not mix index-build cost into online query latency.

### 18.2 Measurement environment

Record:

- CPU;
- GPU, if any;
- RAM;
- OS;
- Python version;
- PyTorch version;
- FlagEmbedding version;
- FAISS version.

### 18.3 Warm-up

Use a documented warm-up phase before measuring steady-state query latency.

### 18.4 Query workload

For fair comparison:

- same 438 frozen test queries;
- same hardware;
- same process conditions where practical;
- exclude first-call model-loading overhead from steady-state query latency;
- report model-load/setup cost separately.

### 18.5 Throughput

If the existing framework supports QPS, report:

```text
QPS = queries / elapsed retrieval time
```

Do not compare CPU BM25 latency to GPU Dense latency without reporting hardware conditions.

---

## 19. Testing Strategy

### 19.1 Encoder unit tests

Must cover:

1. output shape;
2. dimension = 1024;
3. FP32 output;
4. vector norm approximately 1;
5. zero-vector rejection;
6. empty input list behavior;
7. deterministic encoding under fixed environment/config;
8. query/document path consistency.

### 19.2 Index unit tests

Must cover:

1. exact dimension;
2. row count;
3. unique article IDs;
4. position-to-ID mapping;
5. Top-k output length;
6. `top_k > corpus_size`;
7. invalid `top_k`;
8. duplicate prevention;
9. score ordering;
10. persistence and loading;
11. corrupted/mismatched index rejection.

### 19.3 Exactness test

Fixture:

```text
small matrix of normalized vectors
```

Compare:

```text
FAISS IndexFlatIP
vs
NumPy q @ D.T
```

Required:

```text
same ranking
within floating-point tolerance
```

### 19.4 Retriever contract tests

The Dense retriever must satisfy the actual existing `BaseRetriever` contract discovered during implementation.

Minimum invariants:

```text
article_id: int
score: float
rank: 1-indexed int
results sorted descending by relevance score
no duplicate article IDs
```

### 19.5 Artifact tests

Validate:

- config compatibility;
- model revision compatibility;
- corpus fingerprint compatibility;
- ordered ID fingerprint;
- embedding shape/dtype;
- index dimension;
- index row count.

Mismatch policy:

```text
FAIL FAST
```

No silent reuse of an incompatible artifact.

### 19.6 Integration test

Required flow:

```text
query
 ↓
dense encoder
 ↓
IndexFlatIP
 ↓
ID mapping
 ↓
DenseRetriever
 ↓
list[RetrievalResult]
```

### 19.7 Project invariants

Tests/validation must verify:

- corpus still has 59,636 articles;
- IDs remain `[0, 59635]`;
- 8 empty articles are still present;
- benchmark IDs remain unchanged;
- `retrieval_test.json` file content/hash is unchanged;
- BM25 code behavior is unchanged.

---

## 20. Validation and Invariants

### 20.1 Corpus validation

Before build:

```text
row_count == 59636
global_article_id unique
min(global_article_id) == 0
max(global_article_id) == 59635
no missing IDs
```

### 20.2 Ordered-ID validation

The exact ordered ID sequence must be fingerprinted.

Any mismatch between:

```text
saved corpus_ids
current corpus
```

must fail.

### 20.3 Text fingerprint

Hash ordered:

```text
(global_article_id, article_text)
```

not merely a set of IDs.

Reason:

Two corpora can share the same IDs while having different text.

### 20.4 Frozen-test fingerprint

Hash:

```text
retrieval_test.json
```

before evaluation.

The run manifest must store that hash.

### 20.5 Existing BM25 validation

The Dense implementation must not rewrite:

- BM25 index;
- BM25 configuration;
- BM25 reference results;
- metric implementation.

If comparison code calls BM25, it must use the documented official BM25 baseline configuration.

---

## 21. Reproducibility

The final manifest must record at minimum:

```text
project_git_commit
model_name
model_revision
tokenizer/model revision
pooling
embedding_dimension
article_max_length
query_max_length
truncation=True
normalization=True
similarity=inner_product
index_type=IndexFlatIP
embedding_storage_dtype=float32
inference_dtype
article_batch_size
query_batch_size
device
python_version
torch_version
transformers_version
FlagEmbedding_version
faiss_version
numpy_version
corpus_row_count
ordered_corpus_fingerprint
benchmark_train_fingerprint
benchmark_test_fingerprint
code/config fingerprint
```

### 21.1 Seeds

A seed is not inherently required for deterministic inference, but any seed used for:

- data split;
- bootstrap;
- random smoke-test generation;

must be explicitly recorded.

### 21.2 Model drift

Never describe:

```text
BAAI/bge-m3
```

without also recording the exact revision used for a reported result.

### 21.3 Reproduction rule

A future run may be compared to the official run only when all quality-relevant configuration fields match.

---

## 22. Implementation Contract for Coding Agents

This section is binding for Cursor / AGY / other coding agents.

### 22.1 Mandatory workflow

```text
Inspect
→ Plan
→ Implement
→ Test
→ Validate
→ Report
```

### 22.2 Before coding

The agent MUST read:

1. this specification;
2. `AGENTS.md`;
3. the actual `src/retrieval/base.py`;
4. actual Phase 5 retriever modules;
5. actual evaluator;
6. actual benchmark files/schema;
7. actual production corpus schema;
8. current dependency files;
9. current `.gitignore`;
10. relevant Phase 5 documentation and callers.

### 22.3 No guessing

Do not guess:

- method signatures;
- file paths;
- import styles;
- CLI flags;
- corpus column names;
- serialization format;
- result JSON schema;
- dependency versions.

Use the actual repository.

### 22.4 No unrelated changes

Do NOT:

- refactor Phase 5;
- rewrite BM25;
- rewrite TF-IDF;
- rename metrics;
- redesign `BaseRetriever`;
- change corpus preprocessing;
- change benchmark construction;
- move files for aesthetic reasons;
- modify archive scripts without direct necessity;
- add a web server;
- add API infrastructure;
- add chunking;
- add hybrid retrieval;
- add reranking;
- add RAG.

### 22.5 Frozen-test protection

The implementation must not modify:

```text
data/benchmarks/retrieval_test.json
```

The agent must not use frozen-test metrics to choose:

- model;
- max length;
- batch size;
- precision;
- pooling;
- index;
- any other quality-relevant configuration.

### 22.6 Conflict rule

If repository reality conflicts with this specification:

```text
STOP
→ report exact conflict
→ identify affected file/contract
→ do not silently choose an interpretation
```

The agent must not “fix” the specification/repository mismatch by improvisation.

### 22.7 Code quality rule

Prefer:

```text
minimal new code
+
existing abstractions
+
parallel dense implementation
```

over broad refactoring.

### 22.8 Documentation rule

If implementation changes a project behavior, schema, artifact contract, or phase status, update the repository's existing owning document rather than creating arbitrary duplicate documentation.

---

## 23. Execution Order

### Step 1 — Repository inspection

Inspect actual:

```text
AGENTS.md
BaseRetriever
Phase 5 retrievers
evaluator
benchmark schema
production corpus
requirements
.gitignore
```

### Step 2 — Dependency feasibility

Verify:

```text
torch
FlagEmbedding
faiss
```

are installable/usable in the project environment.

Do not change the environment blindly.

### Step 3 — Corpus integrity gate

Validate:

```text
59,636 articles
IDs 0..59,635
8 empty articles preserved
canonical order
```

### Step 4 — Token-length audit

Use tokenizer statistics without using frozen-test labels.

### Step 5 — Lock max lengths

Create a pre-test manifest containing exact:

```text
article_max_length
query_max_length
```

### Step 6 — Implement encoder

Implement only the needed BGE-M3 dense path.

### Step 7 — Implement exact index

Build:

```text
IndexFlatIP
```

and position→`global_article_id` mapping.

### Step 8 — Build corpus artifacts

Persist:

- embeddings;
- index;
- IDs;
- config;
- fingerprints;
- build report.

### Step 9 — Run development validation

Use `retrieval_train.json`.

No test tuning.

### Step 10 — Freeze manifest

Lock:

```text
model
revision
text representation
max lengths
normalization
index
top-k
dtype
software versions
corpus/test fingerprints
code commit
```

### Step 11 — Frozen test evaluation

Run Dense once for the official hold-out report.

Run/consume the official Phase 5 BM25 reference under the same evaluation contract.

### Step 12 — Comparison report

Produce:

- all required metric keys;
- deltas;
- latency/QPS;
- zero-hit count;
- per-query run artifacts.

### Step 13 — Hard-failure analysis

Compute:

```text
BM25 fail → Dense rescue
BM25 fail → both fail
BM25 success → Dense fail
both success
```

and the required secondary analyses.

### Step 14 — Final validation

Run:

```text
tests
artifact validation
corpus validation
benchmark immutability checks
```

### Step 15 — Final report

Report:

- files changed;
- files added;
- tests passed;
- validation passed;
- exact model revision;
- exact configuration;
- benchmark results;
- failure analysis;
- limitations;
- conflicts encountered, if any.

---

## 24. Acceptance Criteria

Phase 6 is accepted only when all of the following are true.

### Architecture

- [ ] Dense retriever subclasses the existing `BaseRetriever`.
- [ ] Dense retriever returns the existing `RetrievalResult`.
- [ ] Existing Phase 5 behavior is unchanged.
- [ ] No unrelated refactor is introduced.

### Model

- [ ] `BAAI/bge-m3` is used for the official baseline.
- [ ] Exact model revision is recorded.
- [ ] Dense-only output is used.
- [ ] Native CLS representation is used.
- [ ] Dimension is validated as 1024.

### Text length

- [ ] Token-length statistics are computed before frozen-test evaluation.
- [ ] Article max length is selected using a documented deterministic rule.
- [ ] Query max length is selected using a documented deterministic rule.
- [ ] Frozen-test queries are not used to make that decision.
- [ ] Truncation count/rate is reported.

### Similarity/index

- [ ] Embeddings are L2-normalized.
- [ ] Index uses exact inner product.
- [ ] Official index is `IndexFlatIP`.
- [ ] FAISS row-to-`global_article_id` mapping is preserved.
- [ ] Exactness matches a NumPy fixture.

### Corpus/benchmark

- [ ] 59,636 article rows remain.
- [ ] IDs remain exactly `[0, 59635]`.
- [ ] 8 empty articles remain.
- [ ] `retrieval_test.json` is unchanged.
- [ ] Relevant IDs are not transformed.

### Evaluation

- [ ] Same evaluator as BM25.
- [ ] Same metric keys.
- [ ] Same cutoff policy.
- [ ] Same qrels.
- [ ] Same corpus.
- [ ] Same query IDs.
- [ ] Latency and QPS are reported with hardware conditions.

### Hard-failure analysis

- [ ] BM25 hard failures are computed programmatically.
- [ ] Dense rescue count/rate is reported.
- [ ] Both-fail cases are reported.
- [ ] The four-way outcome matrix is reported.
- [ ] Deterministic manual sampling is used.
- [ ] No qrels are changed based on error analysis.

### Reproducibility

- [ ] Model revision recorded.
- [ ] Corpus fingerprint recorded.
- [ ] Test fingerprint recorded.
- [ ] Config recorded.
- [ ] Software versions recorded.
- [ ] Device/dtype recorded.
- [ ] Artifact compatibility checks work.
- [ ] Mismatch fails fast.

### Scope

- [ ] No fine-tuning.
- [ ] No chunking.
- [ ] No hybrid retrieval.
- [ ] No reranking.
- [ ] No RAG.
- [ ] No production API/UI.

---

## 25. Known Limitations

### 25.1 One vector per article

Phase 6 represents each article with one fixed-size vector.

A relevant provision near the end of a long article can be diluted or lost when truncation occurs.

This is a known limitation, not an implementation bug.

The project's roadmap explicitly assigns chunking to the following phase.

### 25.2 Generic multilingual model

BGE-M3 is not a Vietnamese-legal-specific fine-tuned model in this baseline.

Therefore:

- domain terminology may remain difficult;
- legally similar provisions may be confused;
- exact identifiers/numbers can still favor BM25;
- performance is benchmark-dependent.

### 25.3 Evidence does not establish universal superiority

Existing Vietnamese legal retrieval research shows that different multilingual models can behave differently across legal datasets.

Therefore the project must report the actual Phase 6 result rather than claiming that BGE-M3 is inherently superior.

### 25.4 Benchmark limitations

The benchmark contains 438 frozen test queries.

A result on this benchmark should be interpreted as evidence about this project benchmark, not all Vietnamese legal search.

### 25.5 Hardware dependence

Dense encoding latency is strongly hardware-dependent.

Therefore latency comparisons must record hardware and inference configuration.

### 25.6 Model/license/provenance caveat

The official BGE-M3 model is available under the repository/model-card licensing terms, but any Vietnamese-specific fine-tuned candidate would require a separate provenance and license audit before adoption.

### 25.7 Exact search scalability

`IndexFlatIP` is appropriate for the current 59.6K-article baseline but is not a claim that the same index is optimal for a much larger future corpus.

ANN can be reconsidered in a future scaling phase.

---

## 26. Evidence / Sources

### Official BGE-M3

1. **BGE-M3 model card — Hugging Face**  
   https://huggingface.co/BAAI/bge-m3  
   Evidence used for multilingual capability, 1024-dimensional vectors, and 8192-token sequence length.

2. **BGE-M3 dense retrieval documentation**  
   https://bge-model.com/tutorial/1_Embedding/1.2.4.html  
   Evidence used for normalized `[CLS]` dense representation and similarity behavior.

3. **Official FlagEmbedding BGE-M3 implementation/reference**  
   https://github.com/FlagOpen/FlagEmbedding  
   Evidence used for supported dense encoding paths and inference configuration.

4. **BGE-M3 paper**  
   Chen et al., “BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.”  
   https://arxiv.org/abs/2402.03216

### Multilingual E5

5. **multilingual-e5-base model card — Hugging Face**  
   https://huggingface.co/intfloat/multilingual-e5-base  
   Evidence used for language coverage, 768-dimensional representation, `query:`/`passage:` prefixes, and 512-token limitation.

6. **multilingual-e5-large model card — Hugging Face**  
   https://huggingface.co/intfloat/multilingual-e5-large  
   Evidence used for 1024-dimensional representation, multilingual coverage, prefixes, and 512-token usage limit.

7. **Multilingual E5 technical report**  
   Wang et al., “Multilingual E5 Text Embeddings: A Technical Report.”  
   https://arxiv.org/abs/2402.05672

### Vietnamese legal retrieval evidence

8. **Improving Vietnamese Legal Document Retrieval using Synthetic Data**  
   https://arxiv.org/abs/2412.00657  
   Used only as benchmark/domain evidence that BGE-M3 and multilingual-E5 are relevant zero-shot candidates in Vietnamese legal retrieval. It is NOT treated as proof that BGE-M3 will win on PROJECT 2.

### FAISS

9. **FAISS index documentation**  
   https://github.com/facebookresearch/faiss/wiki/Faiss-indexes  
   Evidence used for exact `IndexFlatIP`, bytes/vector, exhaustive search, and cosine-after-normalization behavior.

10. **FAISS metric documentation**  
    https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances  
    Evidence used for the distinction between inner product and cosine similarity.

11. **FAISS index-selection guidance**  
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index  
    Evidence used to justify exact Flat indexing for exact-result requirements and relatively small search workloads.

### Project-internal evidence

12. `AGENTS.md`  
    Repository architecture, retriever contract, corpus constraints, metric names, frozen-test policy, generated-artifact policy, and coding-agent rules.

13. `PROJECT2.pdf`  
    Project roadmap establishing Phase 6 Dense Retrieval before chunking, RAG, hybrid retrieval, and reranking.

---

## Final Design Lock

The official Phase 6 implementation is:

```text
Existing 59,636 article corpus
        ↓
Existing production article text
        ↓
BAAI/bge-m3
        ↓
Native CLS dense representation
        ↓
L2 normalization
        ↓
FP32 vectors
        ↓
FAISS IndexFlatIP
        ↓
Top-100 exact retrieval
        ↓
global_article_id + score + 1-indexed rank
        ↓
Existing PROJECT 2 evaluator
        ↓
BM25 vs Dense comparison
        ↓
BM25 hard-failure rescue analysis
```

The only quality-relevant numerical configuration intentionally left open until the pre-test audit is:

```text
article_max_length
query_max_length
```

Those values MUST be selected from deterministic token-length statistics and locked in a manifest **before `retrieval_test.json` is read**.

Everything else required for the official baseline is fixed above.
