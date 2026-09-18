# Phase 7 --- Document Chunking Specification

**PROJECT 2 --- Vietnamese Legal Document Intelligence / Vietnamese
Legal Retrieval System**

**Status:** Specification / Implementation Contract\
**Phase:** 7 --- Document Chunking\
**Predecessor:** Phase 6 --- Baseline 2: Dense Retrieval

------------------------------------------------------------------------

## 1. Purpose

Phase 7 introduces **document chunking** as a new retrieval granularity
below the existing legal-article level.

The purpose is to determine, through controlled experiments, whether
dividing legal articles into smaller retrieval units improves retrieval
quality while preserving legal meaning, provenance, reproducibility, and
compatibility with the existing article-level benchmark.

The central research question is:

> **Which retrieval granularity and chunk-boundary strategy provides the
> best measured quality/cost trade-off for Vietnamese legal documents
> while preserving article-level provenance?**

Phase 7 must not assume in advance that:

-   256, 512, or 1024 tokens is optimal;
-   overlap is beneficial;
-   fixed-size, recursive, sentence, paragraph, or structure-aware
    chunking is optimal;
-   article-level retrieval is necessarily inferior to chunk-level
    retrieval.

Research findings supplied for this phase are treated as **evidence,
hypotheses, or open questions**, not as benchmark results for this
project.

------------------------------------------------------------------------

## 2. Scope

### 2.1 In scope

-   Corpus granularity and token-length audit
-   Inspection of Vietnamese legal structural boundaries
-   Fixed-size chunking
-   Recursive / paragraph-first splitting
-   Sentence-based chunking
-   Paragraph-based chunking
-   Structure-aware / legal-structure chunking
-   Controlled chunk-size experiments
-   Controlled overlap experiments
-   Article-level aggregation of chunk retrieval results
-   Chunk metadata, identity, and provenance
-   Deterministic chunk generation
-   Chunk corpus generation
-   Retrieval benchmark comparison against Phase 6
-   Engineering-cost measurement
-   Failure-case analysis
-   Selection and freeze of a final chunking configuration

### 2.2 Out of scope

-   Retraining or fine-tuning BGE-M3
-   Changing the Phase 6 embedding model or representation
-   Changing the canonical legal corpus
-   Changing benchmark labels or split construction
-   Tuning BM25
-   Hybrid retrieval
-   Reciprocal Rank Fusion
-   Reranking
-   RAG generation
-   Production vector-database selection
-   Distributed vector infrastructure
-   Independent redesign of the Phase 6 baseline

Advanced methods such as semantic chunking, proposition-level chunking,
contextual retrieval, late chunking, RAPTOR, and other hierarchical
methods are **not mandatory**. They require separate justification and
must not silently enter the primary benchmark.

------------------------------------------------------------------------

## 3. Locked Inputs From Previous Phases

### 3.1 Canonical corpus

The current processed retrieval corpus contains:

-   59,636 legal articles
-   immutable `global_article_id` values `0..59,635`
-   8 empty articles
-   existing article/document metadata
-   existing processed legal text

The 8 empty articles must remain represented and must not cause ID
compaction, deletion, or renumbering.

### 3.2 Retrieval benchmark

-   Total: 2,190 queries
-   Training: 1,752 queries
-   Frozen test: 438 queries

The benchmark has **article-level relevance labels**.

The frozen test set must not be used to select:

-   chunking strategy;
-   chunk size;
-   overlap;
-   aggregation;
-   search depth;
-   implementation parameters;
-   any other quality-affecting configuration.

Test is used only after the final Phase 7 configuration is frozen.

------------------------------------------------------------------------

## 4. Phase 6 Compatibility Contract

Phase 6 is completed and frozen.

``` text
Model: BAAI/bge-m3
Embedding dimension: 1024
Representation: native BGE-M3 dense representation
Normalization: L2
Data type: float32
Reference index: FAISS IndexFlatIP
Search: exact
```

Primary Phase 7 retrieval experiments must keep the Phase 6 embedding
configuration fixed.

``` text
Article
   ↓
Chunking strategy
   ↓
Chunks
   ↓
Frozen BGE-M3
   ↓
1024-d L2-normalized float32 vectors
   ↓
FAISS IndexFlatIP
   ↓
Chunk retrieval
   ↓
chunk_id → global_article_id
   ↓
Article aggregation
   ↓
Existing article-level evaluator
```

Changing the embedding model, representation, pooling, normalization, or
dimension is a separate experiment, not part of the primary Phase 7
benchmark.

------------------------------------------------------------------------

## 5. Research Findings Incorporated From the Preliminary Study

The supplied research adds several useful constraints to the original
specification.

### 5.1 Legal structure is a first-class hypothesis

Vietnamese legal text commonly contains a hierarchy such as:

``` text
Văn bản
  ↓
Phần
  ↓
Chương
  ↓
Mục
  ↓
Điều
  ↓
Khoản
  ↓
Điểm
```

Not every level exists in every article/document.

The hierarchy must be **validated against the actual corpus** before
being used as a hard chunking rule.

### 5.2 Fixed-size remains a required baseline

Literature does not establish one universal chunk size. Therefore
fixed-size 256/512/1024-token configurations remain required baselines
rather than being discarded because they are less structure-aware.

### 5.3 Recursive splitting is added as a required comparison

The research provides enough justification to include a **recursive /
paragraph-first** strategy as a low-complexity comparison.

It is treated as a boundary mechanism, not as evidence of superiority.

### 5.4 Structure-aware chunking is a high-priority hypothesis, not a predetermined winner

Legal-specific evidence motivates testing legal-structure boundaries.
However, evidence from German legal text, GDPR, judgments, or other
domains is not assumed to transfer quantitatively to this Vietnamese
corpus.

### 5.5 Sentence and paragraph chunking remain diagnostic candidates

Sentence and paragraph boundaries may preserve natural text units, but
Vietnamese legal clauses can be long and enumerative. These strategies
therefore remain in the specification, while the experiment order avoids
spending the full combinatorial budget on them before corpus profiling.

### 5.6 Advanced context methods remain separate

Parent-child retrieval, contextual prefixes, late chunking, semantic
chunking, proposition-level chunking, and similar methods change more
than chunk boundaries or add substantial complexity. They must be
evaluated separately if later justified.

------------------------------------------------------------------------

## 6. Core Design Principle

Phase 7 treats chunking as a **retrieval-granularity problem**, not
merely a token-splitting problem.

A valid chunking method must balance:

1.  local semantic specificity;
2.  legal completeness;
3.  retrieval quality;
4.  provenance;
5.  reproducibility;
6.  computational cost;
7.  storage cost;
8.  downstream context reconstruction.

Target token size is a constraint, not necessarily the semantic
boundary.

------------------------------------------------------------------------

## 7. Required Chunking Strategies

### 7.1 Fixed-size chunking

Required candidate sizes:

-   256 tokens
-   512 tokens
-   1024 tokens

Initial comparison uses zero overlap.

The implementation must explicitly define what happens when a token
boundary falls inside a sentence or legal structural unit.

### 7.2 Recursive / paragraph-first splitting

Recursive splitting must use a deterministic hierarchy of separators
appropriate to the available corpus representation.

Conceptually:

``` text
paragraph
  ↓
sentence
  ↓
token
```

The exact separator hierarchy must be documented from the actual
implementation.

It must:

-   preserve source text;
-   be deterministic;
-   define oversized-unit fallback;
-   report fallback frequency;
-   not silently introduce generated context.

### 7.3 Sentence-based chunking

Preserve sentence boundaries and group consecutive sentences toward a
target token budget.

Requirements:

-   deterministic sentence segmentation;
-   reproducible boundaries;
-   no text loss;
-   deterministic fallback for oversized sentences;
-   explicit handling of Vietnamese punctuation and segmentation edge
    cases.

Sentence-based chunking is a candidate, not a presumed final strategy.

### 7.4 Paragraph-based chunking

Preserve reliable paragraph boundaries where available.

Explicitly handle:

-   short paragraphs;
-   oversized paragraphs;
-   paragraphs containing multiple legal clauses;
-   irregular or missing paragraph boundaries.

Paragraph-based chunking is a diagnostic candidate because
formatting-based paragraphs may not consistently correspond to legal
semantic units.

### 7.5 Structure-aware / legal-structure chunking

Use reliably detected legal structure where available.

Candidate hierarchy:

``` text
Article
  ↓
Clause / Khoản
  ↓
Point / Điểm
  ↓
Sentence
  ↓
Token fallback
```

Conceptual algorithm:

``` text
Article
  ↓
Inspect legal structural units
  ↓
If article fits target:
    keep article as one chunk
Else:
    split at the highest reliable legal boundary
        ↓
    if a resulting unit is still oversized:
        use a smaller legal boundary
            ↓
        sentence fallback
            ↓
        token fallback
```

The actual hierarchy and parser rules must be based on corpus audit
results.

A structure parser failure must never silently produce a different
experimental condition. Fallback usage must be recorded.

------------------------------------------------------------------------

## 8. Corpus Granularity Audit

Before the main retrieval experiments, produce a corpus profile using
the **BGE-M3 tokenizer**.

Required article statistics:

-   article count;
-   empty article count;
-   p50/p75/p90/p95/p99/max token count;
-   fraction ≤256 tokens;
-   fraction ≤512 tokens;
-   fraction ≤1024 tokens;
-   fraction \>1024 tokens.

Required structural statistics where detectable:

-   articles containing clauses;
-   clause count distribution;
-   point count distribution;
-   structural-unit token distributions;
-   malformed or ambiguous markers;
-   fallback/unparseable frequency.

Also inspect a deterministic or predeclared random sample of
approximately 50--100 articles for manual structural feasibility
assessment.

The audit must answer:

1.  How much of the corpus actually benefits from sub-article chunking?
2.  How common are oversized legal units?
3.  How reliable are Clause/Point boundaries?
4.  Is structure-aware parsing sufficiently reliable for benchmarking?

No frozen-test relevance labels may be used in this audit.

------------------------------------------------------------------------

## 9. Experimental Matrix

Experiments must be staged to avoid uncontrolled combinatorial search.

### 9.1 Reference condition: C0

**C0 --- Article-level Phase 6 reproduction**

Purpose:

-   verify the Phase 7 evaluation pipeline reproduces the existing
    article-level retrieval contract;
-   provide the reference condition for all train/dev comparisons.

C0 is an engineering/sanity condition, not a new Phase 6 result.

### 9.2 Tier 1: Required baseline screening

  -----------------------------------------------------------------------------
  ID           Strategy               Target size          Overlap Purpose
  ------------ ----------------- ---------------- ---------------- ------------
  C1           Fixed-size                     256                0 Small
                                                                   baseline

  C2           Fixed-size                     512                0 Medium
                                                                   baseline

  C3           Fixed-size                    1024                0 Large
                                                                   baseline

  C4           Fixed-size                     512              10% Overlap
                                                                   baseline

  C5           Recursive /                    512                0 Structural
               paragraph-first                                     boundary
                                                                   baseline

  C6           Sentence-based                 512                0 Sentence
                                                                   diagnostic

  C7           Paragraph-based                512                0 Paragraph
                                                                   diagnostic
  -----------------------------------------------------------------------------

### 9.3 Tier 2: Structure-aware experiments

Run after Analysis-0 confirms that legal-structure extraction is
sufficiently reliable.

  -------------------------------------------------------------------------------
  ID           Strategy                 Target size          Overlap Purpose
  ------------ ------------------- ---------------- ---------------- ------------
  C8           Structure-aware,          Variable /                0 Legal
               Clause-first          target-bounded                  boundary
                                                                     hypothesis

  C9           Structure-aware +     Variable / 512                0 Legal
               token fallback                target                  boundary
                                                                     with
                                                                     fallback
  -------------------------------------------------------------------------------

Possible hierarchy variants may be tested only when justified by the
audit:

``` text
Article → Clause → Point → Sentence → Token
```

or:

``` text
Article → Clause → Sentence → Token
```

### 9.4 Tier 3: Conditional overlap refinement

If a strategy/size family is competitive, test:

-   0%;
-   10%;
-   20%.

Do not test every possible combination.

For 256 and 1024, additional overlap configurations may be included only
when the earlier results justify them.

### 9.5 Tier 4: Context reconstruction

Parent-child retrieval and article-title/header context injection are
**separate experiments**.

An improvement caused by adding parent context must not be attributed
solely to chunk-boundary quality.

### 9.6 Advanced methods

Semantic chunking, proposition-level chunking, contextual retrieval,
late chunking, RAPTOR, and other advanced methods are optional research
extensions and are outside the primary selection matrix.

------------------------------------------------------------------------

## 10. Experimental Variables

### Independent variables

-   chunking strategy;
-   target size;
-   overlap;
-   legal structural hierarchy;
-   deterministic fallback policy.

### Fixed variables

-   canonical corpus;
-   query text;
-   benchmark labels;
-   train/test partition;
-   BGE-M3 model and revision;
-   embedding dimension;
-   native dense representation;
-   L2 normalization;
-   float32;
-   reference FAISS IndexFlatIP;
-   evaluation implementation;
-   final article-level cutoffs.

Only the intended experimental variable may change between controlled
comparisons.

------------------------------------------------------------------------

## 11. Chunk Data Model

Every non-empty chunk must retain enough information to reconstruct its
provenance and reproduce indexing.

Minimum fields:

``` text
chunk_id
global_article_id
law_id
article_identifier
chunk_order
total_chunks
chunk_text
token_count
structure_path
content_hash
chunking_strategy
chunking_version
```

Required where deterministically available:

``` text
char_start
char_end
```

Optional structural fields:

``` text
parent_section_id
parent_clause_id
parent_point_id
structural_unit
structural_label
```

The chunk corpus must be sufficient to rebuild embeddings and indexes
without reparsing the original HTML/source documents.

------------------------------------------------------------------------

## 12. Provenance Invariants

### 12.1 Chunk → Article

Every non-empty chunk maps to exactly one `global_article_id`.

### 12.2 Article → Current Chunks

Every article has a deterministic representation of its current chunks.

### 12.3 Empty articles

The 8 empty articles remain represented at article level even if:

``` text
total_chunks = 0
```

No empty article may receive fabricated chunk text.

### 12.4 No orphan chunks

No chunk may reference a nonexistent article.

### 12.5 Unique identity

`chunk_id` must be unique within its chunking version.

### 12.6 Full text coverage

For non-empty articles, chunk generation must not silently discard
source content.

For overlap configurations, duplicated text caused by the explicitly
configured overlap is permitted and must be distinguishable from
accidental duplication.

The validator must detect:

-   missing text;
-   unexpected duplication;
-   invalid order;
-   invalid offsets;
-   impossible provenance.

------------------------------------------------------------------------

## 13. Chunk Identity and Versioning

`global_article_id` remains the immutable parent identity.

A chunk identity must not depend on an assumed fixed number of chunks
per article.

Every generated corpus must carry a `chunking_version`.

A deterministic identity may incorporate:

``` text
global_article_id
+
chunking_version
+
chunk_order
+
content_hash
```

Changing any identity-affecting configuration must create a new chunking
version.

Examples:

-   strategy;
-   target size;
-   overlap;
-   structural hierarchy;
-   fallback policy;
-   context-injection policy.

------------------------------------------------------------------------

## 14. Chunk Text Requirements

Chunk text must:

-   preserve represented legal content;
-   preserve Vietnamese Unicode;
-   preserve legally meaningful markers;
-   avoid arbitrary generated prose;
-   avoid deletion merely to satisfy a token budget;
-   be deterministic.

The primary benchmark must use the source-derived chunk text.

Contextual prefixes, generated summaries, article-title injection, or
LLM-generated descriptions are not part of the primary benchmark unless
explicitly registered as a separate experiment.

------------------------------------------------------------------------

## 15. Article-Level Evaluation

The benchmark gold labels are article-level.

Phase 7 retrieval is chunk-level:

``` text
query
  ↓
top-K chunks
```

Evaluation must therefore perform:

``` text
retrieved chunk
  ↓
global_article_id
  ↓
aggregate scores per article
  ↓
remove duplicate article identities
  ↓
rank unique articles
  ↓
evaluate using existing article-level labels
```

Raw chunk Recall@K must not be reported as a direct replacement for
Phase 6 article Recall@K.

------------------------------------------------------------------------

## 16. Article Aggregation

Aggregation is a separate methodological variable.

### 16.1 Primary aggregation baseline

Use:

``` text
article_score = max(chunk_score)
```

Rationale:

-   limits bias toward articles containing more chunks;
-   represents the presence of a highly relevant local unit;
-   simple and reproducible.

This is an experimental baseline, not a universal claim.

### 16.2 Alternative aggregation

If justified by development results, separately investigate:

-   top-n chunk scores;
-   mean;
-   sum;
-   weighted aggregation.

Aggregation experiments must be reported separately from
chunk-size/strategy experiments.

Do not simultaneously tune chunk strategy, overlap, and aggregation and
attribute the total change to chunking alone.

------------------------------------------------------------------------

## 17. Search Depth

Distinguish:

``` text
chunk retrieval depth
```

from:

``` text
final article evaluation cutoff
```

The final article cutoffs remain:

``` text
1, 5, 10, 20, 50, 100
```

A top-100 chunk search is **not** equivalent to a top-100 unique-article
result.

The implementation must:

1.  define a chunk candidate depth before final evaluation;
2.  verify on development data that the depth is sufficient for stable
    article-level evaluation;
3.  freeze that depth in the selection manifest;
4.  keep the same evaluation contract for all primary comparisons.

If the chosen chunk depth causes systematic failure to produce enough
unique article candidates, the depth must be increased during
development before the final manifest is frozen.

------------------------------------------------------------------------

## 18. Metrics

### Retrieval metrics

Maintain Phase 6-compatible metrics:

``` text
Recall@1
Recall@5
Recall@10
Recall@20
Recall@50
Recall@100

MRR@10
MAP@10
MAP@100
nDCG@10
nDCG@100
```

### Selection metrics

The main quality view should jointly consider:

``` text
nDCG@10
MRR@10
Recall@10
```

No single metric is an automatic winner.

### Engineering metrics

Record:

-   total chunk count;
-   chunks per article;
-   mean/median token count;
-   p95/p99 token count;
-   overlap duplication factor;
-   chunk-generation time;
-   embedding time;
-   embedding throughput;
-   index build time;
-   index size;
-   query latency;
-   QPS.

### Diagnostic metrics

Where useful, report:

-   correct-parent rate;
-   duplicate-parent rate in retrieved chunk candidates;
-   relevant-article first-hit distribution;
-   long-article behavior;
-   short-article behavior;
-   queries with no relevant article after aggregation;
-   per-query metric delta versus C0.

------------------------------------------------------------------------

## 19. Development / Selection Protocol

### 19.1 Data separation

The 438-query frozen test set is never used for development decisions.

The 1,752-query training benchmark is the development pool.

If an internal validation subset is needed, it must be derived from the
training pool **before quality-affecting experiments begin**, with its
split rule and seed recorded.

A reasonable protocol is:

``` text
retrieval_train.json
      ↓
predeclared internal train/dev split
      ↓
model/strategy development
      ↓
freeze configuration
      ↓
frozen test once
```

No test-derived threshold may be used for selection.

### 19.2 Engineering experiments

Engineering checks may be repeated as needed.

Examples:

-   parser correctness;
-   deterministic output;
-   artifact serialization;
-   small-corpus smoke tests;
-   index compatibility.

### 19.3 Strategy selection

Use development data only to compare:

-   strategy;
-   target size;
-   overlap;
-   structural hierarchy;
-   aggregation, if separately studied;
-   search depth;
-   implementation configuration that affects quality.

### 19.4 Important correction to the previous specification

The Phase 6 frozen-test result `nDCG@10 = 0.6277` is a **final hold-out
reference**, not a development selection threshold.

It must **not** be used as a target such as:

> "select a chunking strategy only if train nDCG@10 ≥ 0.6277"

Such a rule would use a test-derived value to influence development.

Instead:

-   compare candidates on development data;
-   analyze quality together with engineering cost;
-   document the selection rationale;
-   only then evaluate the selected configuration on the frozen test.

------------------------------------------------------------------------

## 20. Selection Policy

Selection must be based on measured development evidence.

Consider jointly:

1.  retrieval quality;
2.  stability across internal validation/development;
3.  chunk count;
4.  embedding/build cost;
5.  storage;
6.  query latency;
7.  parser reliability;
8.  legal-boundary preservation;
9.  implementation complexity;
10. downstream context suitability.

No candidate is automatically selected because it has the highest single
metric.

### Pareto analysis

When candidates are close, report the trade-off among:

``` text
retrieval quality
vs.
chunk count
vs.
embedding/build cost
vs.
index size
vs.
query latency
```

A small quality difference should not be interpreted without its
engineering consequences.

------------------------------------------------------------------------

## 21. Reproducibility Requirements

Every experiment must record:

-   strategy;
-   target size;
-   overlap;
-   structural hierarchy;
-   fallback policy;
-   tokenizer/model used for token counting;
-   chunking version;
-   corpus fingerprint;
-   benchmark fingerprint;
-   article count;
-   chunk count;
-   random seed where applicable;
-   software/runtime environment;
-   aggregation method;
-   chunk search depth;
-   evaluation configuration.

Chunk generation must be deterministic.

Identical:

``` text
corpus
+
configuration
+
chunking version
```

must produce identical:

-   chunk identities;
-   parent IDs;
-   ordering;
-   content;
-   required metadata.

------------------------------------------------------------------------

## 22. Validation Gates

### Gate 1 --- Corpus integrity

Verify:

-   59,636 articles preserved;
-   IDs `0..59,635` unchanged;
-   8 empty articles preserved;
-   no source article lost.

### Gate 2 --- Chunk integrity

Verify:

-   valid parent IDs;
-   unique chunk IDs;
-   valid ordering;
-   valid text coverage;
-   valid token counts;
-   required metadata;
-   valid offsets where present.

### Gate 3 --- Determinism

Two identical runs must produce identical chunk IDs, parent IDs, order,
content, and required metadata.

### Gate 4 --- Structural parser integrity

For structure-aware strategies:

-   report parse success/failure;
-   report fallback frequency;
-   validate sampled boundaries;
-   fail or explicitly fallback when malformed structure could change
    experimental meaning.

### Gate 5 --- Benchmark integrity

The existing benchmark files and labels remain unchanged.

### Gate 6 --- Leakage

Verify:

-   no frozen-test tuning;
-   no benchmark modification;
-   no cross-split contamination;
-   no selection rule derived from frozen-test metrics.

### Gate 7 --- Retrieval evaluation

Verify:

-   chunk retrieval;
-   chunk → article mapping;
-   aggregation;
-   duplicate-parent removal;
-   article-level metric computation.

### Gate 8 --- Regression

Existing Phase 5 and Phase 6 components remain functional.

------------------------------------------------------------------------

## 23. Required Artifacts

Expected roles:

``` text
data/processed/legal_retrieval/
    corpus_chunks_<version>.parquet
```

and:

``` text
results/benchmarks/
    chunking_corpus_audit.json
    chunking_experiment_results.json
    chunking_selection_manifest.json
    chunking_test_metrics.json
    chunking_test_predictions.json
    chunking_vs_phase6_comparison.json
```

Exact filenames may follow existing project conventions, but artifact
roles must remain identifiable.

The final chunk corpus must be sufficient to rebuild downstream
embeddings/indexes without reparsing the original source.

------------------------------------------------------------------------

## 24. Expected Implementation Components

A suitable logical structure is:

``` text
src/retrieval/chunking/
    base.py
    fixed_size.py
    recursive.py
    sentence.py
    paragraph.py
    structure_aware.py
    models.py
```

Possible scripts:

``` text
scripts/retrieval/
    audit_chunking_corpus.py
    build_chunks.py
    evaluate_chunking.py
    compare_chunking_strategies.py
```

These are implementation suggestions, not permission for unrelated
refactoring.

Existing Phase 5/6 modules must remain unchanged unless a demonstrated
interface-compatibility issue requires a minimal change.

------------------------------------------------------------------------

## 25. Error Handling

Fail explicitly for:

-   invalid chunk size;
-   invalid overlap;
-   unsupported strategy;
-   missing parent ID;
-   duplicate chunk identity;
-   malformed structure when no declared fallback is valid;
-   inconsistent chunk order;
-   text loss;
-   invalid benchmark mapping;
-   incompatible configuration;
-   corrupted artifact.

Silent fallback is not permitted when it changes the experimental
meaning.

For structure-aware strategies, any declared fallback must be:

-   deterministic;
-   measurable;
-   recorded in the experiment report.

------------------------------------------------------------------------

## 26. No Hidden Confounders

Do not silently change:

-   source text normalization;
-   tokenizer;
-   embedding model;
-   embedding representation;
-   pooling;
-   vector normalization;
-   query preprocessing;
-   relevance labels;
-   metric definitions;
-   article identity;
-   evaluation code.

Any such change requires a separate experiment.

------------------------------------------------------------------------

## 27. Research Hypotheses

These are hypotheses, not conclusions.

### H1

Smaller retrieval units may improve local semantic matching.

### H2

Very small chunks may lose legal context and reduce retrieval quality.

### H3

Structure-aware boundaries may be better aligned with Vietnamese legal
semantics than generic boundaries.

### H4

Recursive splitting may provide a useful low-complexity compromise when
formatting is sufficiently regular.

### H5

Sentence and paragraph boundaries may help preserve natural units, but
Vietnamese legal sentence/clause structure may limit their benefit.

### H6

Overlap may reduce boundary-related misses but increase chunk count,
storage, and duplicate candidates.

### H7

There is no justified assumption that 256, 512, or 1024 tokens is
universally optimal for this corpus.

No hypothesis may be presented as a project finding before measurement.

------------------------------------------------------------------------

## 28. Explicit Non-Decisions

Remain open until development evidence is available:

``` text
optimal chunk size
optimal overlap
final chunking strategy
final legal structural hierarchy
final aggregation beyond max-score
final chunk search depth
parent-child retrieval policy
context-injection policy
semantic chunking
proposition-level chunking
contextual retrieval
late chunking
RAPTOR/hierarchical retrieval
```

------------------------------------------------------------------------

## 29. Completion Criteria

Phase 7 can be declared **COMPLETE / FROZEN** only when:

1.  corpus granularity audit is complete;
2.  required chunking strategies are implemented;
3.  required staged experiments are executed;
4.  chunk provenance is validated;
5.  article-level aggregation is validated;
6.  benchmark integrity is confirmed;
7.  development/internal-validation selection is completed without
    frozen-test use;
8.  selection manifest is frozen;
9.  frozen test evaluation is completed once;
10. Phase 6 comparison is reported;
11. chunking artifacts are fingerprinted and validated;
12. reproducibility checks pass;
13. relevant tests pass;
14. final Phase 7 report is generated.

------------------------------------------------------------------------

## 30. Execution Order

``` text
1. Inspect existing corpus/schema
        ↓
2. Validate corpus and benchmark invariants
        ↓
3. Run corpus token/structure audit
        ↓
4. Predeclare internal development/validation split if needed
        ↓
5. Implement chunk data model and provenance
        ↓
6. Implement fixed-size chunking
        ↓
7. Implement recursive chunking
        ↓
8. Implement sentence/paragraph chunking
        ↓
9. Implement structure-aware chunking
        ↓
10. Validate deterministic generation
        ↓
11. Run C0 Phase 6 article-level reproduction
        ↓
12. Run Tier 1 screening
        ↓
13. Run Tier 2 structure-aware experiments if justified
        ↓
14. Run conditional size/overlap refinement
        ↓
15. Analyze quality/cost and per-query deltas
        ↓
16. Freeze development configuration
        ↓
17. Build final chunk corpus
        ↓
18. Freeze selection manifest
        ↓
19. Run frozen test once
        ↓
20. Compare against Phase 6
        ↓
21. Validate final artifacts
        ↓
22. Generate Phase 7 final report
```

------------------------------------------------------------------------

## 31. Agent / Implementation Rules

Any implementation agent must:

-   read this specification completely before coding;
-   inspect the actual project architecture first;
-   preserve corpus and benchmark contracts;
-   preserve Phase 6 as a frozen baseline;
-   avoid unrelated refactoring;
-   avoid changing Phase 5/6 behavior;
-   implement typed, testable components;
-   add focused unit/integration tests;
-   run validation gates before declaring completion.

If repository reality conflicts with this specification:

``` text
STOP
↓
Report the exact conflict
↓
Identify the affected file/contract
↓
Do not silently choose an interpretation
```

The agent must not resolve a specification conflict by inventing a new
requirement.

------------------------------------------------------------------------

## 32. Final Design Lock

At the end of Phase 7, the project must have an empirically supported
answer to:

> **What chunking strategy, granularity, and configuration should be
> used for Vietnamese legal documents in the downstream
> retrieval/indexing pipeline?**

The answer must be based on:

``` text
controlled development experiments
+
article-level evaluation
+
engineering measurements
+
provenance validation
+
reproducibility
```

The Phase 6 frozen-test result remains a final comparison reference
only. It must not influence Phase 7 selection.

The final Phase 7 output becomes the input contract for Phase 8:

``` text
Canonical Legal Corpus
        ↓
Legal Article
        ↓
FROZEN CHUNKING CONFIGURATION
        ↓
Chunk Corpus
        ↓
Embedding / Vector Indexing
```

Phase 8 must consume the frozen Phase 7 chunking output rather than
independently redefining chunk boundaries.
