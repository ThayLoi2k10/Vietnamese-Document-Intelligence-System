# Phase 8 --- Vector Database / Document Indexing Specification

**PROJECT 2 --- Vietnamese Legal Document Intelligence / Vietnamese
Legal Retrieval System**

**Status:** Specification / Implementation Contract\
**Phase:** 8 --- Vector Database / Document Indexing\
**Predecessor:** Phase 7 --- Document Chunking\
**Reference Baseline:** Phase 6 --- BGE-M3 + FAISS IndexFlatIP

------------------------------------------------------------------------

## 1. Purpose

Phase 8 converts the frozen Phase 7 chunk corpus into a **persistent,
metadata-aware, maintainable vector-indexing layer** suitable for the
downstream retrieval pipeline.

Phase 8 is not primarily about obtaining a higher retrieval score by
changing the embedding model. Its purpose is to establish reliable
indexing infrastructure supporting:

-   persistence;
-   chunk identity and provenance;
-   metadata retrieval/filtering;
-   deterministic upsert;
-   update/delete;
-   incremental ingestion;
-   consistency validation;
-   rebuild and rollback;
-   query/retrieval integration;
-   measurable latency and storage behavior.

The central question is:

> **What indexing architecture provides the required persistence,
> metadata, lifecycle, and retrieval behavior for the frozen Phase 7
> chunk corpus without introducing unnecessary operational complexity?**

------------------------------------------------------------------------

## 2. Scope

### 2.1 In scope

-   Final Phase 7 chunk-corpus consumption
-   Vector-store requirements analysis
-   Vector-index abstraction
-   Persistent vector storage
-   Vector + metadata schema
-   Chunk-to-article provenance
-   Similarity search
-   Metadata filtering
-   Deterministic point/chunk identity
-   Upsert
-   Update
-   Delete/deactivation
-   Incremental ingestion
-   Idempotency
-   Consistency validation
-   Reconciliation
-   Rebuild
-   Backup/recovery
-   Local development deployment
-   Exact/ANN comparison where relevant
-   Latency, throughput, memory, and storage benchmarking
-   End-to-end integration with retrieval

### 2.2 Out of scope

-   Changing the Phase 7 chunking strategy
-   Redefining chunk boundaries
-   Fine-tuning BGE-M3
-   Changing the Phase 6 embedding representation without a separate
    experiment
-   Hybrid retrieval
-   Reranking
-   RAG generation
-   Production API authentication
-   Full distributed deployment unless required by measured project
    scale
-   Unrelated database/application refactoring

------------------------------------------------------------------------

## 3. Locked Inputs

### 3.1 Frozen Phase 7 output

Phase 8 must consume the **frozen Phase 7 chunking configuration and
chunk corpus**.

The vector layer must not independently:

-   split chunks;
-   merge chunks;
-   alter chunk order;
-   alter `global_article_id`;
-   invent new chunk identities.

If a different chunking configuration is desired, it is a new Phase 7
version and must be completed/frozen before indexing.

### 3.2 Parent identity

Existing:

``` text
global_article_id = 0..59,635
```

is immutable.

The 8 empty articles remain represented in the source/provenance layer
even if they have no vector points.

### 3.3 Embedding contract

Unless explicitly registered as a separate experiment:

``` text
Model: BAAI/bge-m3
Dimension: 1024
Representation: native BGE-M3 dense representation
Normalization: L2
Dtype: float32
```

The vector store must accept the frozen Phase 7 embedding representation
without changing it.

------------------------------------------------------------------------

## 4. Design Principle

The vector database is a **derived index**, not the canonical legal
corpus.

Conceptually:

``` text
Canonical legal corpus
        ↓
Frozen Phase 7 chunk corpus
        ↓
Frozen embeddings
        ↓
Vector database / index
```

The authoritative source remains the corpus/chunk metadata layer.

The vector store must be rebuildable from authoritative chunk +
embedding artifacts.

This prevents a corrupted or stale vector index from becoming the only
source of truth.

------------------------------------------------------------------------

## 5. Functional Requirements

The Phase 8 indexing layer must support the following.

### FR-1 Persistent storage

Vectors and required metadata survive process restart.

### FR-2 Deterministic identity

The same logical chunk must map to the same logical vector-store point
identity within a corpus/version.

### FR-3 Similarity search

Given a normalized query vector, return:

``` text
chunk_id
score
rank
```

plus required provenance metadata.

### FR-4 Metadata retrieval

A result must be traceable to:

``` text
chunk_id
global_article_id
law_id
article_identifier
chunk_order
chunking_version
```

### FR-5 Metadata filtering

The architecture must support filters required by the project, subject
to the requirements audit.

Potential filter fields include:

``` text
law_id
document type
effective/status metadata
global_article_id
chunking_version
```

A field must not be indexed merely because it exists. Filter indexes
should be justified by actual query requirements.

### FR-6 Upsert

A chunk/vector can be inserted or replaced idempotently.

### FR-7 Delete/deactivation

A chunk can be removed or marked inactive according to the lifecycle
policy.

### FR-8 Incremental ingestion

New or changed legal content can be indexed without requiring a full
rebuild for every ingestion event.

### FR-9 Reconciliation

The system can detect discrepancies between authoritative metadata and
vector-store state.

### FR-10 Rebuild

The complete vector index can be reconstructed from authoritative
artifacts.

### FR-11 Version isolation

Different chunking/model/index versions must not silently collide.

### FR-12 Validation

Index integrity can be checked independently of retrieval quality.

------------------------------------------------------------------------

## 6. Non-Functional Requirements

### NFR-1 Correctness

Vector-store results must correspond to the intended stored vectors and
metadata.

### NFR-2 Provenance

Every retrieved vector must be traceable to exactly one chunk and one
parent article.

### NFR-3 Reproducibility

A rebuild from the same frozen artifacts must produce an equivalent
logical index.

### NFR-4 Idempotency

Retrying the same ingestion operation must not create duplicate logical
chunks.

### NFR-5 Recoverability

A failed indexing operation must not leave untracked partial state.

### NFR-6 Performance

Latency and throughput must be measured at the actual Phase 7 chunk
scale rather than assumed from generic benchmarks.

### NFR-7 Maintainability

The architecture should minimize custom lifecycle logic that duplicates
functionality already provided by the chosen vector store.

------------------------------------------------------------------------

## 7. Architecture Boundary

The recommended logical architecture is:

``` text
                 ┌─────────────────────────────┐
                 │ Canonical Legal Corpus      │
                 └──────────────┬──────────────┘
                                ↓
                 ┌─────────────────────────────┐
                 │ Frozen Phase 7 Chunk Corpus │
                 └──────────────┬──────────────┘
                                ↓
                 ┌─────────────────────────────┐
                 │ BGE-M3 Embedding Layer      │
                 └──────────────┬──────────────┘
                                ↓
                 ┌─────────────────────────────┐
                 │ Vector Index / Database     │
                 │                             │
                 │ vector + payload + identity │
                 └──────────────┬──────────────┘
                                ↓
                 ┌─────────────────────────────┐
                 │ Similarity Search           │
                 │ + optional metadata filter  │
                 └──────────────┬──────────────┘
                                ↓
                 ┌─────────────────────────────┐
                 │ Chunk Results                │
                 └──────────────┬──────────────┘
                                ↓
                 ┌─────────────────────────────┐
                 │ Phase 7 Article Aggregation │
                 └─────────────────────────────┘
```

Phase 8 owns the indexing layer and lifecycle.

Phase 7 owns chunk boundaries and chunk semantics.

------------------------------------------------------------------------

## 8. Candidate Technologies

The research considered:

-   FAISS;
-   Qdrant;
-   Milvus;
-   pgvector;
-   Weaviate;
-   ChromaDB.

The decision must be based on actual project requirements and measured
scale.

### 8.1 FAISS

Strengths:

-   exact `IndexFlatIP`;
-   simple;
-   mature similarity-search library;
-   low overhead;
-   useful as the correctness/reference index.

Limitations for Phase 8:

-   persistence is file-based rather than database-style lifecycle
    management;
-   metadata filtering is external;
-   update/delete semantics require custom management;
-   application-level consistency logic becomes the project's
    responsibility.

FAISS therefore remains a **reference/research index**, even if it is
not selected as the primary lifecycle store.

### 8.2 Qdrant

Candidate strengths:

-   persistent vector storage;
-   payload metadata;
-   filtering;
-   upsert;
-   delete/update lifecycle;
-   Python client;
-   local deployment option;
-   path toward server deployment.

Qdrant is a strong candidate for the Phase 8 prototype because its
capabilities align with the required lifecycle features.

However, Phase 8 must not claim that Qdrant is automatically optimal
before requirements and benchmarks are completed.

### 8.3 Milvus

Strengths:

-   large-scale distributed architecture;
-   multiple ANN/index options;
-   broad scaling capabilities.

Potential issue:

-   operational complexity may be disproportionate to the current
    corpus/chunk scale.

Milvus should therefore require explicit scale/deployment justification.

### 8.4 pgvector

Strengths:

-   SQL and vector search in PostgreSQL;
-   strong transactional semantics;
-   metadata filtering through SQL;
-   exact and ANN search.

It becomes particularly relevant if PostgreSQL is already a
system-of-record dependency.

### 8.5 Weaviate / ChromaDB

Both may be useful in specific application contexts, but the project
must justify their inclusion based on requirements rather than feature
count.

------------------------------------------------------------------------

## 9. Technology Decision Protocol

Do not decide solely from vendor feature lists.

### Step 1 --- Requirements audit

Record whether the project actually requires:

-   persistence;
-   metadata filtering;
-   incremental upsert;
-   delete/update;
-   local-only operation;
-   server deployment;
-   transactional metadata;
-   expected growth;
-   backup/recovery;
-   operational simplicity.

### Step 2 --- Scale audit

Use the actual frozen Phase 7 corpus to measure:

-   total chunks;
-   vector dimension;
-   vector storage;
-   metadata storage;
-   expected annual growth;
-   query volume.

Do not assume a fixed average number of chunks per article.

### Step 3 --- Prototype candidates

At minimum, compare:

``` text
FAISS reference
+
one production-oriented candidate
```

Qdrant should be the first production-oriented candidate to prototype
unless the requirements audit gives a concrete reason to prioritize
another technology.

### Step 4 --- Benchmark

Measure:

-   correctness;
-   metadata filtering correctness;
-   search latency;
-   ingestion throughput;
-   upsert latency;
-   delete/update behavior;
-   restart persistence;
-   rebuild time;
-   memory;
-   disk usage.

### Step 5 --- Decision

Document the selected architecture and the evidence supporting it.

Do not use a generic statement such as "HNSW is faster" as a substitute
for a project-specific benchmark.

------------------------------------------------------------------------

## 10. Exact Search vs. ANN

### 10.1 Research reference

Phase 6 FAISS `IndexFlatIP` remains the exact-search reference.

It provides a correctness oracle for sampled or controlled comparisons.

### 10.2 ANN

ANN may be introduced when the selected vector store uses HNSW or
another approximate index.

The decision must be based on:

``` text
measured recall / ranking agreement
+
latency
+
memory
+
build/update cost
```

Do not assume that a reported percentage such as "99% recall" transfers
directly to this corpus.

### 10.3 Required comparison

For an ANN candidate, evaluate a representative query sample against the
exact reference.

Report:

-   Recall@K against exact results;
-   rank agreement where useful;
-   latency;
-   memory/storage;
-   build time.

The exact index remains the correctness reference, not necessarily the
production index.

------------------------------------------------------------------------

## 11. Vector Point Schema

Each vector-store point must logically contain:

### Identity

``` text
point_id
chunk_id
```

### Provenance

``` text
global_article_id
law_id
article_identifier
chunk_order
total_chunks
chunking_version
```

### Retrieval metadata

``` text
token_count
content_hash
chunking_strategy
```

### Optional legal metadata

Only where available and required:

``` text
document type
issuing authority
effective date
status
chapter/section identifiers
```

### Payload text

The store may retain `chunk_text` if the chosen architecture requires
retrieval-time context access.

If chunk text is stored outside the vector database, the mapping to the
authoritative chunk corpus must be deterministic and efficient.

------------------------------------------------------------------------

## 12. Point Identity

The logical identity must be deterministic.

Preferred conceptual mapping:

``` text
chunk_id
    ↓
point_id
```

The same chunk must not produce a new random point ID on every retry.

If the chosen store has ID-format restrictions, a deterministic encoding
or stable mapping table may be used.

Do not use a random UUID as the only identity without retaining a
deterministic mapping back to `chunk_id`.

------------------------------------------------------------------------

## 13. Collection / Index Versioning

A vector collection/index must identify at least:

``` text
embedding_model
embedding_revision
embedding_dimension
embedding_representation
normalization
dtype
chunking_version
corpus_fingerprint
```

A change in any vector-compatible dimension or semantics must not
silently overwrite an incompatible collection.

Examples requiring a new index/collection version:

-   embedding model change;
-   embedding revision change;
-   chunking version change;
-   vector dimension change;
-   normalization change;
-   incompatible metadata schema change.

------------------------------------------------------------------------

## 14. Lifecycle State Model

The indexing pipeline should expose explicit lifecycle states.

Conceptual states:

``` text
discovered
    ↓
parsed
    ↓
chunked
    ↓
embedded
    ↓
indexed
    ↓
committed
```

Failure path:

``` text
any state
    ↓
failed
```

A failed item must contain enough information to retry or reconcile.

------------------------------------------------------------------------

## 15. Incremental Ingestion Pipeline

The logical pipeline is:

``` text
New / changed legal document
        ↓
1. Identity + deduplication
        ↓
2. Parse / article extraction
        ↓
3. Clean / normalize
        ↓
4. Chunk using frozen Phase 7 configuration
        ↓
5. Validate chunks
        ↓
6. Embed with frozen BGE-M3 configuration
        ↓
7. Upsert vectors + metadata
        ↓
8. Validate write
        ↓
9. Commit source/index state
        ↓
10. Record ingestion result
```

Phase 8 must not reimplement chunking rules. It consumes the Phase 7
chunking component/output.

------------------------------------------------------------------------

## 16. Identity and Deduplication

At minimum, distinguish:

``` text
document identity
article identity
chunk identity
content version
```

Recommended source-level concepts:

``` text
document_id
document_version
content_hash
```

For existing corpus articles:

``` text
global_article_id
```

remains immutable.

For future new articles, IDs must follow the project's authoritative
ID-allocation policy and must never reuse an existing ID.

------------------------------------------------------------------------

## 17. Idempotent Upsert

An ingestion retry must be safe.

Conceptually:

``` text
same chunk_id
+
same chunking_version
+
same content_hash
→
same logical vector point
```

If the content hash changes:

``` text
same parent identity
+
new content version
→
new chunk representation
```

The old representation must be deactivated or replaced according to the
versioning policy.

------------------------------------------------------------------------

## 18. Update Handling

When an article's source content changes:

``` text
1. detect content/version change
2. generate new Phase 7 chunks
3. validate all new chunks
4. embed new chunks
5. write new vector points
6. verify write
7. deactivate old chunk points
8. update authoritative index state
```

Do not delete the old representation before the new representation has
passed validation unless the selected store/transaction design
explicitly guarantees safe rollback.

------------------------------------------------------------------------

## 19. Delete / Revocation Handling

For a revoked or inactive legal source:

``` text
1. mark source/article inactive in authoritative metadata
2. deactivate or delete associated vector points
3. preserve historical identity/provenance
4. record the lifecycle event
```

The project must define whether retrieval filters inactive content by
default.

That policy is a product/retrieval requirement and must not be invented
by the indexing implementation.

------------------------------------------------------------------------

## 20. Partial Failure and Consistency

A vector store and authoritative metadata store may not share one
transaction.

The implementation therefore needs an explicit consistency model.

Minimum requirements:

-   every write has a deterministic operation identity;
-   partial failures are recorded;
-   retries are safe;
-   reconciliation can identify missing or stale points;
-   no successful ingestion is reported without verification.

A practical model is:

``` text
Authoritative state
        ↓
pending
        ↓
vector upsert
        ↓
verification
        ↓
committed
```

If indexing fails:

``` text
pending / failed
```

must remain visible to the reconciliation process.

Do not claim full distributed atomicity unless the selected
infrastructure actually provides it.

------------------------------------------------------------------------

## 21. Reconciliation

A reconciliation process should be able to detect:

### Missing vector

Authoritative chunk exists but vector point does not.

### Stale vector

Vector point exists but content/version no longer matches authoritative
state.

### Orphan vector

Vector point references a chunk that no longer exists in the
authoritative active corpus.

### Metadata mismatch

Vector payload does not match the authoritative chunk metadata.

The system should report counts and affected IDs.

------------------------------------------------------------------------

## 22. Rebuild Strategy

A full rebuild is required when:

-   chunking strategy changes;
-   embedding model/revision changes;
-   embedding configuration changes;
-   major metadata schema changes;
-   index corruption occurs;
-   a clean migration is required.

Recommended procedure:

``` text
1. create new index/collection
2. load frozen authoritative chunks
3. load/reuse compatible embeddings
4. insert/upsert all points
5. validate counts and metadata
6. compare retrieval against reference
7. promote new index
8. retain old index for rollback window
9. remove old index only after validation
```

The old index must not be destroyed before the new index passes
validation.

------------------------------------------------------------------------

## 23. Backup and Recovery

The project must preserve:

``` text
chunk corpus
+
chunk metadata
+
embedding artifacts
+
index configuration
```

The vector database itself is a derived artifact.

Where embedding artifacts are available, an index can be rebuilt without
repeating expensive BGE-M3 inference.

Recovery must be tested, not merely documented.

------------------------------------------------------------------------

## 24. Query Interface Contract

The Phase 8 vector layer should expose a minimal typed interface
conceptually equivalent to:

``` text
search(
    query_vector,
    limit,
    filter=None
) -> list[VectorSearchResult]
```

Each result should contain enough information for Phase 7-compatible
aggregation:

``` text
chunk_id
global_article_id
score
rank
```

and required metadata where requested.

The vector layer must not perform article-level aggregation internally
unless explicitly designed as a separate layer.

Recommended boundary:

``` text
Vector store
    ↓
chunk results
    ↓
retrieval/aggregation layer
    ↓
article results
```

------------------------------------------------------------------------

## 25. Filtering Contract

Metadata filtering must be applied according to the selected store's
supported semantics.

The implementation must distinguish:

``` text
pre-filter
```

from:

``` text
post-filter
```

because post-filtering can reduce the number of valid candidates and
change retrieval behavior.

For any required filter, tests must verify:

1.  only allowed metadata values are returned;
2.  no disallowed point leaks into results;
3.  filtered retrieval remains correctly ranked;
4.  candidate limits remain sufficient.

Do not claim filter correctness from schema presence alone.

------------------------------------------------------------------------

## 26. Retrieval Compatibility

Phase 8 must preserve the Phase 7 retrieval contract:

``` text
query
  ↓
BGE-M3 query embedding
  ↓
vector search
  ↓
top-K chunks
  ↓
chunk_id → global_article_id
  ↓
max-score article aggregation
  ↓
article-level evaluation
```

Phase 8 must not silently introduce:

-   hybrid scoring;
-   BM25;
-   reranking;
-   score fusion;
-   query rewriting.

Those belong to later phases.

------------------------------------------------------------------------

## 27. Benchmark Plan

### 27.1 Correctness benchmark

Compare the selected vector-store retrieval against the Phase 6 exact
FAISS reference on the same frozen embeddings for a controlled query
sample.

Measure:

-   Recall@1/5/10/20/50/100;
-   ranking agreement where useful;
-   score consistency where applicable.

### 27.2 Performance benchmark

Measure:

-   cold-start latency;
-   warm query latency;
-   p50;
-   p95;
-   p99;
-   QPS;
-   memory;
-   disk usage.

### 27.3 Lifecycle benchmark

Measure:

-   insert throughput;
-   batch upsert throughput;
-   single-point update;
-   batch update;
-   delete/deactivation;
-   restart persistence;
-   recovery;
-   rebuild time.

### 27.4 Filter benchmark

For every required metadata filter:

-   correctness;
-   latency;
-   candidate recall;
-   interaction with `limit`.

------------------------------------------------------------------------

## 28. Scale Benchmark

Use the **actual Phase 7 chunk count**.

Do not hard-code assumptions such as:

``` text
3 chunks/article
5 chunks/article
500K vectors
```

before Phase 7 produces the actual corpus statistics.

If future growth is evaluated, define explicit synthetic scaling
factors, for example:

``` text
1× current corpus
2× current corpus
5× current corpus
```

and document how the scale dataset was produced.

------------------------------------------------------------------------

## 29. Engineering Measurements

At minimum record:

``` text
vector count
vector dimension
vector bytes
metadata bytes
total disk size
RAM usage
build time
batch upsert throughput
query p50
query p95
query p99
QPS
restart/load time
rebuild time
```

Also record:

``` text
CPU
GPU
software versions
vector-store version
configuration
```

This prevents vendor benchmarks from being mistaken for project
measurements.

------------------------------------------------------------------------

## 30. Testing Requirements

### Unit tests

Test:

-   deterministic point IDs;
-   payload serialization;
-   metadata mapping;
-   filter construction;
-   upsert idempotency;
-   delete/update behavior;
-   configuration compatibility.

### Integration tests

Test:

``` text
chunk
 ↓
embedding
 ↓
vector-store upsert
 ↓
search
 ↓
metadata retrieval
 ↓
article aggregation
```

### Persistence test

``` text
write
 ↓
close/restart
 ↓
search
```

must preserve expected results.

### Reconciliation test

Intentionally create:

-   missing point;
-   stale point;
-   orphan point;

and verify that reconciliation detects them.

### Failure test

Simulate indexing failure and verify that:

-   partial state is detectable;
-   retry is safe;
-   final state is consistent.

------------------------------------------------------------------------

## 31. Required Artifacts

Expected roles:

``` text
data/indices/
    <vector_store>/
        config.json
        manifest.json
        collection/index artifacts
```

and:

``` text
results/benchmarks/
    vector_store_requirements.json
    vector_store_benchmark.json
    vector_store_correctness.json
    vector_store_lifecycle_test.json
    vector_store_filter_test.json
    vector_store_final_report.md
```

Exact paths/names may follow existing project conventions.

Every artifact must record the relevant:

-   Phase 7 chunking version;
-   corpus fingerprint;
-   embedding fingerprint;
-   vector-store version;
-   configuration.

------------------------------------------------------------------------

## 32. Reproducibility

A reproducible index build must be tied to:

``` text
corpus fingerprint
+
chunking version
+
embedding model/revision
+
embedding configuration
+
vector-store configuration
+
software versions
```

The build manifest must make it possible to answer:

> Which exact chunk corpus and embedding artifacts produced this index?

------------------------------------------------------------------------

## 33. Security / Operational Boundary

Phase 8 does not implement public API authentication or authorization.

However, the design must avoid embedding application secrets directly
into:

-   source code;
-   committed configuration;
-   index artifacts;
-   benchmark outputs.

Connection settings should be environment/configuration driven.

Local development should not require exposing a vector service publicly.

------------------------------------------------------------------------

## 34. Agent / Implementation Rules

Any implementation agent must:

-   read this specification completely before coding;
-   inspect the actual Phase 7 output and repository architecture;
-   preserve Phase 7 as frozen;
-   preserve Phase 6 as the exact/reference baseline;
-   avoid unrelated refactoring;
-   use existing abstractions where possible;
-   implement typed, testable components;
-   add focused unit/integration tests;
-   run validation and benchmark gates before declaring completion.

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

Do not invent a missing product requirement.

------------------------------------------------------------------------

## 35. Completion Criteria

Phase 8 can be declared **COMPLETE / FROZEN** only when:

1.  Phase 7 frozen chunk corpus is consumed unchanged;
2.  vector-store requirements are documented;
3.  candidate architecture is evaluated;
4.  selected vector store/index is implemented;
5.  persistent restart behavior passes;
6.  deterministic identity and idempotent upsert pass;
7.  update/delete lifecycle passes;
8.  metadata filtering passes where required;
9.  reconciliation passes;
10. rebuild/recovery procedure passes;
11. exact-vs-ANN correctness comparison is completed where ANN is used;
12. performance benchmarks are completed at actual Phase 7 scale;
13. integration with retrieval and article aggregation passes;
14. artifacts are fingerprinted and reproducible;
15. relevant tests pass;
16. final Phase 8 report is generated.

------------------------------------------------------------------------

## 36. Execution Order

``` text
1. Read frozen Phase 7 chunking output
        ↓
2. Validate chunk/corpus/embedding invariants
        ↓
3. Audit Phase 8 functional requirements
        ↓
4. Measure actual chunk/vector scale
        ↓
5. Define vector-store abstraction
        ↓
6. Build FAISS exact reference compatibility
        ↓
7. Prototype Qdrant or justified alternative
        ↓
8. Implement vector + payload schema
        ↓
9. Implement deterministic point identity
        ↓
10. Implement persistent upsert/search
        ↓
11. Implement filtering where required
        ↓
12. Implement update/delete/deactivation
        ↓
13. Implement incremental ingestion lifecycle
        ↓
14. Implement reconciliation
        ↓
15. Implement rebuild/recovery
        ↓
16. Run correctness benchmark
        ↓
17. Run lifecycle/filter/performance benchmarks
        ↓
18. Select and freeze indexing architecture
        ↓
19. Run end-to-end integration tests
        ↓
20. Validate final artifacts
        ↓
21. Generate Phase 8 final report
```

------------------------------------------------------------------------

## 37. Final Design Lock

The final Phase 8 architecture must be justified by:

``` text
actual Phase 7 scale
+
functional requirements
+
correctness measurements
+
latency/throughput measurements
+
storage/memory measurements
+
lifecycle tests
+
reproducibility
```

The final architecture must preserve this boundary:

``` text
Phase 7
= decides WHAT the retrieval unit is

Phase 8
= decides HOW those frozen retrieval units are persisted,
  indexed, filtered, updated, and recovered
```

The vector store is a derived index over the frozen Phase 7 chunk
corpus.

It must not independently redefine chunk semantics.
