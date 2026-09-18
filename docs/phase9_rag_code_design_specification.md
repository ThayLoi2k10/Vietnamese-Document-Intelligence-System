# Phase 9 — RAG Code Design Specification

**Status:** Design specification / implementation-ready blueprint  
**Artifact:** `phase9_rag_code_design_specification.md`  
**Phase:** Project 2 — Vietnamese Legal Document Intelligence / Vietnamese Legal Retrieval System  
**Purpose:** Source of truth for the implementation of Phase 9 RAG/LLM software architecture.

---

## 1. Purpose

Phase 9 defines the software architecture required to turn an existing legal retrieval capability into a testable Retrieval-Augmented Generation (RAG) system for Vietnamese legal questions.

The target logical flow is:

```text
User Question
      ↓
Retrieval
      ↓
Relevant Chunks / Articles
      ↓
Context Assembly
      ↓
Prompt Construction
      ↓
LLM Generation
      ↓
Structured Answer + Legal Evidence
```

The implementation must be designed around explicit interfaces so that retrieval, reranking, context policies, prompting, LLM inference, parsing, citation validation, abstention, and evaluation can be replaced independently.

The specification is intentionally a **software/code design document**, not a research report and not an implementation.

### Design status legend

- **[LOCK]** — Requirement is fixed for Phase 9 and implementation must conform to it.
- **[OPEN]** — Intentionally undecided; implementation must expose a stable extension point rather than silently choosing a final value.
- **[HYP]** — Hypothesis/candidate for experimentation; never treat as a production or architecture requirement.
- **[ASSUMPTION]** — Necessary architectural assumption made solely to make interfaces explicit; it must remain easy to revise.

---

## 2. Scope

### 2.1 In scope [LOCK]

1. Retriever abstraction and Phase 8 adapter boundary.
2. Optional reranking abstraction.
3. Retrieval-result normalization.
4. Context selection and assembly.
5. Context deduplication.
6. Optional parent/context expansion when supported upstream.
7. Context ordering policy.
8. Context token-budget management.
9. Versioned prompt architecture.
10. LLM generation abstraction.
11. Structured legal answer representation.
12. Citation validation.
13. Abstention as a first-class behavior.
14. RAG pipeline orchestration.
15. Evaluation interfaces.
16. Configuration-driven experiments.
17. Frozen-test protection.
18. Reproducibility metadata and artifact contracts.
19. Unit, integration, and leakage-test design.
20. Explicit error taxonomy.

### 2.2 Existing upstream contract [LOCK]

Phase 6 frozen dense retrieval established:

- Model: `BAAI/bge-m3`.
- Dense representation: official/native dense representation.
- Dimension: 1024.
- Vector normalization: L2 normalized.
- dtype: float32.
- Exact FAISS reference implementation: `IndexFlatIP`.
- Retrieval depth reference: top-100.
- Immutable article identity: `global_article_id`.
- Frozen test set: 438 queries.

Phase 6 reference metrics:

- nDCG@10 = 0.6277
- Recall@10 = 0.7969
- Recall@100 = 0.9441
- MRR@10 = 0.6146

These metrics are **historical reference values only**. They are not Phase 9 validation thresholds and must not be used as tuning objectives for Phase 9.

### 2.3 Upstream implementation status [LOCK]

Phase 7–8 chunking/indexing implementation is **not assumed to be locked** for Phase 9. Phase 9 must communicate through an abstraction that can consume either:

- chunk-level retrieval, or
- article-level retrieval.

When chunk-level retrieval is used, parent/article relationships and chunk metadata may be consumed if actually supplied by the adapter.

When article-level retrieval is used, Phase 9 must not manufacture a mandatory chunk architecture around it.

---

## 3. Non-Goals

The following are explicitly outside Phase 9 code architecture [LOCK]:

- Fine-tuning, LoRA, QLoRA, or any LLM training.
- LLM pretraining or training infrastructure.
- Production deployment architecture.
- API gateway/deployment infrastructure.
- Autoscaling.
- Cloud platform selection.
- GPU provider selection.
- Production monitoring/observability architecture.
- Multi-turn conversation management.
- Agentic RAG.
- Autonomous tool-using agents.
- Redesign of Phase 5–8 retrieval systems.
- Benchmark execution during this design stage.
- Actual LLM/model loading or inference during this design stage.

Future Work may mention these subjects, but Phase 9 must not make them hard dependencies.

---

## 4. Phase 8 → Phase 9 Interface

### 4.1 Boundary principle [LOCK]

Phase 9 depends on a **retrieval domain contract**, not on FAISS, Qdrant, or any other vector/database implementation.

The Phase 8 adapter is responsible for translating the upstream retrieval implementation into Phase 9 domain objects.

```text
Phase 8 implementation
    ↓
Phase 8 Retrieval Adapter
    ↓
Phase 9 RetrievalResult contract
    ↓
RAG pipeline
```

### 4.2 Required normalized retrieval concepts [LOCK]

A retrieval response must be able to represent:

- query text;
- retrieval mode (`CHUNK` or `ARTICLE`);
- zero or more ranked retrieval items;
- stable item/source identities;
- source text;
- ranking information;
- score information when supplied by the upstream retriever;
- legal/document metadata necessary for provenance;
- optional parent/article relationships;
- optional token count when known.

### 4.3 Normalized `RetrievalResult` schema [LOCK]

Logical schema:

```text
RetrievalResult
├── query: string
├── retrieval_mode: RetrievalMode
├── items: List<RetrievalItem>
├── aggregated_articles: List<ArticleResult> | null
└── retriever_metadata: RetrievalMetadata
```

`retriever_metadata` must remain internal unless explicitly selected for prompt exposure.

### 4.4 `RetrievalItem` schema [LOCK]

A normalized retrieval item should contain:

| Field | Required | Meaning |
|---|---:|---|
| `chunk_id` | No | Stable chunk identity when chunk retrieval exists. |
| `global_article_id` | No/conditional | Immutable article identity when available. |
| `law_id` | Yes | Legal document/source identity. |
| `article_identifier` | Yes/conditional | Human/legal article identifier when applicable. |
| `chunk_text` / `text` | Yes | Retrieved legal text. |
| `chunk_order` | No | Position of chunk within its source/article. |
| `total_chunks` | No | Number of chunks belonging to the parent/article, when known. |
| `score` | No | Retrieval relevance score. |
| `rank` | Yes | Rank at the originating retrieval stage. |
| `token_count` | No | Token count computed with an agreed tokenizer, if available. |
| `parent_reference` | No | Parent/article relationship if supplied by upstream. |
| `source_metadata` | No | Additional normalized provenance metadata. |

At least one stable provenance identity must exist for every usable item. The exact required identity depends on retrieval mode:

- chunk mode: `chunk_id` should be present;
- article mode: an article/source identity must be present;
- `global_article_id` must be preserved whenever upstream provides it.

### 4.5 `ArticleResult` schema [LOCK]

Logical schema:

```text
ArticleResult
├── global_article_id: string | null
├── law_id: string
├── article_identifier: string | null
├── best_chunk_score: number | null
├── article_rank: integer
├── contributing_chunks: List<ReferenceIdentity>
└── article_metadata: SourceMetadata
```

`ArticleResult` is an optional normalized aggregation view. It does not imply that the retriever must be article-level internally.

### 4.6 Score semantics [LOCK]

Scores must be treated as **retriever- or reranker-defined relevance signals**, not as probabilities and not as directly comparable across models unless the component contract explicitly establishes comparability.

The score field must be accompanied by component identity/configuration in experiment metadata.

A reranker must be allowed to produce a new score without overwriting the original retrieval score in a way that loses provenance. Prefer distinct semantic fields at the domain level, e.g. original retrieval score versus reranking score.

### 4.7 Rank semantics [LOCK]

Every ranked collection must identify which stage produced the rank.

At minimum distinguish:

```text
retrieval_rank
rerank_rank
context_rank
```

These are not interchangeable.

### 4.8 Adapter responsibilities [LOCK]

The adapter must:

- validate and normalize Phase 8 output;
- preserve immutable identities;
- preserve text and legally relevant metadata;
- expose retrieval mode explicitly;
- detect malformed or incomplete input;
- avoid leaking database/vector-engine concepts into Phase 9 domain models.

The adapter must not:

- build prompts;
- decide LLM model selection;
- perform citation semantic evaluation;
- silently fill required fields with fabricated values;
- silently reorder or truncate results unless that behavior is explicitly part of the adapter contract.

---

## 5. Architecture Overview

### 5.1 Logical architecture [LOCK]

```text
                         ┌──────────────────────┐
                         │   ExperimentConfig   │
                         └──────────┬───────────┘
                                    │
                                    ▼
┌─────────────┐   ┌──────────────┐   ┌────────────────┐
│ User Query  │ → │   Retriever  │ → │ Optional       │
└─────────────┘   └──────────────┘   │ Reranker       │
                                     └───────┬────────┘
                                             ▼
                                   ┌──────────────────┐
                                   │ Context Assembler│
                                   └────────┬─────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │ Prompt Builder   │
                                   └────────┬─────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │ LLM Generator    │
                                   └────────┬─────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │ Output Parser    │
                                   └────────┬─────────┘
                                            ▼
                               ┌─────────────────────────┐
                               │ Citation + Abstention  │
                               │ Validation              │
                               └───────────┬─────────────┘
                                           ▼
                                  ┌──────────────────┐
                                  │ RAG Response     │
                                  └──────────────────┘

                   ┌───────────────────────────────────────┐
                   │ Separate Evaluation / Experiment Ops │
                   └───────────────────────────────────────┘
```

### 5.2 Baseline path [LOCK]

```text
Query
  ↓
Retriever
  ↓
Context Assembler
  ↓
Prompt Builder
  ↓
LLM Generator
  ↓
Structured Output Parser
  ↓
Citation Validation
  ↓
Abstention Validation
  ↓
RAG Response
```

### 5.3 Reranking path [LOCK]

```text
Query
  ↓
Retriever
  ↓
Top-N retrieval candidates
  ↓
Optional Reranker
  ↓
Top-K context candidates
  ↓
Context Assembler
  ↓
Prompt Builder
  ↓
LLM Generator
  ↓
Structured Output Parser
  ↓
Citation Validation
  ↓
Abstention Validation
  ↓
RAG Response
```

### 5.4 Separation of concerns [LOCK]

The system must keep these concepts distinct:

```text
retrieval depth
≠ reranking depth
≠ final context K
≠ evaluation cutoff
```

No component may infer one from another unless the configuration explicitly defines that relationship.

---

## 6. Package / Module Structure

### 6.1 Recommended structure [LOCK]

```text
src/
  rag/
    retrieval/
      domain/
      interfaces/
      adapters/
    reranking/
      interfaces/
      adapters/
    context/
      domain/
      policies/
    prompting/
      domain/
      templates/
      builders/
    generation/
      domain/
      interfaces/
      adapters/
      mocks/
    output/
      schemas/
      parsers/
    citation/
      validation/
      provenance/
    abstention/
      detection/
      policies/
    pipeline/
    evaluation/
      interfaces/
      metrics/
      backends/
    config/
      schemas/
      loaders/
      validation/
    reproducibility/
    errors/

tests/
  unit/
    retrieval/
    reranking/
    context/
    prompting/
    generation/
    output/
    citation/
    abstention/
    config/
  integration/
  leakage/
  fixtures/

configs/
  experiments/
  prompts/
  datasets/

artifacts/
  schemas/
  experiment-manifests/
  evaluation-reports/
```

### 6.2 Module responsibility matrix [LOCK]

| Module | Responsibility | Must NOT own |
|---|---|---|
| `retrieval` | Normalized retrieval contract and adapters | Prompting or generation |
| `reranking` | Optional candidate reranking contract/adapters | Context formatting or LLM calls |
| `context` | Candidate selection, deduplication, expansion, ordering, budget enforcement | Prompt wording or generation |
| `prompting` | Versioned prompt composition and instruction policy | Retrieval implementation or parsing model output |
| `generation` | Provider/model-agnostic LLM request/response contract | Citation semantic evaluation |
| `output` | Structured response schema and parsing | Retrieval or model selection |
| `citation` | Citation syntactic/provenance validation | Determining semantic truth of claims |
| `abstention` | Abstention detection, validation, and reason taxonomy | Retrieval or generation itself |
| `pipeline` | Orchestrating components through dependency injection | Provider-specific behavior |
| `evaluation` | Evaluation datasets, evaluators, reports | Mutating production pipeline behavior |
| `config` | Configuration schema, validation, and experiment semantics | Hidden runtime defaults |
| `reproducibility` | Run manifest/version metadata | Choosing experiment outcomes |
| `errors` | Shared explicit error taxonomy | Silent recovery |

### 6.3 Dependency direction [LOCK]

Dependencies should point inward toward stable domain contracts.

```text
Adapters / external integrations
              ↓
      Interfaces / ports
              ↓
        Domain models
```

The core pipeline must depend on interfaces, not concrete model or database clients.

---

## 7. Domain Models

### 7.1 Core value objects [LOCK]

At minimum, define domain objects for:

```text
Question
RetrievalItem
RetrievalResult
ArticleResult
RankedCandidate
ContextItem
AssembledContext
PromptRequest
GenerationRequest
GenerationResult
LegalAnswer
Citation
CitationValidationResult
AbstentionDecision
RAGResponse
ExperimentConfig
ExperimentManifest
EvaluationInput
EvaluationReport
```

### 7.2 Identity model [LOCK]

Identity fields must remain distinct:

- `law_id` — source/legal-document identity.
- `article_identifier` — legal article label when applicable.
- `global_article_id` — immutable article identity established upstream.
- `chunk_id` — chunk identity when chunking is actually used.

No component may overwrite one identity with another merely because they are convenient.

### 7.3 Provenance model [LOCK]

Every user-visible citation must be traceable back through:

```text
RAGResponse
  → Citation
  → Retrieved evidence reference
  → RetrievalItem / ContextItem
  → Source identity
```

Provenance must be explicit enough to validate existence and, when applicable, quote matching.

### 7.4 Internal vs public metadata [LOCK]

Maintain a deliberate boundary between:

```text
Internal retrieval metadata
```

and

```text
LLM-exposed evidence metadata
```

Fields such as vector score, internal rank bookkeeping, adapter names, database IDs, or debugging information must not automatically enter the prompt.

The context representation must expose only metadata needed for grounding, citation, legal readability, or explicit controlled experimentation.

---

## 8. Retriever Interface

### 8.1 Interface [LOCK]

Logical interface:

```text
Retriever
  retrieve(query, retrieval_options) → RetrievalResult
```

### 8.2 Required retriever options [LOCK]

`retrieval_options` must be able to represent at minimum:

- retrieval depth (`retrieval_top_k`);
- retrieval mode when explicitly requested/supported;
- deterministic execution requirement where relevant;
- component version/identifier for provenance.

The core pipeline must not assume a specific vector store.

### 8.3 Retriever implementations [OPEN]

Potential implementations may wrap:

- Phase 8 index/search implementation;
- FAISS;
- Qdrant;
- another compatible retrieval engine.

The architecture does not lock any one backend beyond preserving the Phase 6 frozen retrieval contract where that contract is applicable.

### 8.4 Empty retrieval [LOCK]

An empty result is a valid domain outcome and must be explicitly represented rather than converted into an exception solely because no candidate was found.

However, malformed retrieval records are errors.

### 8.5 Retrieval validation [LOCK]

The adapter must reject or explicitly classify:

- missing required source identity;
- missing required text;
- invalid rank;
- invalid rank ordering where ordering is required by contract;
- inconsistent retrieval mode;
- malformed metadata;
- duplicated identity where the upstream contract requires uniqueness.

No silent repair should change experiment semantics.

---

## 9. Reranker Interface

### 9.1 Interface [LOCK]

Logical interface:

```text
Reranker
  rerank(query, candidates, options) → RankedResults
```

### 9.2 Optionality [LOCK]

The reranker is optional.

The pipeline must support:

```text
Reranker = None
```

or an injected reranker implementation.

The absence of a reranker must not require a special retrieval implementation.

### 9.3 Candidate models [HYP]

Current research candidates:

- BGE-Reranker-v2-M3
- ViRanker

Neither is a locked winner. The core code must not contain winner-specific branching.

### 9.4 Reranker contract [LOCK]

The reranker must:

- consume normalized retrieval candidates;
- preserve original source identity;
- produce a reranking score and rerank rank;
- preserve original retrieval score/rank for provenance and analysis;
- support configurable reranking depth.

The reranker must not:

- build prompts;
- decide final context budget;
- mutate legal source text;
- perform LLM generation;
- decide evaluation cutoffs.

### 9.5 Distinct depth parameters [LOCK]

The configuration must explicitly distinguish:

```text
retrieval_top_k
reranker_top_k
context_top_k
```

For clarity, `reranker_top_k` means the number of retrieved candidates passed through reranking; `context_top_k` means the maximum number of candidates selected for assembled context, subject to token budget and policy.

---

## 10. Context Assembly

### 10.1 Responsibility [LOCK]

`ContextAssembler` converts ranked retrieval evidence into a bounded, ordered, deduplicated context suitable for prompt construction.

Logical interface:

```text
ContextAssembler
  assemble(ranked_evidence, context_policy) → AssembledContext
```

### 10.2 Assembly pipeline [LOCK]

```text
Retrieved Results
      ↓
Optional score filtering
      ↓
Optional reranking (performed by pipeline before assembly)
      ↓
Deduplication
      ↓
Optional parent/context expansion
      ↓
Context ordering
      ↓
Token budget check
      ↓
Final Context
```

The actual reranking operation belongs to the `Reranker` stage, not to the `ContextAssembler`; the diagram indicates that assembly consumes already reranked candidates when reranking is enabled.

### 10.3 Selection policy [LOCK]

Configurable fields:

```text
context_top_k
score_threshold (optional)
maximum_context_items
```

These are context-selection controls and must not be conflated with retrieval depth.

### 10.4 Deduplication [LOCK]

Deduplication policy must be explicit and configurable.

Minimum supported concepts:

1. exact duplicate chunk identity;
2. duplicate source identity where policy requires article-level uniqueness;
3. overlapping chunks;
4. multiple chunks from the same article.

Recommended logical policy dimensions:

```text
DUPLICATE_CHUNK_ID
DUPLICATE_TEXT_EXACT
OVERLAP_BY_TOKEN_OR_CHARACTER_THRESHOLD
SAME_ARTICLE_LIMIT
```

Exact overlap thresholds are [OPEN] and must be configuration, not hidden constants.

Deduplication must retain enough provenance to explain which candidates contributed to the final context.

### 10.5 Parent/context expansion [OPEN]

Parent expansion is conditional.

It may be enabled only when the retrieval data actually supports parent/article relationships.

Possible policies:

```text
DISABLED
PARENT_ARTICLE
NEIGHBORING_CHUNKS
ARTICLE_CONTEXT_WINDOW
```

No Phase 9 code may assume that every `chunk_id` has a parent or that every retrieval source is chunked.

### 10.6 Ordering [LOCK]

Ordering is a policy abstraction.

Supported candidate strategies:

- descending relevance/reranking score;
- original legal/document order.

Additional strategies can be added later without changing the pipeline contract.

Ordering must preserve the identity/provenance of each context item.

### 10.7 Context representation [LOCK]

Each `ContextItem` should contain at minimum:

```text
context_item_id
source identity
law_id
article_identifier (when available)
global_article_id (when available)
chunk_id (when available)
text
source rank / score metadata as internal provenance
context position
```

Only an explicitly selected subset becomes LLM-visible.

### 10.8 Token budget [LOCK]

Context assembly must enforce:

```text
prompt tokens
+
context tokens
+
expected output tokens
≤ configured model context budget
```

The exact budget is [OPEN]. There must be no hard-coded default such as `128K` treated as universal.

### 10.9 Budget failure behavior [LOCK]

The assembler must use an explicit policy for budget overflow, such as:

```text
TRUNCATE_BY_POLICY
DROP_LOWEST_PRIORITY_ITEMS
REQUIRE_SMALLER_CONTEXT_K
RAISE_CONTEXT_BUDGET_ERROR
```

The chosen policy must be encoded in experiment configuration. No silent fallback may alter experiment semantics.

### 10.10 Token counting dependency [OPEN]

The exact tokenizer used for context budgeting depends on the eventual generation model.

The `TokenCounter` abstraction should therefore be injected or explicitly associated with the generation configuration rather than hidden inside the context assembler.

---

## 11. Prompt Architecture

### 11.1 Components [LOCK]

Prompt construction must be decomposed into:

```text
PromptTemplate
PromptContext
GenerationInstructions
PromptBuilder
```

### 11.2 `PromptTemplate` [LOCK]

A versioned template defines the structural arrangement of instructions, question, and evidence.

Example conceptual versions:

```text
prompt_v1
prompt_v2
...
```

The version ID is part of experiment metadata.

### 11.3 `GenerationInstructions` [LOCK]

Instructions must cover:

1. Grounding: answer from retrieved legal evidence.
2. Citation: cite supporting evidence using structured citation references.
3. Abstention: indicate insufficient support when evidence is not adequate.
4. Legal answer formatting.
5. User question interpretation.
6. Retrieved evidence usage.

Prompt instructions are **controls, not guarantees** against hallucination.

### 11.4 `PromptContext` [LOCK]

The prompt context object should separate:

```text
system/instruction content
user query
evidence items
citation reference mapping
prompt version
```

The builder is responsible for deterministic assembly of these inputs.

### 11.5 Prompt builder responsibilities [LOCK]

`PromptBuilder` must:

- validate required prompt inputs;
- apply the configured template version;
- expose only intended evidence fields;
- preserve a deterministic evidence-reference mapping;
- emit a `PromptRequest` suitable for the generator.

It must not:

- retrieve documents;
- call the LLM;
- validate semantic citation correctness;
- decide model selection.

### 11.6 Prompt validity [LOCK]

Invalid prompts are explicit errors, not silently auto-corrected in ways that change the experiment.

---

## 12. LLM Generator Interface

### 12.1 Interface [LOCK]

Logical interface:

```text
LLMGenerator
  generate(request: GenerationRequest) → GenerationResult
```

### 12.2 Model abstraction [LOCK]

The core architecture must not hard-code a model name.

Current candidates [HYP] include:

- Qwen3-8B
- Qwen3-14B
- Qwen3-32B
- future models selected for further research

### 12.3 `GenerationRequest` [LOCK]

Minimum logical fields:

```text
model_identifier
prompt
max_output_tokens
generation_parameters
thinking_mode
reproducibility_options
request_metadata
```

### 12.4 `GenerationResult` [LOCK]

Minimum logical fields:

```text
model_identifier
raw_response
parsed_response (optional until parsing stage)
generation_metadata
usage_metadata (when available)
finish_status
```

The result must preserve raw output for controlled debugging/evaluation, while public output uses the validated structured schema.

### 12.5 Thinking mode [OPEN]

Thinking/non-thinking is a configuration option when the selected model supports it.

No core module may assume that:

- all models support thinking mode;
- thinking mode has one universal API representation;
- reasoning traces are part of user-visible output.

### 12.6 Determinism [LOCK]

The interface must support deterministic/reproducible execution where the underlying model/provider allows it.

Reproducibility metadata must capture generation parameters, but code must not falsely claim bit-for-bit determinism for a backend that cannot guarantee it.

### 12.7 Mock generator [LOCK]

A fake/mock implementation must exist for unit and integration tests.

Tests must be able to execute without loading or downloading a real model.

---

## 13. Structured Output

### 13.1 Public schema [LOCK]

Logical public output:

```json
{
  "answer": "string",
  "citations": [],
  "confidence_level": "FULL | PARTIAL | INSUFFICIENT",
  "abstention_reason": "string | null"
}
```

### 13.2 `Citation` object [LOCK]

A citation object must support:

```text
citation_number
law_id
article_identifier
chunk_id (optional)
global_article_id (optional)
supporting_evidence_or_quote (optional)
```

Additional internal provenance fields may exist outside the public schema.

### 13.3 Confidence semantics [LOCK]

`confidence_level` is a categorical support-status field, not a calibrated probability.

Meaning:

- `FULL`: retrieved evidence sufficiently supports the answer at the configured evaluation/application policy.
- `PARTIAL`: evidence supports only part of the requested answer or material qualifications remain.
- `INSUFFICIENT`: available evidence is not sufficient for a grounded answer.

The exact decision policy is [OPEN] and must be evaluated rather than assumed correct from the label alone.

### 13.4 Reasoning trace [LOCK]

No chain-of-thought or hidden reasoning trace may be included in the public answer schema.

Internal debugging metadata, when needed, must be stored separately and must not be surfaced as public answer content by default.

### 13.5 Parsing [LOCK]

The parser must distinguish:

```text
valid structured output
malformed structured output
schema-valid but semantically invalid references
```

These states must not be collapsed.

---

## 14. Citation Validation

### 14.1 Responsibility [LOCK]

Citation validation is a post-generation layer.

Logical interface:

```text
CitationValidator
  validate(answer, evidence_registry) → CitationValidationResult
```

### 14.2 Minimum syntactic/provenance checks [LOCK]

1. Citation number/reference is well formed.
2. Citation reference exists in the retrieved evidence registry.
3. Citation target identity is valid.
4. Citation does not point to an unavailable source.
5. Quote, when provided, can be matched against source evidence according to the configured matching policy.
6. Overall output conforms to the expected response schema.

### 14.3 Syntactic vs semantic correctness [LOCK]

The system must explicitly separate:

```text
citation syntactic validity
```

from

```text
citation semantic correctness
```

Parser-level validation can establish existence and consistency, but it cannot by itself prove that a citation semantically supports a legal claim.

Semantic correctness is an evaluation concern.

### 14.4 Evidence registry [LOCK]

The pipeline should maintain an internal immutable mapping:

```text
citation/exposed evidence reference
      → ContextItem
      → RetrievalItem provenance
```

This registry is the authoritative source for syntactic citation validation within a run.

### 14.5 Invalid citation behavior [LOCK]

Invalid citation output must result in an explicit validation state/error according to configuration.

There must be no hidden behavior that silently replaces citations with unrelated sources.

---

## 15. Abstention

### 15.1 First-class behavior [LOCK]

Abstention is a domain behavior, not a string-matching heuristic.

Supported states:

```text
FULL
PARTIAL
INSUFFICIENT
```

### 15.2 Conceptual signal flow [LOCK]

```text
retrieval not sufficient
        ↓
context not sufficient
        ↓
answer support insufficient
        ↓
LLM instructed to abstain
        ↓
abstention validation
        ↓
RAGResponse
```

### 15.3 `AbstentionDecision` [LOCK]

Logical fields:

```text
status
reason_code
explanation (optional)
supporting_evidence_count
policy_version
```

### 15.4 Detection [LOCK]

The system must not determine abstention solely by looking for a keyword such as `"không biết"`.

Detection may combine:

- retrieval/context sufficiency signals;
- structured output status;
- missing/invalid citations;
- explicit model abstention field;
- configurable answer-support checks.

Exact combined policy is [OPEN] and belongs in a versioned abstention policy.

### 15.5 Confidence and calibration [LOCK]

`FULL/PARTIAL/INSUFFICIENT` must not be interpreted as calibrated probability unless an explicit calibration experiment and contract are later added.

### 15.6 Abstention evaluation [LOCK]

Evaluation must distinguish:

- correct answer when support is sufficient;
- correct abstention when support is insufficient;
- incorrect answer despite insufficient support;
- unnecessary abstention when adequate support exists;
- partial answer misclassified as full.

---

## 16. RAG Pipeline

### 16.1 Orchestrator [LOCK]

The central orchestration object is:

```text
RAGPipeline
```

It must accept injected dependencies for:

- `Retriever`;
- optional `Reranker`;
- `ContextAssembler`;
- `PromptBuilder`;
- `LLMGenerator`;
- `OutputParser`;
- `CitationValidator`;
- `AbstentionDetector/Validator`;
- relevant policy/config objects.

### 16.2 Execution flow [LOCK]

```text
User Query
    ↓
Validate question/config
    ↓
Retriever
    ↓
Optional Reranker
    ↓
Context Assembler
    ↓
Prompt Builder
    ↓
LLM Generator
    ↓
Output Parser
    ↓
Citation Validator
    ↓
Abstention Validator
    ↓
RAG Response
```

### 16.3 Pipeline responsibilities [LOCK]

The orchestrator must:

- enforce component ordering;
- pass typed domain objects between stages;
- preserve provenance;
- apply configuration explicitly;
- collect run metadata needed for reproducibility;
- surface explicit errors.

The orchestrator must not:

- call concrete FAISS/Qdrant APIs directly;
- contain prompt strings;
- contain model-specific parsing rules;
- implement deduplication itself;
- encode model winner logic;
- silently fallback between incompatible configurations.

### 16.4 Dependency injection [LOCK]

No hidden singleton/global component should be required by the pipeline core.

The same pipeline should be constructible with:

```text
real Retriever + mock LLM
mock Retriever + mock LLM
mock Retriever + mock Reranker + mock LLM
```

This is required for test isolation and controlled ablation.

### 16.5 Deterministic experiment semantics [LOCK]

Every behavior that could affect evaluation must be attributable to configuration or component version.

Examples:

- retrieval top-k;
- reranker enabled/model/depth;
- context ordering;
- deduplication policy;
- expansion policy;
- prompt version;
- model identifier;
- thinking mode;
- generation parameters;
- token budgets.

---

## 17. Evaluation Architecture

### 17.1 Separation [LOCK]

Evaluation must be a separate subsystem from production-like generation orchestration.

The evaluator consumes dataset items and predictions; it must not silently mutate the generation configuration.

### 17.2 Top-level interface [LOCK]

Logical interface:

```text
RAGEvaluator
  evaluate(dataset, predictions, evaluation_config) → EvaluationReport
```

### 17.3 Retrieval metrics [LOCK]

The evaluation layer may consume retrieval evaluator outputs including:

- Recall@k;
- MRR@10;
- nDCG@10;
- MAP.

Phase 9 must not unnecessarily reimplement Phase 5–8 retrieval evaluation if a compatible evaluator abstraction already exists.

### 17.4 Generation evaluation interfaces [LOCK]

Provide independent evaluator contracts for:

```text
FaithfulnessEvaluator
AnswerRelevanceEvaluator
CitationPrecisionEvaluator
CitationRecallEvaluator
AbstentionCorrectnessEvaluator
UnsupportedClaimEvaluator
```

They may be composed into `RAGEvaluator`.

### 17.5 RAGAS backend [OPEN/HYP]

RAGAS may be used as a backend/candidate evaluator, but the domain evaluation API must remain independent of RAGAS.

The architecture must make it possible to replace RAGAS with another evaluator without rewriting the RAG domain model.

### 17.6 Evaluation report [LOCK]

Logical report fields:

```text
experiment_id
dataset_split
sample_count
metric_results
metric_configurations
component_versions
error_summary
reproducibility_metadata
```

### 17.7 Semantic evaluation [LOCK]

Semantic citation correctness and unsupported-claim detection are evaluation tasks. They are not guaranteed by structured parsing alone.

---

## 18. Experiment Configuration

### 18.1 Configuration-driven design [LOCK]

Experiments must be fully represented by configuration rather than ad hoc code edits.

### 18.2 Required configuration dimensions [LOCK]

```text
retrieval_top_k
reranker_enabled
reranker_model
reranker_top_k
context_top_k
context_ordering
deduplication_policy
parent_expansion
prompt_version
llm_model
thinking_mode
generation_parameters
max_context_tokens
max_output_tokens
```

Additional fields may be added for reproducibility and evaluator configuration.

### 18.3 Configuration object hierarchy [LOCK]

Recommended conceptual structure:

```text
ExperimentConfig
├── experiment_id
├── dataset
├── retriever
├── reranker
├── context
├── prompting
├── generation
├── evaluation
├── reproducibility
└── validation
```

Each nested object should validate its own invariants while the top-level validator checks cross-component compatibility.

### 18.4 Model names [OPEN]

Model identifiers are configuration values, not source-code constants.

### 18.5 One-variable-at-a-time experiments [LOCK]

Experiment definitions must encode a base configuration and a controlled delta when the objective is one-variable-at-a-time comparison.

The experiment system must make the following explicit:

```text
baseline configuration
changed parameter(s)
unchanged parameters
split
component versions
```

The architecture must prevent accidental multi-variable changes from being represented as a nominal single-variable experiment.

---

## 19. Dataset / Test Protection

### 19.1 Split contract [LOCK]

Phase 9 dataset structure:

```text
Training retrieval queries: 1,752
    ↓
RAG Development: 1,401
RAG Validation: 351

Frozen Test: 438
```

### 19.2 Development split [LOCK]

RAG development may be used for:

- prompt experiments;
- context experiments;
- top-k experiments;
- reranker experiments;
- LLM candidate experiments.

### 19.3 Validation split [LOCK]

RAG validation is used to compare/tune configurations prior to final locking.

### 19.4 Frozen test [LOCK]

The frozen test set must not be used to:

- tune a prompt;
- select a model;
- select top-k;
- select reranker;
- select context ordering;
- select a context policy;
- optimize generation parameters.

It is executed only after the experiment configuration is locked.

### 19.5 Configuration-level leakage prevention [LOCK]

Dataset configuration must carry an explicit split role, at minimum:

```text
DEVELOPMENT
VALIDATION
FROZEN_TEST
```

The experiment configuration validator should reject any experiment declared as tuning/selection/development when its split is `FROZEN_TEST`.

### 19.6 Access model [LOCK]

The frozen test split should be treated as a protected artifact with an explicit access API rather than a generic dataset path that every experiment can freely load.

### 19.7 Test-leakage invariant [LOCK]

No code path should require the test split merely to initialize the RAG pipeline.

Frozen-test access must occur in an explicit evaluation operation after lock state has been established.

---

## 20. Error Handling

### 20.1 Principles [LOCK]

Errors must be:

- typed;
- explicit;
- attributable to a pipeline stage;
- preserved in experiment logs;
- non-silent when they can alter semantics.

### 20.2 Required error classes [LOCK]

At minimum:

```text
InvalidRetrievalResultError
MissingRequiredMetadataError
EmptyRetrievalError (when empty retrieval is escalated by a configured policy)
ContextBudgetExceededError
InvalidPromptError
LLMGenerationError
MalformedStructuredOutputError
InvalidCitationError
UnsupportedModelConfigurationError
EvaluationDataLeakageError
IncompatibleComponentVersionError
ConfigurationValidationError
```

### 20.3 Empty retrieval distinction [LOCK]

An empty retrieval result is semantically different from a malformed retrieval result.

The first is a valid pipeline state that may lead to abstention; the second is a data/adapter error.

### 20.4 No hidden fallback [LOCK]

Do not silently:

- switch retrievers;
- disable reranking;
- alter context K;
- downgrade/upgrade model;
- replace prompt version;
- choose another dataset split;
- ignore invalid citations.

Any fallback that changes experiment semantics must be explicit configuration and logged.

### 20.5 Error provenance [LOCK]

Every raised or recorded error should be attributable to:

```text
experiment_id
pipeline_stage
component_identifier
configuration snapshot
```

when available.

---

## 21. Reproducibility

### 21.1 Required metadata [LOCK]

Every experiment/run manifest must track:

- experiment ID;
- model ID and revision when available;
- prompt version;
- retriever configuration;
- reranker configuration;
- context policy;
- generation parameters;
- dataset split;
- code version;
- timestamp;
- evaluation configuration.

### 21.2 Component versioning [LOCK]

Version identifiers must be preserved for all experiment-relevant components, including:

```text
retriever
reranker
prompt template
context policy
output schema version
abstention policy
citation validation policy
generator/model
```

### 21.3 Reproducibility manifest [LOCK]

Recommended manifest structure:

```text
ExperimentManifest
├── experiment_id
├── created_at
├── code_version
├── dataset
├── retriever
├── reranker
├── context
├── prompt
├── generation
├── output_schema_version
├── citation_policy_version
├── abstention_policy_version
└── evaluation
```

### 21.4 Determinism caveat [LOCK]

Reproducibility metadata should make an execution reproducible **to the extent supported by the underlying inference stack**. The architecture must not falsely claim deterministic output for stochastic or nondeterministic components.

---

## 22. Testing Strategy

### 22.1 Test philosophy [LOCK]

Testing must validate contracts and pipeline semantics without requiring production models or real inference.

### 22.2 Unit tests [LOCK]

#### Retrieval

- normalization of valid retrieval responses;
- validation of required metadata;
- rank/score preservation;
- chunk/article mode handling;
- empty retrieval behavior.

#### Reranking

- disabled reranker path;
- candidate preservation;
- new score/rank creation;
- original retrieval provenance preservation;
- invalid candidate handling.

#### Context

- top-k selection;
- score thresholding;
- duplicate chunk removal;
- overlap handling;
- same-article policy;
- parent expansion enabled/disabled;
- relevance ordering;
- document/legal ordering;
- context position assignment;
- token budget enforcement;
- deterministic overflow behavior.

#### Prompting

- template version selection;
- deterministic prompt construction;
- evidence mapping;
- prompt input validation;
- LLM-visible metadata filtering.

#### Generation

- request validation;
- parameter propagation;
- mock generator behavior;
- raw response preservation;
- generation failure propagation.

#### Output

- valid structured output parsing;
- malformed JSON/schema rejection;
- enum validation for confidence state;
- abstention-reason validation.

#### Citation

- valid citation reference;
- citation to missing evidence;
- duplicate/ambiguous citation IDs;
- quote matching when enabled;
- syntax vs semantic evaluation separation.

#### Abstention

- sufficient context + answer;
- insufficient retrieval;
- insufficient context;
- explicit structured abstention;
- incorrect heuristic-like text that should not itself force abstention.

#### Configuration

- missing required fields;
- incompatible component combinations;
- invalid dataset role;
- invalid model thinking-mode configuration;
- invalid context budget configuration;
- frozen-test tuning rejection.

### 22.3 Integration tests [LOCK]

Minimum chains:

```text
Retriever → Context
Context → Prompt
Prompt → LLM adapter
LLM output → Parser
Parser → Citation validation
```

Full pipeline with mock components:

```text
Mock Retriever
    ↓
Mock Reranker (optional)
    ↓
Real ContextAssembler
    ↓
Real PromptBuilder
    ↓
Mock LLM
    ↓
Real OutputParser
    ↓
Real CitationValidator
    ↓
Real AbstentionValidator
```

This enables broad pipeline verification with no real model inference.

### 22.4 Leakage tests [LOCK]

Must include automated tests that prove:

- frozen test is not available to development/tuning configuration;
- selecting `FROZEN_TEST` for a tuning/selection experiment raises an explicit error;
- frozen test is invoked only through an explicit final-evaluation pathway;
- an experiment manifest records the dataset split.

### 22.5 Contract tests [LOCK]

Each adapter should have contract tests against the shared interface so replacing one backend does not silently change domain semantics.

### 22.6 Mock/fake strategy [LOCK]

LLM and external model inference must be mockable.

Suggested fake behaviors:

```text
Returns valid structured answer
Returns malformed output
Returns invalid citation
Returns abstention
Raises generation failure
Returns partial answer
```

No test should require downloading or loading an actual LLM unless a future dedicated model-integration test suite explicitly chooses to do so.

---

## 23. Configuration / Artifacts

### 23.1 Configuration files [LOCK]

Recommended separation:

```text
configs/
  experiments/
  prompts/
  datasets/
```

### 23.2 Prompt artifacts [LOCK]

Each prompt version should be immutable once an experiment using it is locked.

Logical metadata:

```text
prompt_version
prompt_name
template_content
instruction_policy_version
created_at
```

### 23.3 Experiment configuration artifact [LOCK]

Each experiment must have a machine-readable configuration artifact containing all experiment-relevant parameters.

### 23.4 Evaluation artifact [LOCK]

Each evaluation should produce an artifact containing:

- experiment ID;
- split;
- metrics;
- metric configuration;
- component versions;
- run metadata;
- error summary.

### 23.5 Schema artifacts [LOCK]

Public response schema, citation schema, and core experiment manifest schema should be versioned artifacts.

### 23.6 No implicit defaults for experimental variables [LOCK]

Experimental variables must have explicit configuration values. Defaults may exist only where they are semantically safe and documented; an unspecified value must not silently select a research condition.

---

## 24. Implementation Order

The following is an implementation plan only. Nothing in this section is executed by this specification.

1. **Domain schemas/interfaces**
   - Define stable domain types.
   - Define identity/provenance contracts.
   - Define error taxonomy.

2. **Phase 8 retrieval adapter**
   - Implement normalization into `RetrievalResult`.
   - Add adapter contract tests.

3. **Reranker abstraction**
   - Implement optional reranker port and mock/fake.
   - Add candidate-preservation tests.

4. **Context assembly**
   - Implement selection, deduplication, ordering, optional expansion, and token-budget policies.

5. **Prompt system**
   - Implement versioned templates, `PromptContext`, instructions, and `PromptBuilder`.

6. **LLM abstraction + mock**
   - Implement provider-neutral request/response types and mock generator.

7. **Output parser**
   - Implement public schema parsing and strict validation.

8. **Citation validator**
   - Implement syntactic/provenance checks and explicit validation results.

9. **Abstention handling**
   - Implement policy-based detection/validation and reason codes.

10. **RAG pipeline orchestrator**
    - Wire all components using dependency injection.

11. **Evaluation interfaces**
    - Add evaluator ports and report schemas.

12. **Experiment configuration**
    - Implement configuration schema, cross-field validation, and split protection.

13. **Unit tests**
    - Complete isolated component tests.

14. **Integration tests**
    - Complete mock end-to-end pipeline tests.

15. **Validation gates**
    - Verify architecture-level gates before any future experiments are allowed.

---

## 25. Validation Gates

### Gate 1 — Interface integrity [LOCK]

Pass criteria:

- Phase 9 core imports no concrete vector DB dependency.
- Retriever is accessible only through its interface.
- Reranker is optional.
- Generator is accessible only through its interface.
- Mock generator can execute the full pipeline.

### Gate 2 — Provenance integrity [LOCK]

Pass criteria:

- Every LLM-visible evidence item maps to a source identity.
- Every citation maps to a retrieved/context evidence reference.
- Original retrieval provenance remains available after reranking.

### Gate 3 — Context policy integrity [LOCK]

Pass criteria:

- retrieval depth, reranking depth, context K, and evaluation cutoff are distinct fields;
- deduplication and ordering policies are explicit;
- token budget is enforced through configuration;
- no universal hard-coded context length is assumed.

### Gate 4 — Output integrity [LOCK]

Pass criteria:

- structured output parser is strict;
- public schema excludes reasoning traces;
- citation syntax/provenance is validated;
- confidence level is categorical, not probabilistic.

### Gate 5 — Abstention integrity [LOCK]

Pass criteria:

- abstention is first-class;
- empty/insufficient retrieval can propagate to abstention;
- abstention is not a keyword-only heuristic;
- abstention reasons are structured.

### Gate 6 — Leakage protection [LOCK]

Pass criteria:

- `FROZEN_TEST` cannot be selected for tuning/selection experiments;
- final evaluation must explicitly opt into the frozen split after lock state;
- experiment manifests record the split.

### Gate 7 — Reproducibility [LOCK]

Pass criteria:

- experiment configuration fully captures experimental variables;
- prompt/model/component versions are recorded;
- code version and dataset split are recorded;
- generation parameters are recorded.

### Gate 8 — Testability [LOCK]

Pass criteria:

- unit tests do not require real LLM inference;
- integration tests can run entirely with mock external model components;
- component replacement does not require pipeline rewrites.

---

## 26. Open Decisions

The following decisions remain intentionally open and must not be silently locked by the implementation.

### 26.1 Retrieval mode [OPEN]

Whether the final upstream implementation is chunk-level or article-level retrieval remains open.

**Affected components:** retrieval adapter, context assembly, provenance, parent expansion.

**Required decision:** confirm the exact normalized Phase 8 payload once Phase 8 is locked.

### 26.2 Final reranker [OPEN/HYP]

Candidates currently identified:

- BGE-Reranker-v2-M3 [HYP]
- ViRanker [HYP]

**Affected components:** reranker adapter configuration, evaluation, experiment matrix.

**Required decision:** none at architecture stage; both must remain supported.

### 26.3 Final LLM [OPEN/HYP]

Candidates include Qwen3-8B, Qwen3-14B, Qwen3-32B, and potential future models.

**Affected components:** generation adapter, tokenizer/token counting, thinking mode, prompt constraints.

**Required decision:** none at architecture stage; model identifier remains configuration-driven.

### 26.4 Exact context token budget [OPEN]

The architecture requires a configurable budget but does not lock a numeric value.

**Affected components:** context policy and generation configuration.

### 26.5 Tokenizer used for budget accounting [OPEN]

The exact tokenizer depends on the generation model.

**Affected components:** `TokenCounter`, context assembler, generation configuration.

### 26.6 Parent/context expansion semantics [OPEN]

Exact expansion rules depend on the finalized upstream data model.

**Affected components:** context policy, provenance, deduplication.

### 26.7 Deduplication thresholds [OPEN]

Exact overlap criteria and same-article limits require future controlled experiments.

**Affected components:** context policy and experiment configuration.

### 26.8 Abstention decision policy [OPEN]

The architecture is locked around first-class abstention, but the exact threshold/policy for classifying `FULL`, `PARTIAL`, and `INSUFFICIENT` requires experimental validation.

**Affected components:** abstention policy, evaluation.

### 26.9 Citation quote matching policy [OPEN]

The minimum contract permits quote verification when a quote is present, but the exact matching strategy is not locked.

**Affected components:** citation validator and evaluator.

### 26.10 Semantic faithfulness evaluator [OPEN]

The interface is locked, but the final evaluator/backend and exact operationalization are not.

**Affected components:** evaluation subsystem only.

---

## 27. Future Work

The following may be addressed later without changing the Phase 9 core architecture:

- additional reranker implementations;
- additional LLM provider adapters;
- dedicated model-integration tests;
- more sophisticated semantic citation evaluation;
- calibrated confidence research;
- additional context expansion policies;
- richer evaluation backends;
- production deployment architecture;
- production observability and monitoring;
- multi-turn conversation state;
- agentic RAG;
- fine-tuning/model adaptation.

Future work must continue to respect the same dependency-inversion and provenance principles.

---

## 28. Design Lock

### 28.1 Locked architectural principles [LOCK]

1. **Separation of concerns** — each module owns one bounded responsibility.
2. **Dependency inversion** — core pipeline depends on interfaces, not providers.
3. **Configuration-driven experiments** — experiment conditions are data/configuration, not hidden code branches.
4. **Model/provider agnosticism** — no hard-coded LLM or vector DB in core architecture.
5. **Retrieval/LLM separation** — retrieval and generation have independent contracts.
6. **Evaluation separation** — evaluation does not redefine the generation pipeline.
7. **Explicit provenance** — source identity survives every transformation.
8. **Deterministic experiment configuration** — behavior-relevant configuration is explicit and versioned.
9. **No hidden fallback** — semantic changes require explicit configuration or explicit failure.
10. **No frozen-test leakage** — frozen test remains protected from tuning/selection.
11. **No unnecessary Phase 8 coupling** — chunk-level and article-level retrieval are both supported by abstraction.
12. **No premature optimization** — implementation should prioritize correctness, clarity, testability, and reproducibility before optimization.

### 28.2 Explicitly not locked [LOCK]

The following remain configuration/research decisions rather than architecture locks:

```text
final reranker
final LLM model
thinking mode choice
numeric context budget
exact tokenizer
exact deduplication threshold
exact parent expansion policy
exact abstention thresholds
exact semantic citation evaluator
vector/database implementation of Phase 8
```

### 28.3 Conflict policy [LOCK]

If an implementation artifact or future research result conflicts with a locked contract in this specification, the conflict must be surfaced explicitly.

Required procedure:

```text
STOP
  ↓
Record the conflict
  ↓
Identify affected component(s)
  ↓
Describe both interpretations
  ↓
Do not silently choose one
  ↓
Request an explicit architecture decision
```

No implementation detail may silently override a locked contract.

### 28.4 Final implementation-stage boundary [LOCK]

This document does **not** authorize:

- code execution;
- model loading;
- model download;
- corpus encoding;
- retrieval benchmark execution;
- RAG benchmark execution;
- large-scale answer generation;
- GPU workloads;
- cloud/GPU platform selection;
- deployment.

Its output is solely the code design specification required to begin implementation in a later step.

---
