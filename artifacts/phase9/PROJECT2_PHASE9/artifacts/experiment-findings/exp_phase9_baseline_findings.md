# Phase 9 — Baseline Experiment Findings

## Experiment

- Experiment ID: exp_phase9_baseline
- Code version: phase9-v0.9.0
- Dataset: vietnamese_legal_retrieval
- Dataset split: DEVELOPMENT
- Total benchmark queries: 1,401
- Successful queries: 1,379
- Failed queries: 22
- Execution success rate: 98.4297%
- Execution failure rate: 1.5703%

## Configuration

### Retrieval

- Retriever: phase8_vector_store
- Component version: bge-m3-dense-v1.0
- Retrieval mode: CHUNK
- Retrieval top-k: 10
- Distance metric: cosine

### Context

- Context top-k: 5
- Score threshold: 0.3
- Maximum context items: 5
- Maximum context tokens: 4096
- Deduplication policy: EXACT_CHUNK_ID
- Expansion policy: DISABLED
- Ordering strategy: SCORE_DESCENDING

### Generation

- Model: Qwen/Qwen2.5-7B-Instruct
- Temperature: 0.0
- Top-p: 1.0
- Maximum output tokens: 1024
- Seed: 42
- Repetition penalty: 1.1

### Reranking

- Reranker: DISABLED

## Quality Metrics

Metrics were computed on the 1,379 successfully evaluated queries.

| Metric | Value |
|---|---:|
| Citation Precision | 0.430898 |
| Citation Recall | 0.469982 |
| Abstention Correctness | 0.970268 |
| Faithfulness | 0.896962 |

## Execution Findings

1. The baseline successfully produced evaluable outputs for 1,379 of 1,401 benchmark queries.
2. 22 queries failed during structured-output processing.
3. The observed execution failure rate was 1.5703%.
4. The quality metrics do not include the 22 failed queries.

## Failure Findings

The 22 failed queries were classified as follows:

| Structural failure | Count |
|---|---:|
| INVALID_JSON_STRUCTURE | 19 |
| INVALID_JSON_CONTENT | 2 |
| EXTRA_DATA_AFTER_JSON | 1 |

Language contamination was observed as an additional orthogonal attribute:

| Language condition | Count |
|---|---:|
| NO_LANGUAGE_CONTAMINATION | 14 |
| CHINESE_CONTAMINATION | 8 |

Removing Markdown JSON fences alone did not recover any of the 22 failed outputs as valid JSON.

Therefore, the observed failures cannot be attributed solely to Markdown fencing. The raw outputs contain additional structural/content violations.

## Error Analysis Findings

- 22/22 failed outputs remained malformed after Markdown fence removal.
- 19/22 failures were classified as INVALID_JSON_STRUCTURE.
- 2/22 failures were classified as INVALID_JSON_CONTENT.
- 1/22 failures was classified as EXTRA_DATA_AFTER_JSON.
- 8/22 failures contained CHINESE_CONTAMINATION.
- 14/22 failures contained NO_LANGUAGE_CONTAMINATION.
- Chinese contamination is treated as an additional failure attribute rather than a separate structural failure category.
- The analysis does not establish a specific generation-level root cause such as maximum-token truncation without additional evidence.

## Interpretation

The Phase 9 baseline establishes a reproducible reference configuration for the RAG system.

The results provide measurements for:

- execution reliability,
- citation precision,
- citation recall,
- abstention correctness,
- faithfulness,
- and structured-output failure behavior.

These measurements are baseline observations. No claim of improvement, superiority, or optimality is made because no alternative Phase 9 configuration has yet been evaluated against this baseline.

The dataset split used for this experiment is DEVELOPMENT. It is not the frozen retrieval test split used in the earlier retrieval evaluation.

## Limitations

1. Quality metrics are computed only on successfully evaluated queries.
2. The 22 failed queries are excluded from those quality metric calculations.
3. The current experiment does not isolate the contribution of individual RAG components.
4. No reranker comparison was performed.
5. No alternative generation configuration was evaluated.
