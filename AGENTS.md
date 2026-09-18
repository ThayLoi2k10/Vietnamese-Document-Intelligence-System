# AGENTS.md

Guidance for coding agents working in this repository. Read this file before changing code, data, or documentation.

This is an **existing Vietnamese Legal NLP / RAG research project**. Preserve the current architecture and intent. Do not redesign the system, do not refactor unless the user explicitly asks, and do not invent project facts that are not in the repo.

---

## 1. Project identity and purpose

**Name (from project docs):** Vietnamese Legal Document Intelligence System.

**Primary task currently implemented:** article-level **information retrieval** over Vietnamese legal text. A user query is ranked against a flattened corpus of **59,636 articles**. The retrieval unit is the **article**, identified by `global_article_id` in the continuous range `[0, 59635]`.

**Longer-term intent (documented, not all implemented in code):** RAG / QA over legal documents. Docs describe later phases such as dense retrieval (Phase 6), chunking (Phase 7), RAG evaluation (Phase 10), and hybrid retrieval (Phase 11). Those phases are roadmap and evaluation design, not present as library modules under `src/` today.

**Three data streams:**

| Stream | Source | Current role |
|---|---|---|
| Legal retrieval | Local retrieval source under `data/raw/legal_retrieval/` | Production retrieval corpus and IR benchmark (Phase 5) |
| Legal QA | Hugging Face `thangvip/vietnamese-legal-qa` | Preprocessed QA pairs and QA benchmark splits for later RAG/QA eval |
| Legal documents | Hugging Face `th1nhng0/vietnamese-legal-documents` | Full-text documents, metadata, citation relationships |

Ground-truth field `relevant_laws` in raw retrieval JSON maps to **flattened article IDs**, not parent law-document IDs.

**Current official lexical baseline (Phase 5, hold-out test, 438 queries):** BM25 tuned with `k1 = 1.80`, `b = 0.85`. Documented reference metrics include nDCG@10 = 0.4866, Recall@10 = 0.6671, MRR@10 = 0.4610, Recall@100 = 0.8727, latency ≈ 3.26 ms/query. TF-IDF is the earlier keyword baseline, not the official one.

There is **no serving app**, API, or RAG chain in this repo. Runtime is Python library code plus one-shot scripts.

---

## 2. Current architecture and major directories

Architecture is **thin library + phase scripts**, not a packaged web service.

```text
PROJECT 2/
├── src/                      # Reusable retrievers and IR metrics
│   ├── retrieval/            # BaseRetriever, TfidfRetriever, BM25Retriever
│   └── evaluation/           # Precision/Recall/MRR/MAP/nDCG
├── scripts/
│   ├── data_acquisition/     # Hugging Face downloads
│   ├── processing/           # Flatten + Unicode/whitespace preprocess
│   ├── validation/           # Schema, ID, split integrity gates
│   ├── benchmarks/           # Build retrieval/QA evaluation splits
│   ├── retrieval/            # Fit/save keyword indices; tune BM25
│   ├── evaluation/           # Hold-out IR eval and error analysis
│   └── archive/              # One-off dataset audits; not the production path
├── tests/                    # Unit tests for retrieval metrics
├── docs/                     # Phase specs, schemas, experimental reports
├── data/                     # Raw, processed, benchmarks, indices
├── results/benchmarks/       # Metrics and prediction JSON artifacts
├── requirements.txt
└── AGENTS.md
```

**Library contracts (`src/`):**

- `src/retrieval/base.py` — `RetrievalResult(article_id: int, score: float, rank: int)` (rank is 1-indexed) and abstract `BaseRetriever` with `fit`, `search`, `batch_search`, `save`, `load`. Corpus ID column: `global_article_id` or `article_id`. Text column: `text`, `article_content`, or `content`.
- `src/retrieval/tfidf_retriever.py` — sklearn unigram TF-IDF, L2-normalized, `sublinear_tf=True`, cosine via sparse matmul. Serialized with joblib (`data/indices/tfidf_index.joblib`).
- `src/retrieval/bm25_retriever.py` — Okapi BM25 inverted index. Tokenizer: `re.findall(r"\w+", text.lower())` (`default_tokenizer`). Serialized with pickle (`data/indices/bm25_index.pkl`). The default index fit uses `k1=1.5`, `b=0.75`; after `load`, `k1` / `b` may be changed for tuning without rebuilding postings.
- `src/evaluation/retrieval_metrics.py` — per-query metrics then macro-average. Binary relevance. Duplicate retrieved IDs are de-duplicated. Metric key names used by eval scripts: `P@{k}`, `R@{k}` for k in `{1,5,10,20,50,100}`, plus `MRR@10`, `MAP@10`, `MAP@100`, `nDCG@10`, `nDCG@100`.

**Script convention:** scripts set `PROJECT_ROOT = Path(__file__).resolve().parents[2]`, may insert it on `sys.path` to import `src`, use hard-coded default paths under `data/` and `results/`, often skip work if outputs already exist unless `--force`, and often write via temp file + atomic replace.

**Dependencies listed in `requirements.txt`:** pandas, pyarrow, fastparquet, beautifulsoup4, datasets, scikit-learn.

---

## 3. Important data directories and their roles

```text
data/
├── raw/
│   ├── legal_documents/     # metadata, content shards, relationships
│   ├── legal_qa/            # vietnamese_legal_qa.parquet
│   └── legal_retrieval/     # legal_corpus.json, train.json, public_test.json, private_test.json
├── processed/
│   ├── legal_documents/     # preprocessed documents + validation_report.txt
│   ├── legal_qa/            # legal_qa_preprocessed.parquet + validation_report.txt
│   └── legal_retrieval/     # corpus_articles.parquet (staging), corpus_articles_preprocessed.parquet (production)
├── benchmarks/              # Frozen evaluation splits + validation reports (when present)
└── indices/                 # Fitted retriever indices (generated)
```

`.gitignore` excludes `data/raw/`, `data/processed/`, and `data/indices/`. `data/benchmarks/` and `results/` are not assumed to be gitignored; check the repository state before treating an artifact as ignored.

**Retrieval corpus production schema** in `corpus_articles_preprocessed.parquet` is the schema enforced by the preprocessing/validation code:
`global_article_id`, `doc_id`, `law_id`, `aid`, `text`, `text_raw`, `text_preprocessed`, `char_count_preprocessed`, `word_count_preprocessed`, `is_empty_preprocessed`.
`text` and `text_preprocessed` currently contain the same preprocessed article text. Exactly **8** articles are empty/whitespace and are flagged (`is_empty_preprocessed = True`), not removed from the corpus.

**Retrieval benchmark JSON** (`data/benchmarks/retrieval_{train,test}.json`):

- `query_id` (int)
- `query_text` (NFC-normalized string)
- `relevant_article_ids` (sorted unique ints in `[0, 59635]`)

Documented split: **1,752 train / 438 test** queries (80/20, stratified on single- vs multi-label, `random_state=42`). About 25% of queries are multi-label. `public_test.json` and `private_test.json` have empty `relevant_laws` and are **not** used for automatic IR metrics. The QA benchmark is currently separate from automatic retriever evaluation; do not assume QA pairs are fed into the retrieval evaluator.

`law_id` correspondence with the `legal_documents` stream is not a guaranteed one-to-one join; do not casually join the streams by `law_id` without checking the relevant schema/relationship logic.

**QA benchmark Parquet** (produced by `scripts/benchmarks/build_qa_benchmark.py`): `data/benchmarks/qa_{train,val,test}.parquet`. Documented sizes: 20,403 / 4,374 / 4,368 pairs from 29,145 total. Split is stratified group split on `content_hash` (MD5 of context) to avoid context leakage across `doc_id`s that share article text.

**Current working-tree caveats:** `data/benchmarks/` currently contains the QA validation report; do not assume the `qa_train/qa_val/qa_test` Parquet files exist unless the repository is inspected. `data/indices/` may be empty in a fresh/clean working tree even though index paths are documented.

**Results:** `results/benchmarks/` holds JSON metrics/predictions (TF-IDF, BM25 default, BM25 tuned, tuning report, win/loss analysis). These are experimental artifacts, not source of truth for IDs.

---

## 4. Current preprocessing / evaluation status

Documented as completed through **Phase 5**:

| Stage | Status in repo |
|---|---|
| Dataset understanding / audit | Docs in `docs/dataset_understanding.md`; exploratory scripts in `scripts/archive/` |
| Unicode NFC + structural whitespace preprocess | `scripts/processing/` + `docs/preprocessing/` |
| Flatten `legal_corpus.json` → article table | `scripts/processing/build_corpus_article.py` |
| Retrieval + QA benchmark construction and validation | `scripts/benchmarks/`, `scripts/validation/`, `docs/evaluation/phase4_evaluation_benchmark.md`, `docs/benchmarks/evaluation_dataset.md` |
| Keyword baselines TF-IDF and BM25 | `src/retrieval/`, `scripts/retrieval/`, `scripts/evaluation/`, `docs/phase5_keyword_baselines.md` |
| Dense Bi-Encoder baseline (Phase 6) | `src/retrieval/dense/`, `scripts/retrieval/`, `scripts/evaluation/`, `docs/evaluation/dense_baseline_report.md` |
| Document Chunking (Phase 7) | `src/retrieval/chunking/`, `scripts/retrieval/`, `docs/evaluation/chunking_experiment_report.md` |
| Vector Database Indexing (Phase 8) | `src/retrieval/indexing/`, `scripts/indexing/`, `results/benchmarks/vector_store_final_report.md` |

**Not implemented in `src/`:** RAG generation, hybrid retrieval (Phase 11), a CI workflow (Phase 23 is mentioned in preprocessing docs as a future gate, not present as GitHub Actions in the tree inspected for this file).

Keyword eval protocol: fit the index on the **full article corpus**; **tune only on train**; **report only on the frozen test**. Default evaluation `top_k = 100`. BM25 tuning evaluates `nDCG@10`, `R@10`, and `MAP@10`; the selected configuration is determined by **train nDCG@10**. The implemented `b` grid is exactly `{0.40, 0.55, 0.70, 0.75, 0.85}`. The BM25 tuning script evaluates candidate `k1` values from its configured grid; do not infer a grid from this document if the script is changed.

Documented lexical failure mode used to justify Phase 6: **99 / 438** test queries have R@10 = 0 for both TF-IDF and BM25 (vocabulary mismatch / paraphrase).

---

## 5. Important architectural constraints

1. **`global_article_id` is the system-wide retrieval key.** It is assigned by sequential flattening of `legal_corpus.json` (document order, then article order). Reordering, deleting rows, or compacting empty articles **invalidates all retrieval labels**.
2. **Do not change the corpus ID space by dropping or reordering articles.** The 8 empty articles are flagged with `is_empty_preprocessed`. Current TF-IDF/BM25 index builders do not filter them out, so do not document or assume that they are excluded from the fitted keyword indices.
3. **New retrievers must subclass `BaseRetriever` and return `list[RetrievalResult]`.** Evaluation scripts consume `article_id` in rank order.
4. **Do not rename metric keys** (`MAP@10` is used, not `AP@10`). Existing result JSON and comparison scripts depend on those strings.
5. **Hold-out test is frozen.** Do not regenerate `retrieval_test.json` or change the split seed as part of ordinary work.
6. **Hyperparameters and architecture choices must not be selected on the test set.**
7. **Preprocess preserves `\n` and `\t`.** Do not collapse all whitespace with `re.sub(r'\s+', ' ', text)` on full documents. Horizontal whitespace is collapsed per line only. NFC is mandatory. NBSP → space. Newlines normalized to `\n`.
8. **Do not strip legal numbering punctuation casually** (e.g. `Điều 1.`, `Khoản 2.`, `24/2010/NĐ-CP`). Docs treat these as structural signals for later chunking.
9. **BM25 tokenization is regex `\w+` lowercasing**, not Vietnamese word segmentation. Changing it changes the official lexical baseline.
10. **`scripts/archive/` is not a stable API.** Do not import it from production scripts.
11. Later retrieval methods (dense, hybrid) are specified to search the **same 59,636 articles** and the same ID space.

---

## 6. Coding workflow an agent must follow

1. **Inspect first.** Read this file, the relevant `docs/` note, and the target script/module plus its callers before editing.
2. **Match existing patterns:** `PROJECT_ROOT` path handling, `--force` idempotency, atomic writes, `sys.exit` codes on validators, Vietnamese docstrings where the surrounding file already uses them.
3. **Prefer adding** a new `BaseRetriever` implementation and a parallel index/eval script over changing TF-IDF/BM25 behavior.
4. **Keep changes small and verifiable.** One concern per change when possible.
5. **Do not drive-by refactor** unrelated files, rename for taste, or “clean up” archive scripts.
6. After code changes, **run the relevant tests or validation scripts** (see §9).
7. If behavior, schema, metrics, or phase status change, **update the matching doc** (see §10).
8. Do not add a web server, new package layout, or new retrieval unit (chunk/document) unless the user explicitly requests that phase of work.

---

## 7. Rules for modifying existing code

- Do **not** redesign architecture or move the library/script split.
- Do **not** refactor, reformat, or rename unless explicitly requested.
- **Preserve interfaces:** `BaseRetriever` methods, `RetrievalResult` fields, metric function signatures, JSON/Parquet column names, CLI flags already used by scripts.
- **Before changing a file, inspect purpose and dependencies** (who imports it, which paths it reads/writes, which docs describe it).
- Changing `src/evaluation/retrieval_metrics.py` affects all eval scripts and `tests/test_retrieval_metrics.py`. Changing flattening order in `build_corpus_article.py` affects the entire IR stack.
- If a task asks for both a new method and a change to an old baseline, implement the new method **beside** the old one unless the user asks to replace the baseline.
- Do not “improve” BM25/TF-IDF defaults in a way that silently invalidates published Phase 5 numbers without an explicit request and a doc update.

---

## 8. Rules for datasets, benchmarks, indices, and generated artifacts

**Do not modify** unless the user explicitly requests it:

- Raw files under `data/raw/`
- Processed Parquet under `data/processed/`
- Benchmark splits under `data/benchmarks/` (`retrieval_*.json`, `qa_*.parquet`, validation reports)
- Indices under `data/indices/`
- Existing files under `results/benchmarks/` that represent completed experiments

**Regeneration:**

- Many scripts no-op if outputs exist. Use `--force` only when the user wants a rebuild.
- Rebuilding the retrieval split or flattening corpus is a **breaking** experiment, not a routine fix.
- Rebuilding indices is allowed when implementing a requested retriever change; do not overwrite the official BM25/TF-IDF artifacts unless asked. Prefer a new filename if adding a new model.
- Do not commit secrets. Do not treat Hugging Face downloads as code.
- `data/raw/`, `data/processed/`, and `data/indices/` are gitignored; do not assume they exist in a fresh clone.

---

## 9. Testing and validation expectations

**Execution context:** run repository commands from the project root (`PROJECT 2`). Python version is not pinned by the current repository configuration. Depending on invocation/environment, `PYTHONPATH` or module-style execution may be needed for `src` imports; follow the existing script invocation patterns rather than inventing a new environment contract.


**Unit tests today:** `tests/test_retrieval_metrics.py` (unittest) covering P@k, R@k, MRR, AP/MAP, nDCG, aggregation, and edge cases.

After changing metrics: run that test module.

After changing processing or benchmarks (only if explicitly requested): run the matching validator in `scripts/validation/`:

- `validate_flattened_corpus.py` / `validate_preprocessed_corpus.py`
- `validate_legal_qa.py` / `validate_legal_documents.py`
- `validate_retrieval_benchmark.py` / `validate_qa_benchmark.py`

Validators are designed to `sys.exit(0)` on pass and `sys.exit(1)` on fail.

After changing a retriever: run the corresponding `scripts/retrieval/build_*` (if index contract changed) and `scripts/evaluation/evaluate_*` **on the existing frozen test JSON**, or a smaller documented smoke path if the script provides one. Index build scripts include a small sanity query after fit.

There is **no** automated e2e test for full-corpus retrieval in `tests/`. Do not claim CI exists unless you find a workflow file.

If browser/UI work appears in a future phase, verify in the browser; this repo currently has no UI.

---

## 10. Documentation expectations

`docs/` is the phase source of truth. Update the **specific** document that already owns the topic; do not create a parallel spec unless asked.

| Topic | Document |
|---|---|
| Raw dataset audit | `docs/dataset_understanding.md` |
| Shared preprocess rules | `docs/preprocessing/dataset_preprocessing.md` |
| Per-dataset preprocess | `docs/preprocessing/legal_*_preprocessing.md` |
| IR/QA split design | `docs/benchmarks/evaluation_dataset.md`, `docs/evaluation/phase4_evaluation_benchmark.md` |
| IR theory for this phase | `docs/retrieval/ir_fundamentals.md` |
| Keyword baseline numbers and official BM25 | `docs/phase5_keyword_baselines.md` |
| TF-IDF vs BM25 comparison | `docs/evaluation/keyword_baseline_comparison.md` |
| TF-IDF error analysis | `docs/evaluation/tfidf_error_analysis.md` |
| Dense baseline numbers and BGE-M3 | `docs/evaluation/dense_baseline_report.md` |
| Dense retrieval specification | `docs/phase6_dense_retrieval_specification.md` |
| Document Chunking specification | `docs/phase7_chunking_specification.md` |
| Document Chunking experiment report | `docs/evaluation/chunking_experiment_report.md` |
| Vector Database Indexing specification | `docs/phase8_vector_database_indexing_specification.md` |
| Vector Database final report | `results/benchmarks/vector_store_final_report.md`, `docs/evaluation/vector_store_final_report.md` |

When a change alters behavior, metrics, schema, IDs, or phase status: update the corresponding doc in the same task. Do not rewrite history of Phase 5 numbers; add a new section or new report if a new experiment supersedes them.

Keep documentation factual and consistent with code. Do not document unimplemented phases as if they were shipped.

---

## Quick reference for the next retrieval phase

If adding hybrid retrieval (the documented Phase 11 direction):

- Same corpus: `data/processed/legal_retrieval/corpus_articles_preprocessed.parquet`
- Same IDs: `global_article_id`
- Same frozen test: `data/benchmarks/retrieval_test.json`
- Same metrics: `evaluate_query_retrieval` / `aggregate_retrieval_metrics`
- Same interface: subclass `BaseRetriever`
- Combine lexical baseline (BM25 tuned: nDCG@10 = 0.4866) and dense baseline (BGE-M3: nDCG@10 = 0.6277) via RRF or linear score normalization
