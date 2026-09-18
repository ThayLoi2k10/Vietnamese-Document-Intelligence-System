# Dataset Understanding & Audit

> **Phase:** Dataset Understanding  
> **Project:** Vietnamese Legal Document Intelligence System  
> **Status:** Completed  
> **Purpose:** Understand the structure, relationships, schema, and characteristics of raw datasets before preprocessing.

---

## 1. Objective

This phase focuses on understanding the raw datasets before data cleaning, transformation, and preprocessing decisions are implemented.

Key investigative targets:
1. Identify available datasets, formats, and physical layouts.
2. Inspect schemas, field representations, and value distributions.
3. Map internal relationships: documents, metadata, articles, questions, answers, and citation links.
4. Establish cross-dataset linkability.
5. Identify structural anomalies, missing values, empty entries, and formatting inconsistencies.
6. Formulate informed preprocessing and pipeline specifications based on empirical evidence.

The boundary of this document is **dataset understanding and audit**, strictly preserving raw state observations without altering source artifacts.

---

## 2. Raw Dataset Structure

The raw and benchmark data are organized as follows:

```text
data/
|-- benchmarks/
|   |-- qa_train.parquet
|   |-- qa_val.parquet
|   |-- qa_test.parquet
|   |-- retrieval_train.json
|   `-- retrieval_test.json
|
|-- processed/
|   |-- legal_documents/
|   |-- legal_qa/
|   `-- legal_retrieval/
|
`-- raw/
    |-- legal_documents/
    |   |-- content/
    |   |   |-- data-00000-of-00012.parquet
    |   |   |-- ...
    |   |   `-- data-00011-of-00012.parquet
    |   |
    |   |-- metadata/
    |   |   `-- data-00000-of-00001.parquet
    |   |
    |   `-- relationships/
    |       `-- data-00000-of-00001.parquet
    |
    |-- legal_qa/
    |   `-- vietnamese_legal_qa.parquet
    |
    `-- legal_retrieval/
        |-- legal_corpus.json
        |-- train.json
        |-- public_test.json
        `-- private_test.json
```

## 3. Dataset Groups Overview

| **Dataset** | **Primary Scope** | **Format** | **Core Granularity** |
|---|---|---|---|
| `legal_documents` | Full text legal documents, metadata, and citation relationships | Parquet | Document-level |
| `legal_qa` | Question-answer pairs derived from legal articles | Parquet | QA Pair-level (flattened from context) |
| `legal_retrieval` | Article-level corpus and retrieval ground truth benchmarks | JSON | Article-level |

## 4. `legal_documents` Inspection

### 4.1 Content Component

**Path:** `data/raw/legal_documents/content/` (12 shards)

Each record represents a complete legal document preserved in raw HTML:

| **Column** | **Type** | **Description** |
|---|---|---|
| `id` | string | Unique document identifier |
| `content_html` | string | Full legal document encoded in HTML DOM |

*Statistics:*

- **Total records:** 170,824
- **DOM elements present:** `<!DOCTYPE html>`, `<html>`, `<head>`, `<style>`, `<body>`, `<div>`, `<table>`, `<p>`.
- **Structural challenge:** Documents contain CSS stylesheets inside `<style>` blocks and layout boilerplate that must be stripped without corrupting textual content.

### 4.2 Metadata Component

**Path:** `data/raw/legal_documents/metadata/` (1 shard)

Contains administrative and legal lifecycle metadata across 17 columns:

- Core attributes: `id`, `title`, `so_ky_hieu`, `ngay_ban_hanh`, `loai_van_ban`, `co_quan_ban_hanh`, `nguoi_ky`, `tinh_trang_hieu_luc`.
- Secondary attributes: `ngay_co_hieu_luc`, `ngay_het_hieu_luc`, `nguon_thu_thap`, `ngay_dang_cong_bao`, `nganh`, `linh_vuc`, `chuc_danh`, `pham_vi`, `thong_tin_ap_dung`.

*Statistics:*

- **Total records:** 171,556
- **Key finding:** 171,556 metadata entries vs. 170,824 content entries (a difference of 732 records).

### 4.3 Relationships Component

**Path:** `data/raw/legal_documents/relationships/` (1 shard)

Represents directed edges in the Vietnamese legal citation network:

| **Column** | **Type** | **Description** |
|---|---|---|
| `doc_id` | string | Source document identifier |
| `other_doc_id` | string | Target document identifier |
| `relationship` | string | Legal citation/dependency relationship type |

*Statistics:*

- **Total directed edges:** 1,033,255
- **Common relationship labels:** `Căn cứ` (668,173), `Dẫn chiếu` (72,554), `hết hiệu lực` (50,939), and `chi tiết/hướng dẫn` (39,419).

## 5. `legal_qa` Inspection

**Path:** `data/raw/legal_qa/vietnamese_legal_qa.parquet`

Raw schema:

| **Column** | **Raw Type** | **Description** |
|---|---|---|
| `doc_name` | string | Source legal document name |
| `doc_type_name` | string | Type classification of source document |
| `article_content` | string | Legal article excerpt used as reference context |
| `generated_qa_pairs` | object (native list[dict]) | Array of question-answer objects |
| `generation_time` | float | Generation process duration |

*Audit Discovery:*

- The `generated_qa_pairs` column is loaded directly by PyArrow as native nested Python lists of dictionaries.
- **Total document contexts:** 9,715 records.
- **Array size:** Exactly 3 QA pairs per record across all entries.
- **True QA volume:** $9,715 \times 3 = 29,145$ question-answer pairs.
- **Attributes per pair:** `question`, `answer`, `difficulty`, `question_type`.

## 6. `legal_retrieval` Inspection

**Path:** `data/raw/legal_retrieval/`

Comprises `legal_corpus.json`, `train.json`, `public_test.json`, and `private_test.json`.

### 6.1 `legal_corpus.json`

Represents the core knowledge base structured hierarchically by source legal acts and constituent articles:

```json
{
  "id": 0,
  "law_id": "14/2022/TT-NHNN",
  "content": [
    {
      "aid": 0,
      "content_Article": "..."
    }
  ]
}
```

*Structural Distribution:*

- **Source law acts (`law_id`):** 2,157 documents.
- **Total constituent articles (`aid`):** 59,636 articles.
- **Article count per law:** Min: 0, Median: 17, Mean: 27.65, Max: 689.
- **Retrieval granularity definition:** The retrieval corpus contains **59,636 article-level units**, each assigned a continuous `global_article_id` from `0` to `59,635`.

### 6.2 Ground-Truth Benchmarks (`train.json`)

Record schema:

```json
{
  "qid": 933,
  "question": "Thưa luật sư tôi có đăng ký kết hôn ...",
  "relevant_laws": [53877, 53875, 53929]
}
```

*Critical Audit Insight on Target Granularity:*

- The targets listed in `relevant_laws` have values spanning the range $[0, 59635]$.
- Despite the field name `relevant_laws`, these integer pointers map 1-to-1 to the **59,636 flattened articles** from `legal_corpus.json`, not the 2,157 parent law document IDs.
- Total annotated training queries: 2,190 queries (split in Phase 4 into 1,752 train / 438 test queries).

## 7. Cross-Dataset Linkability & Empirical Audits

### Task A — Document ID Alignment (`content` vs. `metadata`)

- Content IDs: 170,824
- Metadata IDs: 171,556
- Intersection: 170,824 (100% of content IDs possess corresponding metadata).
- Metadata-only IDs: 732 records.
- *Raw observation:* The 732 metadata-only records have metadata entries but no corresponding records in the content partitions. Their exact underlying cause is not established by this audit and should not be conflated with content-empty records identified during preprocessing.

### Task B — Corpus Integrity Audit (`legal_corpus.json`)

- Uniqueness of internal law `id`: 100% unique (2,157 / 2,157).
- Uniqueness of `law_id`: 100% unique (2,157 / 2,157).
- Uniqueness of `aid` within each law: 100% locally unique.
- Empty content check: Exactly 8 articles contain empty strings or pure whitespace.

### Task C — QA Volume & Taxonomy Audit (`legal_qa`)

- All 9,715 records parse into exactly 3 QA pairs, yielding 29,145 total pairs.
- Raw tags in `difficulty` and `question_type` contain bilingual mixing (e.g., `easy`/`dễ`, `factual`/`thực tế`), requiring deterministic taxonomy canonicalization during preprocessing.

### Task D — Cross-Stream Alignment (`legal_retrieval` vs. `legal_documents`)

- Evaluated matching capability between retrieval corpus source documents (`law_id`, 2,157 items) and full document metadata (`so_ky_hieu`, 71,722 unique items).
- Exact string matches: 1,919 / 2,157 (88.97%).
- Unmatched `law_id` count: 238 (11.03%). The exact causes of these mismatches were not established by the current exact-string audit and may involve identifier formatting, normalization differences, document versions, or source inconsistencies. Further normalization experiments should be conducted before adopting a joining strategy.

## 8. Summary of Baseline Architecture Implications

1. **Retrieval Unit Definition:** The core search space for Phase 5 (Keyword Baselines), Phase 6 (Dense Retrieval), and Phase 11 (Hybrid Retrieval) is strictly defined over the **59,636 articles** derived from `legal_corpus.json`, preserving `global_article_id` continuity from `0` to `59,635`.
2. **Empty Record Isolation:** The 8 empty articles in the retrieval corpus must be isolated or flagged (`is_empty = True`) rather than silently dropped, preventing index shifting.
3. **Encoding Discipline:** Mixed encoding across raw text mandates systematic Unicode canonical normalization (NFC) across all ingestion scripts.
4. **Benchmark Integrity:** Ground truth retrieval evaluation relies on mapping natural queries against the global article range $[0, 59635]$.

## 9. Deliverables Transition Log

- Raw inspection scripts archived under `scripts/archive/` (`dataset_understanding.py`, `profile_corpus_text.py`, `validate_corpus_ids.py`).
- Preprocessing specifications formulated here serve as constraints for downstream processing pipelines:
  - `scripts/processing/preprocess_corpus_articles.py` -> `corpus_articles_preprocessed.parquet`
  - `scripts/processing/preprocess_legal_qa.py` -> `legal_qa_preprocessed.parquet`
  - `scripts/processing/preprocess_legal_documents.py` -> `legal_documents_preprocessed.parquet`
