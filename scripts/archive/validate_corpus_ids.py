import os
import json
import pandas as pd


def resolve_path(rel_path: str) -> str:
    """Tự động dò tìm đường dẫn từ thư mục hiện tại hoặc các cấp cha."""
    candidates = [
        rel_path,
        os.path.join("..", rel_path),
        os.path.join("..", "..", rel_path),
        os.path.join("..", "..", "..", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"Không tìm thấy file: {rel_path}")


def validate_corpus_ids():
    # 1. Nạp dữ liệu
    corpus_file = resolve_path("data/processed/legal_retrieval/corpus_articles.parquet")
    train_file = resolve_path("data/raw/legal_retrieval/train.json")

    print("=" * 80)
    print("STEP 3: ID INTEGRITY VALIDATION")
    print("=" * 80)
    print(f"- Corpus file: {corpus_file}")
    print(f"- Train file:  {train_file}\n")

    df_corpus = pd.read_parquet(corpus_file)
    with open(train_file, "r", encoding="utf-8") as f:
        train_queries = json.load(f)

    # 2. Kiểm tra aid trong corpus
    total_articles = len(df_corpus)
    unique_aid = df_corpus["aid"].nunique()
    duplicate_aid_count = total_articles - unique_aid

    # 3. So sánh aid == global_article_id
    aid_eq_global = df_corpus["aid"] == df_corpus["global_article_id"]
    true_count = int(aid_eq_global.sum())
    false_count = int((~aid_eq_global).sum())

    # 4. Kiểm tra đối soát Ground-Truth từ train.json
    all_gt_refs = []
    for q in train_queries:
        all_gt_refs.extend(q.get("relevant_laws", []))

    total_gt_refs = len(all_gt_refs)
    unique_gt_refs = len(set(all_gt_refs))

    # Đối soát với aid (tập giá trị aid hiện có)
    existing_aids = set(df_corpus["aid"].unique())
    found_by_aid = sum(1 for ref in all_gt_refs if ref in existing_aids)
    missing_by_aid = total_gt_refs - found_by_aid

    # Đối soát với global_article_id (tập giá trị 0 -> 59,635)
    existing_globals = set(df_corpus["global_article_id"].unique())
    found_by_global = sum(1 for ref in all_gt_refs if ref in existing_globals)
    missing_by_global = total_gt_refs - found_by_global

    # 5. In kết quả theo đúng định dạng yêu cầu
    print("--- 1. CORPUS AID INTEGRITY ---")
    print(f"Total articles:               {total_articles:,}")
    print(f"Unique aid:                   {unique_aid:,}")
    print(f"Duplicate aid count:          {duplicate_aid_count:,}")

    print("\n--- 2. AID VS GLOBAL_ARTICLE_ID COMPARISON ---")
    print("aid == global_article_id:")
    print(f"    True count:               {true_count:,} ({true_count / total_articles * 100:.2f}%)")
    print(f"    False count:              {false_count:,} ({false_count / total_articles * 100:.2f}%)")

    print("\n--- 3. GROUND-TRUTH REFERENCES (TRAIN.JSON) VALIDATION ---")
    print(f"Total ground-truth article references:        {total_gt_refs:,} (Unique: {unique_gt_refs:,})")
    print(f"Ground-truth references found by aid:          {found_by_aid:,}")
    print(f"Ground-truth references missing by aid:        {missing_by_aid:,}")
    print(f"Ground-truth references found by global_id:    {found_by_global:,}")
    print(f"Ground-truth references missing by global_id:  {missing_by_global:,}")

    # 6. Kiểm tra tính đa nghĩa nếu map bằng aid
    print("\n--- 4. AMBIGUITY CHECK IF MAPPED BY AID ---")
    max_docs_per_aid = df_corpus.groupby("aid")["doc_id"].nunique().max()
    print(f"Max documents sharing the same aid:            {max_docs_per_aid:,} văn bản")
    print(f"Số lượng Điều luật có aid=0:                   {(df_corpus['aid'] == 0).sum():,} Điều")


if __name__ == "__main__":
    validate_corpus_ids()