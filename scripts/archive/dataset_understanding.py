import ast
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ==============================================================================
# TASK A: Kiểm tra ID relationship giữa content và metadata
# ==============================================================================
def task_a_id_relationship():
    print("=" * 70)
    print("TASK A: KIỂM TRA ID RELATIONSHIP (content.id vs metadata.id)")
    print("=" * 70)
    
    # 1. Đọc metadata IDs (đọc cột 'id')
    meta_path = "data/raw/legal_documents/metadata/data-00000-of-00001.parquet"
    meta_df = pd.read_parquet(meta_path, columns=["id"])
    meta_ids = set(meta_df["id"].astype(str))
    
    # 2. Đọc toàn bộ content IDs từ 12 shard parquet
    content_files = sorted(glob.glob("data/raw/legal_documents/content/*.parquet"))
    content_ids = set()
    for f in content_files:
        c_table = pq.read_table(f, columns=["id"])
        content_ids.update(c_table["id"].to_pylist())
    content_ids = set(str(x) for x in content_ids)
    
    intersection = content_ids.intersection(meta_ids)
    content_only = content_ids - meta_ids
    metadata_only = meta_ids - content_ids
    
    print(f"Tổng số Content IDs:    {len(content_ids):,}")
    print(f"Tổng số Metadata IDs:   {len(meta_ids):,}")
    print(f"Intersection (Giao):    {len(intersection):,}")
    print(f"Content-only (Thừa):    {len(content_only):,}")
    print(f"Metadata-only (Thiếu):  {len(metadata_only):,}")
    
    if metadata_only:
        sample_meta_only = list(metadata_only)[:5]
        print(f"Mẫu ID có trong metadata nhưng không có content: {sample_meta_only}")


# ==============================================================================
# TASK B: Inspect legal_corpus.json
# ==============================================================================
def task_b_inspect_legal_corpus():
    print("\n" + "=" * 70)
    print("TASK B: THỐNG KÊ CHI TIẾT legal_retrieval/legal_corpus.json")
    print("=" * 70)
    
    corpus_path = "data/raw/legal_retrieval/legal_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        
    num_laws = len(corpus)
    law_ids = []
    doc_ids = []
    articles_per_law = []
    empty_articles_count = 0
    non_unique_aid_cases = 0
    
    total_articles = 0
    
    for doc in corpus:
        d_id = doc.get("id")
        l_id = doc.get("law_id")
        doc_ids.append(d_id)
        law_ids.append(l_id)
        
        contents = doc.get("content", [])
        articles_per_law.append(len(contents))
        total_articles += len(contents)
        
        # Kiểm tra aid uniqueness trong từng văn bản
        aids = [item.get("aid") for item in contents]
        if len(aids) != len(set(aids)):
            non_unique_aid_cases += 1
            
        # Kiểm tra article rỗng
        for item in contents:
            txt = item.get("content_Article", "").strip()
            if not txt:
                empty_articles_count += 1

    print(f"Số lượng văn bản luật (Laws):        {num_laws:,}")
    print(f"Tổng số lượng Điều (Articles):       {total_articles:,}")
    print(f"Số Điều/Law - Min:                   {np.min(articles_per_law)}")
    print(f"Số Điều/Law - Max:                   {np.max(articles_per_law)}")
    print(f"Số Điều/Law - Median (p50):          {np.median(articles_per_law)}")
    print(f"Số Điều/Law - Trung bình:            {np.mean(articles_per_law):.2f}")
    
    print("\n--- Kiểm tra ràng buộc duy nhất (Uniqueness) & Rỗng ---")
    print(f"- Trường 'id' có unique không?        {len(doc_ids) == len(set(doc_ids))} (Unique: {len(set(doc_ids)):,})")
    print(f"- Trường 'law_id' có unique không?    {len(law_ids) == len(set(law_ids))} (Unique: {len(set(law_ids)):,})")
    print(f"- 'aid' có trùng trong cùng 1 Law?   {'CÓ LỖI' if non_unique_aid_cases > 0 else 'Hoàn toàn Unique'}")
    print(f"- Số lượng Điều bị rỗng text:        {empty_articles_count}")


# ==============================================================================
# TASK C: Inspect generated_qa_pairs & Schema verification
# ==============================================================================
def task_c_inspect_qa_pairs():
    print("\n" + "=" * 70)
    print("TASK C: INSPECT legal_qa/vietnamese_legal_qa.parquet")
    print("=" * 70)
    
    qa_path = "data/raw/legal_qa/vietnamese_legal_qa.parquet"
    df = pd.read_parquet(qa_path)
    total_docs = len(df)
    
    total_qa = 0
    qa_per_article = []
    q_types = {}
    difficulties = {}
    
    is_standard_json_count = 0
    is_python_literal_count = 0
    parse_error_count = 0
    
    for val in df["generated_qa_pairs"]:
        pairs = []
        if isinstance(val, list):
            pairs = val
        elif isinstance(val, str):
            # Kiểm tra chuẩn JSON
            try:
                pairs = json.loads(val)
                is_standard_json_count += 1
            except Exception:
                # Kiểm tra Python literal (Single quotes / Dict representation)
                try:
                    pairs = ast.literal_eval(val)
                    is_python_literal_count += 1
                except Exception:
                    parse_error_count += 1
                    continue
        
        qa_count = len(pairs)
        total_qa += qa_count
        qa_per_article.append(qa_count)
        
        for p in pairs:
            qt = p.get("question_type", "unknown")
            diff = p.get("difficulty", "unknown")
            q_types[qt] = q_types.get(qt, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1

    print(f"Tổng số tài liệu/Điều luật trong QA:  {total_docs:,}")
    print(f"Tổng số cặp QA pairs:                 {total_qa:,}")
    print(f"Trung bình QA / Điều luật:            {np.mean(qa_per_article):.2f}")
    
    print("\n--- Phân phối Question Type ---")
    for k, v in q_types.items():
        print(f" - {k:<15}: {v:,} ({v/total_qa*100:.2f}%)")
        
    print("\n--- Phân phối Difficulty ---")
    for k, v in difficulties.items():
        print(f" - {k:<15}: {v:,} ({v/total_qa*100:.2f}%)")
        
    print("\n--- Kiểm tra Schema Parsing ---")
    print(f" - Parsed bằng standard json.loads:   {is_standard_json_count:,}")
    print(f" - Parsed bằng ast.literal_eval:     {is_python_literal_count:,}")
    print(f" - Parse lỗi:                         {parse_error_count:,}")


# ==============================================================================
# TASK D: Kiểm tra mapping ID giữa Retrieval và Legal Documents
# ==============================================================================
def task_d_cross_dataset_matching():
    print("\n" + "=" * 70)
    print("TASK D: KIỂM TRA MATCHING GIỮA legal_retrieval VÀ legal_documents")
    print("=" * 70)
    
    # 1. Lấy tập law_id từ legal_corpus.json
    corpus_path = "data/raw/legal_retrieval/legal_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    retrieval_law_ids = set(str(doc.get("law_id")).strip() for doc in corpus if doc.get("law_id"))
    
    # 2. Lấy so_ky_hieu từ legal_documents.metadata
    meta_path = "data/raw/legal_documents/metadata/data-00000-of-00001.parquet"
    meta_df = pd.read_parquet(meta_path, columns=["id", "so_ky_hieu", "title"])
    
    # Chuẩn hóa loại bỏ khoảng trắng thừa
    meta_so_ky_hieu = set(meta_df["so_ky_hieu"].dropna().astype(str).str.strip())
    
    direct_matches = retrieval_law_ids.intersection(meta_so_ky_hieu)
    
    print(f"Số lượng unique law_id trong legal_retrieval:        {len(retrieval_law_ids):,}")
    print(f"Số lượng unique so_ky_hieu trong legal_documents:    {len(meta_so_ky_hieu):,}")
    print(f"Khớp chính xác (Exact string match):                 {len(direct_matches):,} / {len(retrieval_law_ids):,} ({len(direct_matches)/len(retrieval_law_ids)*100:.2f}%)")
    
    # Lấy thử 5 ví dụ match thành công
    if direct_matches:
        sample_matched = list(direct_matches)[:5]
        print(f"\nMẫu 5 mã văn bản match thành công giữa 2 dataset:")
        for sm in sample_matched:
            matched_row = meta_df[meta_df["so_ky_hieu"] == sm].iloc[0]
            print(f" - [law_id: {sm}] -> [doc_id: {matched_row['id']}] | Title: {matched_row['title'][:70]}...")

if __name__ == "__main__":
    task_a_id_relationship()
    task_b_inspect_legal_corpus()
    task_c_inspect_qa_pairs()
    task_d_cross_dataset_matching()