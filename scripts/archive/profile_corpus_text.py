import os
import re
import unicodedata
import pandas as pd
import numpy as np


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


def profile_corpus_text():
    # 1. Đọc dữ liệu
    corpus_file = resolve_path("data/processed/legal_retrieval/corpus_articles.parquet")
    print("=" * 80)
    print("STEP 4: PROFILE CORPUS TEXT QUALITY (CORRECTED)")
    print("=" * 80)
    print(f"Đang đọc: {corpus_file}\n")

    df = pd.read_parquet(corpus_file)
    total_records = len(df)

    # Chuyển đổi text về chuỗi an toàn
    raw_texts = df["text"].fillna("").astype(str)

    # ==========================================
    # A. EMPTY / WHITESPACE AUDIT
    # ==========================================
    empty_exact = (raw_texts == "").sum()
    empty_strip = (raw_texts.str.strip() == "").sum()
    whitespace_only = empty_strip - empty_exact
    non_empty_mask = raw_texts.str.strip().ne("")

    # ==========================================
    # B. LENGTH STATISTICS (Character Level)
    # ==========================================
    char_lens = raw_texts.apply(len)
    clean_char_lens = raw_texts[non_empty_mask].apply(len)

    min_chars = char_lens.min()
    min_non_empty_chars = clean_char_lens.min() if not clean_char_lens.empty else 0
    mean_chars = char_lens.mean()
    median_chars = char_lens.median()
    max_chars = char_lens.max()

    # ==========================================
    # C. NEWLINE AUDIT
    # ==========================================
    newline_counts = raw_texts.apply(lambda t: t.count("\n"))
    articles_with_newline = (newline_counts > 0).sum()
    min_newline = newline_counts.min()
    max_newline = newline_counts.max()
    avg_newline = newline_counts.mean()

    # ==========================================
    # D. HTML PATTERNS
    # ==========================================
    html_tags = ["<html", "<p", "<div", "<br", "<table", "<script", "<style"]
    html_tag_stats = {}
    any_html_pattern = re.compile(r"<\s*(html|p|div|br|table|script|style)[^>]*>", re.IGNORECASE)
    has_any_html_count = raw_texts.apply(lambda t: bool(any_html_pattern.search(t))).sum()

    for tag in html_tags:
        pattern = re.compile(rf"{re.escape(tag)}[\s/>]", re.IGNORECASE)
        count = raw_texts.apply(lambda t: bool(pattern.search(t))).sum()
        html_tag_stats[tag] = count

    # ==========================================
    # E. UNICODE & ABNORMAL CHARACTERS
    # ==========================================
    # [FIXED]: Sửa lỗi "" in t
    has_replacement_char = raw_texts.apply(lambda t: "\ufffd" in t).sum()
    has_null_char = raw_texts.apply(lambda t: "\x00" in t).sum()
    has_bom = raw_texts.apply(lambda t: "\ufeff" in t).sum()
    has_nbsp = raw_texts.apply(lambda t: "\u00a0" in t).sum()
    
    # Ký tự điều khiển (ASCII Control chars trừ \t, \n, \r)
    control_char_pattern = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")
    has_control_chars = raw_texts.apply(lambda t: bool(control_char_pattern.search(t))).sum()

    # Chuẩn hóa Unicode NFC
    is_not_nfc = raw_texts.apply(lambda t: not unicodedata.is_normalized("NFC", t)).sum()

    # ==========================================
    # IN BÁO CÁO THỐNG KÊ
    # ==========================================
    print("--- A. EMPTY / WHITESPACE ---")
    print(f"Total articles:                 {total_records:,}")
    print(f"Exact empty (text == ''):       {empty_exact:,}")
    print(f"Whitespace-only (text != '' & strip == ''): {whitespace_only:,}")
    print(f"Total semantic empty (strip == ''):        {empty_strip:,}")

    print("\n--- B. CHARACTER LENGTH METRICS ---")
    print(f"Min character length (all):     {min_chars}")
    print(f"Min character length (non-empty): {min_non_empty_chars}")
    print(f"Median character length:        {median_chars:.1f}")
    print(f"Mean character length:          {mean_chars:.2f}")
    print(f"Max character length:           {max_chars:,}")

    print("\n--- C. NEWLINE METRICS ---")
    print(f"Articles with '\\n':             {articles_with_newline:,} ({articles_with_newline / total_records * 100:.2f}%)")
    print(f"Min newlines per article:       {min_newline}")
    print(f"Median newlines per article:    {newline_counts.median():.1f}")
    print(f"Mean newlines per article:      {avg_newline:.2f}")
    print(f"Max newlines per article:       {max_newline:,}")

    print("\n--- D. HTML PATTERNS DETECTED ---")
    print(f"Articles with any core HTML tag: {has_any_html_count:,}")
    for tag, count in html_tag_stats.items():
        print(f"  • Pattern '{tag}': {count:,} articles")

    print("\n--- E. UNICODE & ABNORMAL CHARACTERS ---")
    print(f"Replacement char (\\ufffd):       {has_replacement_char:,}")
    print(f"Null character (\\x00):           {has_null_char:,}")
    print(f"Byte Order Mark (\\ufeff):        {has_bom:,}")
    print(f"Control characters:              {has_control_chars:,}")
    print(f"Non-breaking space (\\u00a0):     {has_nbsp:,}")
    print(f"Not normalized NFC:              {is_not_nfc:,} ({is_not_nfc / total_records * 100:.2f}%)")

    # ==========================================
    # F. REPRESENTATIVE SAMPLES (10 ARTICLES)
    # ==========================================
    print("\n" + "=" * 80)
    print("F. REPRESENTATIVE SAMPLES (10 ARTICLES)")
    print("=" * 80)

    # 1. Empty sample (1) - Lấy từ danh sách strip == ""
    empty_indices = df[~non_empty_mask].index
    if len(empty_indices) > 0:
        emp_idx = empty_indices[0]
        row = df.loc[emp_idx]
        print(f"\n[SAMPLE 1 - SEMANTIC EMPTY] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
        print(f"Raw Content: {repr(row['text'])} (len={len(str(row['text']))})")

    # [FIXED]: Lấy tập non-empty thực sự cho các nhóm Short, Medium, Long
    df_non_empty = df[non_empty_mask].copy()
    df_non_empty["char_len"] = clean_char_lens

    # 2. Short samples (3) - Các Điều thực sự ngắn nhưng có nội dung
    print("\n--- 3 ARTICLE THỰC SỰ NGẮN ---")
    short_samples = df_non_empty.sort_values("char_len").head(3)
    for i, (_, row) in enumerate(short_samples.iterrows(), start=2):
        print(f"\n[SAMPLE {i} - SHORT] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']} | Chars: {row['char_len']}")
        print(f"Content:\n{row['text']}")

    # 3. Medium samples (3 - xấp xỉ median)
    print("\n--- 3 ARTICLE TRUNG BÌNH (XẤP XỈ MEDIAN) ---")
    med_diff = (df_non_empty["char_len"] - median_chars).abs()
    med_samples = df_non_empty.loc[med_diff.sort_values().head(3).index]
    for i, (_, row) in enumerate(med_samples.iterrows(), start=5):
        print(f"\n[SAMPLE {i} - MEDIUM] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']} | Chars: {row['char_len']}")
        snippet = row['text'][:300] + ("..." if row['char_len'] > 300 else "")
        print(f"Content (snippet):\n{snippet}")

    # 4. Long samples (3)
    print("\n--- 3 ARTICLE DÀI ---")
    long_samples = df_non_empty.sort_values("char_len", ascending=False).head(3)
    for i, (_, row) in enumerate(long_samples.iterrows(), start=8):
        print(f"\n[SAMPLE {i} - LONG] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']} | Chars: {row['char_len']}")
        snippet = row['text'][:300] + ("..." if row['char_len'] > 300 else "")
        print(f"Content (first 300 chars):\n{snippet}")


if __name__ == "__main__":
    profile_corpus_text()