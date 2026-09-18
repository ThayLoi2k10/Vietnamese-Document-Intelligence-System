import os
import re
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


def classify_whitespace(ws: str) -> str:
    """Phân loại bản chất của chuỗi horizontal whitespace."""
    if "\t" in ws:
        return "TAB"
    if "\xa0" in ws:
        return "NBSP"
    if ws.isspace():
        return "SPACE"
    return "OTHER"


def classify_structural_context(text: str, match_start: int) -> str:
    """
    Kiểm tra whitespace marker có nằm trong cấu trúc xuống dòng
    hay không.
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    prefix = text[line_start:match_start]

    if prefix.strip() == "":
        return "LINE_INDENTATION"

    return "INLINE"


def print_space_dist(name: str, lengths: list, types: list, contexts: list = None):
    """In phân phối độ dài, bản chất whitespace và ngữ cảnh cấu trúc dòng."""
    if not lengths:
        print(f"• {name}: Không có dữ liệu.")
        return

    arr = np.array(lengths)

    print(f"• {name} (Tổng {len(arr):,} lần xuất hiện):")
    print(
        f"  ├─ Min: {arr.min()} | "
        f"Median: {np.median(arr):.1f} | "
        f"Mean: {arr.mean():.2f} | "
        f"Max: {arr.max()}"
    )
    print(
        f"  ├─ 2 - 4 characters: "
        f"{((arr >= 2) & (arr <= 4)).sum():,} "
        f"({((arr >= 2) & (arr <= 4)).sum()/len(arr)*100:.1f}%)"
    )
    print(
        f"  ├─ 5 - 9 characters: "
        f"{((arr >= 5) & (arr <= 9)).sum():,} "
        f"({((arr >= 5) & (arr <= 9)).sum()/len(arr)*100:.1f}%)"
    )
    print(
        f"  ├─ >= 10 characters: "
        f"{(arr >= 10).sum():,} "
        f"({(arr >= 10).sum()/len(arr)*100:.1f}%)"
    )

    type_counts = pd.Series(types).value_counts()
    has_context = contexts is not None and len(contexts) > 0
    type_branch = "  ├─" if has_context else "  └─"

    print(f"{type_branch} Whitespace type:")
    for ws_type, count in type_counts.items():
        print(
            f"      • {ws_type}: {count:,} "
            f"({count / len(types) * 100:.1f}%)"
        )

    if has_context:
        context_counts = pd.Series(contexts).value_counts()
        print("  └─ Structural context:")
        for context, count in context_counts.items():
            print(
                f"      • {context}: {count:,} "
                f"({count / len(contexts) * 100:.1f}%)"
            )


def analyze_structural_whitespace():
    corpus_file = resolve_path("data/processed/legal_retrieval/corpus_articles.parquet")
    print("=" * 80)
    print("EXPERIMENT 5.3: PHÂN BIỆT WHITESPACE ARTIFACT VÀ STRUCTURAL SIGNAL")
    print("=" * 80)
    print(f"Đang nạp dữ liệu: {corpus_file}\n")

    df = pd.read_parquet(corpus_file)
    raw_texts = df["text"].fillna("").astype(str)
    total_articles = len(df)

    # ==========================================
    # 1. ĐỊNH NGHĨA CÁC PATTERN KHẢO SÁT
    # ==========================================
    
    # Pattern 1: Dấu chấm phẩy ';' + horizontal whitespace + 'a)', 'b)', 'c)'
    re_semi_point_a = re.compile(r";([^\S\r\n]{2,})(a\))", re.IGNORECASE)
    re_semi_point_b = re.compile(r";([^\S\r\n]{2,})(b\))", re.IGNORECASE)
    re_semi_point_c = re.compile(r";([^\S\r\n]{2,})(c\))", re.IGNORECASE)
    re_semi_point_any = re.compile(r";([^\S\r\n]{2,})([a-zđ]\))", re.IGNORECASE)

    # Pattern 2: Dấu hai chấm ':' + horizontal whitespace + [a-zđ]\) hoặc 1.
    re_colon_marker = re.compile(r":([^\S\r\n]{2,})([a-zđ]\)|\d+\.)", re.IGNORECASE)

    # Pattern 3: Ký tự kết thúc nội dung + horizontal whitespace + Điểm a-zđ)
    # Mục đích: phát hiện marker pháp lý xuất hiện sau nội dung trên cùng dòng.
    re_inline_point_any = re.compile(
        r"([^\n\r\s])([^\S\r\n]{2,})([a-zđ]\))",
        re.IGNORECASE
    )

    # Pattern 4: Ký tự kết thúc nội dung + horizontal whitespace + numbered marker.
    # Đây là pattern exploratory, không mặc định coi marker là "Khoản".
    re_inline_clause_any = re.compile(
        r"([^\n\r\s])([^\S\r\n]{2,})(\d+\.\s)",
        re.IGNORECASE
    )

    # ==========================================
    # 2. THỐNG KÊ TẦN SUẤT XUẤT HIỆN
    # ==========================================
    count_semi_a = raw_texts.apply(lambda t: bool(re_semi_point_a.search(t))).sum()
    count_semi_b = raw_texts.apply(lambda t: bool(re_semi_point_b.search(t))).sum()
    count_semi_c = raw_texts.apply(lambda t: bool(re_semi_point_c.search(t))).sum()
    count_semi_any = raw_texts.apply(lambda t: bool(re_semi_point_any.search(t))).sum()
    count_colon_any = raw_texts.apply(lambda t: bool(re_colon_marker.search(t))).sum()
    count_inline_point = raw_texts.apply(lambda t: bool(re_inline_point_any.search(t))).sum()
    count_inline_clause = raw_texts.apply(lambda t: bool(re_inline_clause_any.search(t))).sum()

    print("--- 1. THỐNG KÊ SỐ LƯỢNG ARTICLES CHỨA INLINE STRUCTURAL WHITESPACE ---")
    print(f"• Tổng số Điều luật trong corpus:                                {total_articles:,}")
    print(f"• Có pattern [ ; + multi-whitespace + a) ]:                       {count_semi_a:,} articles ({count_semi_a / total_articles * 100:.2f}%)")
    print(f"• Có pattern [ ; + multi-whitespace + b) ]:                       {count_semi_b:,} articles ({count_semi_b / total_articles * 100:.2f}%)")
    print(f"• Có pattern [ ; + multi-whitespace + c) ]:                       {count_semi_c:,} articles ({count_semi_c / total_articles * 100:.2f}%)")
    print(f"• Có pattern [ ; + multi-whitespace + (bất kỳ Điểm a-zđ) ]:       {count_semi_any:,} articles ({count_semi_any / total_articles * 100:.2f}%)")
    print(f"• Có pattern [ : + multi-whitespace + (Điểm hoặc Khoản) ]:        {count_colon_any:,} articles ({count_colon_any / total_articles * 100:.2f}%)")
    print(f"• Có pattern [ text + multi-whitespace + [a-zđ]\\) ]:             {count_inline_point:,} articles ({count_inline_point / total_articles * 100:.2f}%)")
    print(f"• Có pattern [ text + multi-whitespace + \\d+\\.\\s ]:            {count_inline_clause:,} articles ({count_inline_clause / total_articles * 100:.2f}%)")

    # ==========================================
    # 3. PHÂN PHỐI SỐ LƯỢNG, BẢN CHẤT VÀ NGỮ CẢNH DÒNG
    # ==========================================
    whitespace_lengths_semi_any = []
    whitespace_types_semi_any = []
    whitespace_contexts_semi_any = []
    for text in raw_texts:
        for match in re_semi_point_any.finditer(text):
            ws = match.group(1)
            whitespace_lengths_semi_any.append(len(ws))
            whitespace_types_semi_any.append(classify_whitespace(ws))
            whitespace_contexts_semi_any.append(
                classify_structural_context(text, match.start())
            )

    whitespace_lengths_inline_point = []
    whitespace_types_inline_point = []
    for text in raw_texts:
        for match in re_inline_point_any.finditer(text):
            ws = match.group(2)
            whitespace_lengths_inline_point.append(len(ws))
            whitespace_types_inline_point.append(classify_whitespace(ws))

    whitespace_lengths_inline_clause = []
    whitespace_types_inline_clause = []
    for text in raw_texts:
        for match in re_inline_clause_any.finditer(text):
            ws = match.group(2)
            whitespace_lengths_inline_clause.append(len(ws))
            whitespace_types_inline_clause.append(classify_whitespace(ws))

    print("\n--- 2. PHÂN PHỐI ĐỘ DÀI VÀ LOẠI HORIZONTAL WHITESPACE TRƯỚC LEGAL MARKERS ---")
    print_space_dist(
        "Horizontal whitespace trong [ ; + whitespace + Điểm a-z ]",
        whitespace_lengths_semi_any,
        whitespace_types_semi_any,
        whitespace_contexts_semi_any,
    )
    print_space_dist(
        "Horizontal whitespace trong [ Text + whitespace + Điểm a-z ]",
        whitespace_lengths_inline_point,
        whitespace_types_inline_point,
    )
    print_space_dist(
        "Horizontal whitespace trong [ Text + whitespace + Khoản 1., 2. ]",
        whitespace_lengths_inline_clause,
        whitespace_types_inline_clause,
    )

    # ==========================================
    # 4. TRÍCH XUẤT 15 SAMPLES ĐỘC LẬP (DISJOINT DOCS)
    # ==========================================
    print("\n" + "=" * 80)
    print("3. MẪU NGỮ CẢNH THỰC TẾ (15 ARTICLES ĐỘC LẬP)")
    print("=" * 80)

    used_doc_indices = set()
    sample_counter = 1

    def print_context_samples(category_name: str, pattern, max_samples: int = 3, match_group_for_space: int = 2):
        nonlocal sample_counter
        found_indices = []
        for idx, text in enumerate(raw_texts):
            if idx not in used_doc_indices and pattern.search(text):
                found_indices.append(idx)
                if len(found_indices) >= max_samples:
                    break

        print(f"\n>>> NHÓM: {category_name} ({len(found_indices)} mẫu) <<<")
        for idx in found_indices:
            used_doc_indices.add(idx)
            row = df.loc[idx]
            text = raw_texts.loc[idx]
            match = pattern.search(text)
            
            start_pos = max(0, match.start() - 60)
            end_pos = min(len(text), match.end() + 80)
            context_snippet = text[start_pos:end_pos]
            
            raw_ws = match.group(match_group_for_space)
            matched_len = len(raw_ws)

            print(f"\n[Sample {sample_counter}] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
            print(f"• Số horizontal whitespace characters tại điểm phân tách: {matched_len}")
            print(f"• Raw whitespace (repr): {repr(raw_ws)}")
            print(f"• Ngữ cảnh Raw (repr):\n  {repr(context_snippet)}")
            print(f"• Ngữ cảnh hiển thị trực quan:\n  ...{context_snippet}...")
            sample_counter += 1

    print_context_samples("1. CHẤM PHẨY + WHITESPACE + ĐIỂM a), b)", re_semi_point_any, max_samples=4, match_group_for_space=1)
    print_context_samples("2. HAI CHẤM + WHITESPACE + MARKER", re_colon_marker, max_samples=3, match_group_for_space=1)
    print_context_samples("3. CHỮ THƯỜNG + WHITESPACE + ĐIỂM a-zđ", re_inline_point_any, max_samples=4, match_group_for_space=2)
    print_context_samples("4. TEXT + WHITESPACE + NUMBERED MARKER 1., 2., ...", re_inline_clause_any, max_samples=4, match_group_for_space=2)


if __name__ == "__main__":
    analyze_structural_whitespace()