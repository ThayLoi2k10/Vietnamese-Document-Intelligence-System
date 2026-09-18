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


# ==========================================
# CÁC HÀM TRANSFORMATION THỬ NGHIỆM
# ==========================================

def apply_nfc(text: str) -> str:
    """Chuẩn hóa Unicode về dạng dựng sẵn NFC."""
    return unicodedata.normalize("NFC", text)


def apply_nbsp_to_space(text: str) -> str:
    """Thay thế ký tự non-breaking space (\\u00a0) bằng space thông thường."""
    return text.replace("\u00a0", " ")


def remove_carriage_returns(text: str) -> str:
    """Loại bỏ ký tự \\r (chuẩn hóa \\r\\n -> \\n và \\r -> \\n)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_horizontal_spaces(text: str) -> str:
    """
    Thu gọn nhiều khoảng trắng ngang liên tiếp thành 1 space,
    trim khoảng trắng ở đầu/cuối mỗi dòng NHƯNG BẢO TOÀN ký tự xuống dòng \\n.
    """
    lines = text.split("\n")
    cleaned_lines = [re.sub(r"[^\S\r\n]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned_lines)


def full_pipeline_preview(text: str) -> str:
    """Kết hợp toàn bộ các bước chuẩn hóa đề xuất."""
    t = apply_nfc(text)
    t = remove_carriage_returns(t)
    t = apply_nbsp_to_space(t)
    t = collapse_horizontal_spaces(t)
    return t


# ==========================================
# HÀM HIỂN THỊ DIFF KÝ TỰ
# ==========================================

def display_diff(label: str, raw: str, processed: str):
    """In so sánh chi tiết trước và sau khi biến đổi."""
    print(f"\n--- {label} ---")
    print(f"  • Độ dài ký tự: {len(raw)} -> {len(processed)} (Delta: {len(processed) - len(raw)})")
    print(f"  • Số dòng (\\n): {raw.count(chr(10))} -> {processed.count(chr(10))}")
    
    # Tìm các ký tự đặc biệt bị thay đổi
    raw_repr = repr(raw[:150] + ("..." if len(raw) > 150 else ""))
    proc_repr = repr(processed[:150] + ("..." if len(processed) > 150 else ""))
    print(f"  • Raw repr:       {raw_repr}")
    print(f"  • Processed repr: {proc_repr}")


# ==========================================
# MAIN ANALYZER
# ==========================================

def analyze_normalization_effects():
    corpus_file = resolve_path("data/processed/legal_retrieval/corpus_articles.parquet")
    print("=" * 80)
    print("STEP 5.1: PHÂN TÍCH TÁC ĐỘNG CỦA CÁC NORMALIZATION TRANSFORMATIONS")
    print("=" * 80)
    print(f"Đang nạp dữ liệu: {corpus_file}\n")

    df = pd.read_parquet(corpus_file)
    raw_texts = df["text"].fillna("").astype(str)
    total_articles = len(df)

    # 1. THỐNG KÊ TÁC ĐỘNG TOÀN CỤC (CORPUS-WIDE IMPACT)
    print("--- 1. THỐNG KÊ TÁC ĐỘNG TOÀN CỤC TRÊN 59,636 ĐIỀU LUẬT ---")
    
    # (a) Unicode NFC
    nfc_changed_mask = raw_texts.apply(lambda t: unicodedata.normalize("NFC", t) != t)
    nfc_changed_count = nfc_changed_mask.sum()
    print(f"• Số Điều bị thay đổi bởi Unicode NFC:               {nfc_changed_count:,} ({nfc_changed_count / total_articles * 100:.2f}%)")

    # (b) Carriage Return (\r)
    cr_mask = raw_texts.apply(lambda t: "\r" in t)
    cr_count = cr_mask.sum()
    print(f"• Số Điều có chứa ký tự \\r (CR / CRLF):              {cr_count:,} ({cr_count / total_articles * 100:.2f}%)")

    # (c) NBSP (\u00a0)
    nbsp_mask = raw_texts.apply(lambda t: "\u00a0" in t)
    nbsp_count = nbsp_mask.sum()
    print(f"• Số Điều có chứa NBSP (\\u00a0):                    {nbsp_count:,} ({nbsp_count / total_articles * 100:.2f}%)")

    # (d) Horizontal spaces collapse
    space_changed_mask = raw_texts.apply(lambda t: collapse_horizontal_spaces(t) != t)
    space_changed_count = space_changed_mask.sum()
    print(f"• Số Điều bị thay đổi bởi Horizontal Space Collapse: {space_changed_count:,} ({space_changed_count / total_articles * 100:.2f}%)")

    # (e) Cumulative Pipeline
    all_changed_mask = raw_texts.apply(lambda t: full_pipeline_preview(t) != t)
    all_changed_count = all_changed_mask.sum()
    print(f"• Tổng số Điều bị tác động bởi full pipeline:       {all_changed_count:,} ({all_changed_count / total_articles * 100:.2f}%)")

    # 2. KHẢO SÁT MẪU CHI TIẾT THEO TỪNG TRƯỜNG HỢP BIÊN
    print("\n" + "=" * 80)
    print("2. KHẢO SÁT MẪU THỰC TẾ TRƯỚC VÀ SAU BIẾN ĐỔI")
    print("=" * 80)

    # Mẫu A: Chưa chuẩn NFC
    if nfc_changed_count > 0:
        sample_idx = df[nfc_changed_mask].index[0]
        raw_val = raw_texts.loc[sample_idx]
        proc_val = apply_nfc(raw_val)
        row = df.loc[sample_idx]
        print(f"\n[MẪU A - UNICODE NFD -> NFC] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
        display_diff("So sánh Unicode NFC", raw_val, proc_val)
        
        # Chỉ ra các ký tự thay đổi byte
        diff_words = []
        for w1, w2 in zip(raw_val.split(), proc_val.split()):
            if w1 != w2:
                diff_words.append(f"'{w1}' (len {len(w1)}) -> '{w2}' (len {len(w2)})")
        if diff_words:
            print(f"  • Từ bị thay đổi tổ hợp: {', '.join(diff_words[:5])}")

    # Mẫu B: Chứa NBSP (\u00a0)
    if nbsp_count > 0:
        sample_idx = df[nbsp_mask].index[0]
        raw_val = raw_texts.loc[sample_idx]
        proc_val = apply_nbsp_to_space(raw_val)
        row = df.loc[sample_idx]
        print(f"\n[MẪU B - CHỨA NBSP (\\xa0)] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
        display_diff("So sánh NBSP -> Space", raw_val, proc_val)
        print(f"  • Số lượng \\u00a0 trong raw: {raw_val.count(chr(160))}")

    # Mẫu C: Chứa \r (nếu có)
    if cr_count > 0:
        sample_idx = df[cr_mask].index[0]
        raw_val = raw_texts.loc[sample_idx]
        proc_val = remove_carriage_returns(raw_val)
        row = df.loc[sample_idx]
        print(f"\n[MẪU C - CHỨA CARRIAGE RETURN (\\r)] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
        display_diff("So sánh Remove \\r", raw_val, proc_val)
    else:
        print("\n[MẪU C - CARRIAGE RETURN] Toàn bộ corpus không chứa ký tự \\r (0 records).")

    # Mẫu D: Nhiều dòng (\n)
    newline_counts = raw_texts.apply(lambda t: t.count("\n"))
    many_nl_idx = newline_counts.sort_values(ascending=False).index[0]
    raw_val = raw_texts.loc[many_nl_idx]
    proc_val = full_pipeline_preview(raw_val)
    row = df.loc[many_nl_idx]
    print(f"\n[MẪU D - NHIỀU XUỐNG DÒNG \\n] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
    display_diff("So sánh Full Pipeline trên bài nhiều dòng", raw_val, proc_val)
    print("  • 3 dòng đầu sau khi normalize:")
    for line in proc_val.split("\n")[:3]:
        print(f"    | {line}")

    # Mẫu E: Bài bình thường (Median length)
    non_empty = raw_texts[raw_texts.str.strip().ne("")]
    med_len = non_empty.apply(len).median()
    med_idx = (non_empty.apply(len) - med_len).abs().sort_values().index[0]
    raw_val = raw_texts.loc[med_idx]
    proc_val = full_pipeline_preview(raw_val)
    row = df.loc[med_idx]
    print(f"\n[MẪU E - BÀI BÌNH THƯỜNG / MEDIAN] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
    display_diff("So sánh Full Pipeline trên bài bình thường", raw_val, proc_val)

    # Mẫu F: Bài ngắn
    short_idx = non_empty.apply(len).sort_values().index[0]
    raw_val = raw_texts.loc[short_idx]
    proc_val = full_pipeline_preview(raw_val)
    row = df.loc[short_idx]
    print(f"\n[MẪU F - BÀI NGẮN] Global ID: {row['global_article_id']} | Law: {row['law_id']} | AID: {row['aid']}")
    display_diff("So sánh Full Pipeline trên bài ngắn", raw_val, proc_val)
    print(f"  • Nội dung sau normalize: {repr(proc_val)}")


if __name__ == "__main__":
    analyze_normalization_effects()