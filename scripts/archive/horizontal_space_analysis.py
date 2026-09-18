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


def collapse_horizontal_spaces(text: str) -> str:
    """
    Thu gọn nhiều khoảng trắng ngang liên tiếp thành 1 space,
    trim khoảng trắng ở đầu/cuối mỗi dòng NHƯNG BẢO TOÀN ký tự xuống dòng \\n.
    """
    lines = text.split("\n")
    cleaned_lines = [re.sub(r"[^\S\r\n]+", " ", line).strip() for line in lines]
    return "\n".join(cleaned_lines)


def print_diff_sample(sample_id: int, category: str, row: pd.Series, raw_text: str, processed_text: str, line_limit: int = 6):
    """Hiển thị so sánh chi tiết trước và sau khi thu gọn khoảng trắng."""
    print(f"\n{'='*80}")
    print(f"[{sample_id}] NHÓM: {category}")
    print(f"Global ID: {row['global_article_id']} | Law ID: {row['law_id']} | AID: {row['aid']}")
    print(f"Độ dài ký tự: {len(raw_text)} -> {len(processed_text)} (Delta: {len(processed_text) - len(raw_text)})")
    print(f"{'='*80}")

    raw_lines = raw_text.split("\n")
    proc_lines = processed_text.split("\n")

    print("[BEFORE - RAW (dấu cách hiển thị nguyên bản)]:")
    for idx, line in enumerate(raw_lines[:line_limit], start=1):
        print(f"  {idx:02d} | {line}")
    if len(raw_lines) > line_limit:
        print(f"  ... [còn {len(raw_lines) - line_limit} dòng]")

    print("\n[AFTER - COLLAPSED (sau khi chuẩn hóa khoảng trắng ngang)]:")
    for idx, line in enumerate(proc_lines[:line_limit], start=1):
        print(f"  {idx:02d} | {line}")
    if len(proc_lines) > line_limit:
        print(f"  ... [còn {len(proc_lines) - line_limit} dòng]")


def analyze_horizontal_spaces():
    corpus_file = resolve_path("data/processed/legal_retrieval/corpus_articles.parquet")
    print("=" * 80)
    print("STEP 5.2: PHÂN TÍCH CHI TIẾT TÁC ĐỘNG CỦA HORIZONTAL SPACE COLLAPSE")
    print("=" * 80)
    print(f"Đang nạp dữ liệu: {corpus_file}\n")

    df = pd.read_parquet(corpus_file)
    raw_texts = df["text"].fillna("").astype(str)
    total_articles = len(df)

    # 1. ĐỊNH NGHĨA CÁC PATTERN KIỂM TRA
    # Pattern A: Khoảng trắng đầu dòng (Indentation: 1+ space/tab ở đầu dòng)
    re_leading = re.compile(r"(?:^|\n)[^\S\r\n]+", re.MULTILINE)
    # Pattern B: Khoảng trắng cuối dòng (Trailing: 1+ space/tab ở cuối dòng trước \n)
    re_trailing = re.compile(r"[^\S\r\n]+(?:\n|$)", re.MULTILINE)
    # Pattern C: 2+ spaces liên tiếp ở giữa 2 từ trong cùng 1 dòng
    re_internal_multi = re.compile(r"\S[^\S\r\n]{2,}\S")
    # Pattern D: Nhiều spaces sau 'Điều X.'
    re_after_dieu = re.compile(r"Điều\s+\d+[\.:][^\S\r\n]{2,}", re.IGNORECASE)
    # Pattern E: Nhiều spaces sau số thứ tự Khoản '1.  ', '2.  '
    re_after_khoan = re.compile(r"(?:^|\n)\s*\d+\.[^\S\r\n]{2,}", re.MULTILINE)
    # Pattern F: Nhiều spaces sau ký tự Điểm 'a)  ', 'b)  '
    re_after_diem = re.compile(r"(?:^|\n)\s*[a-zđ]\)[^\S\r\n]{2,}", re.IGNORECASE | re.MULTILINE)

    # 2. THỐNG KÊ ĐỊNH LƯỢNG TOÀN CORPUS
    has_change = raw_texts.apply(lambda t: collapse_horizontal_spaces(t) != t)
    has_leading = raw_texts.apply(lambda t: bool(re_leading.search(t)))
    has_trailing = raw_texts.apply(lambda t: bool(re_trailing.search(t)))
    has_internal = raw_texts.apply(lambda t: bool(re_internal_multi.search(t)))
    has_dieu_spaces = raw_texts.apply(lambda t: bool(re_after_dieu.search(t)))
    has_khoan_spaces = raw_texts.apply(lambda t: bool(re_after_khoan.search(t)))
    has_diem_spaces = raw_texts.apply(lambda t: bool(re_after_diem.search(t)))

    print("--- 1. PHÂN BỐ ĐỊNH LƯỢNG CÁC LOẠI KHOẢNG TRẮNG NGANG ---")
    print(f"• Tổng số Điều luật trong corpus:                     {total_articles:,}")
    print(f"• Tổng số Điều bị thay đổi bởi collapse_horizontal:  {has_change.sum():,} ({has_change.mean()*100:.2f}%)")
    print(f"  ├─ Có khoảng trắng đầu dòng (Leading / Indent):     {has_leading.sum():,} ({has_leading.mean()*100:.2f}%)")
    print(f"  ├─ Có khoảng trắng cuối dòng (Trailing):            {has_trailing.sum():,} ({has_trailing.mean()*100:.2f}%)")
    print(f"  ├─ Có 2+ spaces giữa các từ trong dòng:             {has_internal.sum():,} ({has_internal.mean()*100:.2f}%)")
    print(f"  ├─ Có 2+ spaces sau 'Điều X.':                      {has_dieu_spaces.sum():,} ({has_dieu_spaces.mean()*100:.2f}%)")
    print(f"  ├─ Có 2+ spaces sau Khoản ('1.  ', '2.  '):         {has_khoan_spaces.sum():,} ({has_khoan_spaces.mean()*100:.2f}%)")
    print(f"  └─ Có 2+ spaces sau Điểm ('a)  ', 'b)  '):           {has_diem_spaces.sum():,} ({has_diem_spaces.mean()*100:.2f}%)")

    # 3. TRÍCH XUẤT VÀ IN CÁC MẪU ĐẠI DIỆN
    print("\n" + "=" * 80)
    print("2. KHẢO SÁT MẪU TRỰC QUAN TRƯỚC VÀ SAU KHI COLLAPSE")
    print("=" * 80)

    sample_counter = 1

    # Nhóm 1: Nhiều spaces sau 'Điều X.' (2 mẫu)
    dieu_indices = df[has_dieu_spaces].index
    for idx in dieu_indices[:2]:
        raw_val = raw_texts.loc[idx]
        proc_val = collapse_horizontal_spaces(raw_val)
        print_diff_sample(sample_counter, "NHIỀU SPACES SAU 'ĐIỀU X.'", df.loc[idx], raw_val, proc_val, line_limit=4)
        sample_counter += 1

    # Nhóm 2: Nhiều spaces sau Khoản '1.  ' (2 mẫu)
    khoan_indices = df[has_khoan_spaces].index
    for idx in khoan_indices[:2]:
        raw_val = raw_texts.loc[idx]
        proc_val = collapse_horizontal_spaces(raw_val)
        print_diff_sample(sample_counter, "NHIỀU SPACES SAU KHOẢN '1.', '2.'", df.loc[idx], raw_val, proc_val, line_limit=5)
        sample_counter += 1

    # Nhóm 3: Nhiều spaces sau Điểm 'a)  ' (2 mẫu)
    diem_indices = df[has_diem_spaces].index
    for idx in diem_indices[:2]:
        raw_val = raw_texts.loc[idx]
        proc_val = collapse_horizontal_spaces(raw_val)
        print_diff_sample(sample_counter, "NHIỀU SPACES SAU ĐIỂM 'a)', 'b)'", df.loc[idx], raw_val, proc_val, line_limit=5)
        sample_counter += 1

    # Nhóm 4: Khoảng trắng đầu dòng (Indentation thuần túy - 2 mẫu)
    leading_only_indices = df[has_leading & ~has_dieu_spaces & ~has_khoan_spaces].index
    for idx in leading_only_indices[:2]:
        raw_val = raw_texts.loc[idx]
        proc_val = collapse_horizontal_spaces(raw_val)
        print_diff_sample(sample_counter, "LEADING WHITESPACE / INDENT ĐẦU DÒNG", df.loc[idx], raw_val, proc_val, line_limit=4)
        sample_counter += 1

    # Nhóm 5: Nhiều spaces giữa các từ trong câu (Internal multi-space - 2 mẫu)
    internal_indices = df[has_internal & ~has_dieu_spaces & ~has_khoan_spaces].index
    for idx in internal_indices[:2]:
        raw_val = raw_texts.loc[idx]
        proc_val = collapse_horizontal_spaces(raw_val)
        print_diff_sample(sample_counter, "2+ SPACES GIỮA CÁC TỪ TRONG CÂU", df.loc[idx], raw_val, proc_val, line_limit=4)
        sample_counter += 1

    # Nhóm 6: Văn bản dài (> 3,000 ký tự) có nhiều khoảng trắng hỗn hợp (2 mẫu)
    long_mixed_indices = df[has_change & (raw_texts.apply(len) > 3000)].index
    for idx in long_mixed_indices[:2]:
        raw_val = raw_texts.loc[idx]
        proc_val = collapse_horizontal_spaces(raw_val)
        print_diff_sample(sample_counter, "VĂN BẢN DÀI CÓ KHOẢNG TRẮNG HỖN HỢP", df.loc[idx], raw_val, proc_val, line_limit=6)
        sample_counter += 1


if __name__ == "__main__":
    analyze_horizontal_spaces()