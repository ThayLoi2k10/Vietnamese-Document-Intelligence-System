# Phase 12 — Đặc tả Kỹ thuật Tái xếp hạng (Reranking Specification)

## 1. Bối cảnh và Mục tiêu (Context & Objective)

Trong Phase 11, mô hình **Hybrid Retrieval (Weighted RRF)** đã cải thiện đáng kể khả năng tìm kiếm ứng viên, đưa **Recall@100 đạt 95.40%** và giảm tỷ lệ zero-hit xuống chỉ còn 1.6% (7 câu). Tuy nhiên:
- Bộ truy xuất Bi-Encoder và BM25 tính toán điểm tương đồng dựa trên biểu diễn vector tách biệt (dual-encoder) hoặc tần suất từ vựng, không có cơ chế **Cross-Attention đầy đủ** giữa từng token của câu hỏi và từng token của đoạn văn bản luật.
- Các tài liệu thực sự liên quan đôi khi nằm ở thứ hạng từ 6 đến 20, chưa vào được Top-5 để đưa vào context window của LLM.

**Mục tiêu Phase 12:** 
1. Tích hợp mô hình **Cross-Encoder Reranker** (`BAAI/bge-reranker-base`) để tái sắp xếp Top-20 ứng viên do Hybrid Retriever đề xuất thành Top-5 tinh tuyển.
2. Đánh giá thực nghiệm hai tác động:
   - **Tác động Truy xuất (IR Impact):** Đo lường sự thay đổi của Recall@5, Recall@10, Recall@20, MRR, nDCG trước và sau khi reranking.
   - **Tác động Hệ thống RAG (Downstream Impact):** Đánh giá mức độ chuyển hóa của chất lượng truy xuất sang độ chính xác trích dẫn (Citation Precision/Recall), độ trung thực (Faithfulness), và độ trễ (Latency).

---

## 2. Kiến trúc Đường ống (Pipeline Architecture)

Đường ống thực thi tuân thủ nghiêm ngặt mô hình 2 tầng:

```
Query (Câu hỏi)
  ↓
[Tầng 1: Candidate Generation] -> Hybrid Weighted RRF (Top-20 ứng viên từ 59,636 điều luật)
  ↓
[Tầng 2: Deep Cross-Encoder]   -> BGE Reranker (Chấm điểm full cross-attention trên 20 cặp Query-Doc)
  ↓
Top-5 Điều luật tinh tuyển
  ↓
[Downstream RAG Context Policy] -> Đưa vào Context Window -> LLM Generation / Evaluation
```

---

## 3. Lựa chọn Mô hình Reranker (Model Selection Rationale)

1. **Ứng viên xem xét:**
   - `BAAI/bge-reranker-base`: Mô hình Cross-Encoder đa ngôn ngữ (XLM-RoBERTa backbone, 278M tham số, ~1.1GB), hỗ trợ tiếng Việt xuất sắc, chi phí suy luận vừa phải (~15-30ms / 20 pairs trên GPU/CPU).
   - `BAAI/bge-reranker-v2-m3`: Mô hình lớn hơn (~2.2GB), hiệu năng cao nhưng độ trễ tăng gấp 2.5 lần.
2. **Lựa chọn chính thức:** `BAAI/bge-reranker-base` được chọn làm reranker sản xuất nhờ sự cân bằng tối ưu giữa độ chính xác cross-attention và độ trễ chấp nhận được trong ứng dụng thời gian thực.

---

## 4. Giao thức Đánh giá & Metrics (Evaluation Protocol)

### 4.1. Metrics Đánh giá Truy xuất (Retrieval Metrics):
- `Recall@5`, `Recall@10`, `Recall@20`
- `MRR@10`, `MAP@10`
- `nDCG@5`, `nDCG@10`
- `Candidate Pool Recall@20`: Giới hạn trần lý thuyết (upper bound) mà Reranker có thể đạt được.

### 4.2. Metrics Đánh giá RAG (Generation & Grounding Metrics):
- `Citation Precision`: Tỷ lệ trích dẫn chính xác điều luật căn cứ.
- `Citation Recall`: Tỷ lệ căn cứ pháp lý ground-truth được bao phủ.
- `Faithfulness`: Mức độ trung thực và bám sát ngữ cảnh.
- `End-to-end Latency`: Thời gian xử lý toàn trình (Retrieval + Rerank + Generation).

---

## 5. Ràng buộc Đóng băng & Chống Rò rỉ (Frozen Constraints)

1. **Bộ sinh ứng viên:** Sử dụng cố định Hybrid Weighted RRF ($k=5, w_d=0.75, w_b=0.25$) từ Phase 11.
2. **Không gian văn bản:** Sử dụng nội dung nguyên bản `text_preprocessed` của điều luật từ `corpus_articles_preprocessed.parquet`.
3. **Quy tắc chia tập:** Đánh giá mẫu kiểm chứng và tinh chỉnh độ sâu (candidate pool depth) trên tập Train/Dev; báo cáo chính thức trên Frozen Test (`retrieval_test.json`).
