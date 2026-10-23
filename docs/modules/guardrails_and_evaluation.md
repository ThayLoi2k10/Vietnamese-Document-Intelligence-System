# Module: Guardrails, Citation & Evaluation Engine

## 1. Tổng quan & Trách nhiệm (What does it do?)
Module này bao gồm hệ thống các chốt chặn an toàn (Guardrails), cơ chế truy vết căn cứ pháp lý (Citation & Grounding Engine), và bộ thước đo đối chuẩn chất lượng (Evaluation Metrics) cho cả hai tầng: **Truy xuất thông tin (IR)** và **Tạo sinh câu trả lời (RAG)**.

* **Trọng tâm bảo vệ:**
  * **Chống bịa đặt luật (Anti-hallucination):** Ngăn ngừa LLM phát sinh các số hiệu điều luật giả định không có trong văn bản được trích xuất.
  * **Quyền từ chối an toàn (First-Class Abstention):** Chủ động trả về trạng thái từ chối (`ABSTAINED`) khi không đủ căn cứ vững chắc, thay vì cố gắng suy diễn liều lĩnh.
  * **Đo lường toán học khách quan:** Chuẩn hóa các công thức tính Precision, Recall, MRR, MAP, nDCG theo chuẩn TREC.

---

## 2. Bản đồ Cấu trúc Module

```text
src/
├── evaluation/
│   └── retrieval_metrics.py          # Bộ đo IR: P@k, R@k, MRR@k, MAP@k, nDCG@k
└── rag/
    ├── citation/                     # Tầng kiểm chứng trích dẫn & căn cứ
    │   ├── grounding.py              # GroundingVerifier đối soát số hiệu luật, trích đoạn nguyên văn
    │   ├── provenance/
    │   │   └── evidence_registry.py  # Sổ đăng ký bằng chứng pháp lý (Evidence Registry)
    │   └── validation/
    │       ├── citation_validator.py # Xác thực tính hợp lệ của nhãn [1], [2]
    │       └── models.py             # Validation models & kết quả kiểm định
    ├── abstention/                   # Tầng chính sách từ chối trả lời an toàn
    │   ├── detection/
    │   │   ├── abstention_detector.py# Nhận diện phát biểu từ chối từ câu trả lời của LLM
    │   │   └── models.py             # AbstentionDecision models
    │   └── policies/
    │       ├── abstention_policy.py  # Chính sách kích hoạt từ chối (ngưỡng điểm, thiếu context)
    │       └── abstention_validator.py# Thẩm định tính chuẩn mực của quyết định từ chối
    └── evaluation/                   # Bộ đo lường chất lượng RAG
        └── metrics/                  # Faithfulness, Citation Precision/Recall, Contamination
```

---

## 3. Các Thành phần Cốt lõi & Chi tiết Kỹ thuật

### A. Grounding & Citation Verifier (`src/rag/citation/grounding.py`)
* **`GroundingVerifier`:** Đối soát chéo từng câu khẳng định pháp lý của LLM với danh sách văn bản luật thực tế được truy xuất:
  * **Bóc tách số hiệu văn bản chính quy:** Quét bằng Regex chuyên sâu cho văn bản quy phạm pháp luật Việt Nam:
    `\b\d{1,4}/\d{4}/(?:QH\d+|NĐ-CP|TT-[A-ZĐ]+|QĐ-[A-ZĐ]+|NQ-[A-ZĐ]+)\b`
  * **Bóc tách văn bản định danh:** Nhận diện 19 bộ luật trọng điểm (*Bộ luật Lao động, Bộ luật Dân sự, Luật Đất đai, Luật Doanh nghiệp, Luật BHXH,...*) và tự động quy đổi về số hiệu văn bản tương ứng.
  * **Phán quyết Grounding (`verdict`):**
    * `VERIFIED`: Toàn bộ số hiệu văn bản và trích đoạn đều khớp 100% với tài liệu nguồn.
    * `PARTIAL`: Một phần nội dung được hỗ trợ, có chi tiết suy luận mở rộng.
    * `HALLUCINATION_RISK`: Phát hiện mô hình tự ý đưa ra số hiệu văn bản không hề tồn tại trong context được cấp.
    * `ABSTAINED`: Mô hình chủ động thông báo không đủ thông tin để kết luận.

### B. Cơ chế Từ chối Trả lời An toàn (`src/rag/abstention/`)
* **`AbstentionPolicy`:** Định nghĩa các điều kiện kích hoạt chốt chặn an toàn:
  1. *Ngữ cảnh rỗng:* Bộ truy xuất không tìm thấy bất kỳ điều luật nào liên quan đến câu hỏi.
  2. *Điểm truy xuất dưới ngưỡng an toàn:* Điểm tương đồng ngữ nghĩa của tất cả tài liệu Top-1 đều $< 0.40$.
  3. *Tuyên bố từ chối từ LLM:* Khi trường `is_answerable = False` hoặc LLM sử dụng các cấu trúc câu từ chối chuẩn (*"Tài liệu không đề cập...", "Không có căn cứ để xác định..."*).
* **Kết quả:** Trả về câu trả lời chuẩn mực hướng dẫn người dùng bổ sung thông tin hoặc liên hệ luật sư chuyên trách, triệt tiêu rủi ro tư vấn sai.

### C. Bộ Thước đo Truy xuất Tiêu chuẩn (`src/evaluation/retrieval_metrics.py`)
Toàn bộ hệ thống đo lường được viết bằng NumPy tối ưu hóa:
* **Precision@k / Recall@k:** Đo tỷ lệ tài liệu liên quan được tìm thấy trong Top-$k$.
* **Mean Reciprocal Rank (MRR@10):**
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
* **Mean Average Precision (MAP@k):** Tính tích phân diện tích đường cong P-R trên các truy vấn đa nhãn.
* **Normalized Discounted Cumulative Gain (nDCG@k):** Thước đo chuẩn mực quốc tế đánh giá chất lượng xếp hạng có tính đến vị trí xuất hiện của tài liệu:
  $$\text{DCG}@k = \sum_{i=1}^k \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG}@k = \frac{\text{DCG}@k}{\text{IDCG}@k}$$

---

## 4. Hướng dẫn Lập trình viên Sử dụng

```python
from src.rag.citation.grounding import GroundingVerifier
from src.evaluation.retrieval_metrics import compute_retrieval_metrics

# 1. Kiểm chứng căn cứ pháp lý của câu trả lời sinh ra
verifier = GroundingVerifier()
grounding_res = verifier.verify(
    generated_text="Căn cứ Điều 36 Bộ luật Lao động 45/2019/QH14, người lao động có quyền...",
    retrieved_docs=[{"law_id": "45/2019/QH14", "text": "Điều 36. Quyền đơn phương chấm dứt..."}]
)

print(f"Trạng thái: {grounding_res.verdict}")          # 'VERIFIED'
print(f"Điểm tin cậy: {grounding_res.grounding_score}") # 1.0
print(f"Luật hợp lệ: {grounding_res.verified_laws}")     # ['45/2019/QH14']

# 2. Đánh giá chất lượng xếp hạng IR
metrics = compute_retrieval_metrics(
    retrieved_ids=[124, 532, 98],
    relevant_ids=[98, 105],
    k_list=[1, 5, 10]
)
print(f"Recall@5: {metrics['R@5']}, nDCG@10: {metrics['nDCG@10']}")
```
