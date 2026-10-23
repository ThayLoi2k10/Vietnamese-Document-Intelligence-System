# Module: Query Processing & Contextualizer (`src/rag/query/`)

## 1. Tổng quan & Trách nhiệm (What does it do?)
Gói module [`src/rag/query/`](file:///d:/PROJECT/PROJECT%202/src/rag/query/) chịu trách nhiệm tinh chỉnh, chuẩn hóa và tối ưu hóa câu hỏi của người dùng trước khi chuyển giao cho bộ truy xuất dữ liệu:
* **Chuẩn hóa từ viết tắt (Legal Acronym Normalization):** Giải mã 48+ từ viết tắt pháp luật phổ biến (`NLĐ`, `BHXH`, `QSDĐ`, `UBND`).
* **Mở rộng thuật ngữ thận trọng (LLM Rewriting):** Chuyển dịch ngôn ngữ đời thường sang ngôn ngữ quy phạm pháp luật mà không làm trôi dạt ý định người dùng.
* **Ngữ cảnh hóa hội thoại nhiều lượt (Conversation Contextualizer):** Nhận diện câu hỏi tiếp nối (follow-up query), thay thế đại từ chỉ định (*"vậy thì sao?", "trường hợp này"*), và loại bỏ lời chào mở đầu (*"cho tôi hỏi", "luật sư cho em hỏi"*).
* **Cổng chống mất thông tin (Information Loss Gate):** Fallback an toàn nếu việc mở rộng làm mất mát quá 50% từ khóa gốc.

---

## 2. Bản đồ Cấu trúc Module (Package Map)

```text
src/rag/query/
├── __init__.py           # Public exports: Normalizer, Rewriter, Processor, Contextualizer
├── models.py             # Dataclasses: ProcessedQuery, ProcessingMetadata, FailureType
├── normalizer.py         # LegalAcronymNormalizer (Regex Unicode Lookaround, < 0.2 ms)
├── rewriter.py           # LLMQueryRewriter (Prompt conservative expansion & JSON parse)
├── processor.py          # QueryProcessor (Điều phối 3 tầng & Fallback logic)
└── contextualizer.py     # ConversationContextualizer (Xử lý hội thoại nhiều lượt)
```

---

## 3. Các Class Cốt lõi & Luồng Xử lý

### A. `LegalAcronymNormalizer` (`src/rag/query/normalizer.py`)
* Sử dụng Regex Unicode Lookaround để nhận diện chính xác các từ viết tắt pháp lý mà không bị nhầm lẫn với các từ thông thường:
  - Chế độ `Mode.EXPAND` (Mặc định cho Search): Chuyển `"NLĐ"` thành `"NLĐ (người lao động)"`. Giúp cả thuật toán từ khóa BM25 lẫn thuật toán vector Dense đều bắt được tín hiệu tối đa.
  - Chế độ `Mode.REPLACE`: Chuyển `"NLĐ"` thành `"người lao động"`.
* Danh mục 48+ từ viết tắt: Bao phủ toàn diện các lĩnh vực Lao động, Đất đai, Thuế, Doanh nghiệp và Cơ quan Tư pháp.

### B. `ConversationContextualizer` (`src/rag/query/contextualizer.py`)
* Được thiết kế phục vụ giao diện Chatbot hỏi đáp pháp lý tương tác nhiều lượt:
  1. *Khử lời chào mở đầu:* Tự động bóc tách các cụm từ đệm dư thừa như `"cho tôi hỏi:"`, `"luật sư cho em hỏi:"`, `"xin hỏi:"` để câu truy vấn tinh gọn, tập trung vào bản chất pháp lý.
  2. *Hợp nhất chủ đề tiếp nối:* Nếu người dùng hỏi câu tiếp theo dạng *"thời gian đó là bao lâu?"* hoặc *"còn đối với việc này thì sao?"*, contextualizer sẽ trích xuất chủ thể từ câu hỏi trước đó và ghép vào câu hỏi hiện tại.

### C. `QueryProcessor` (`src/rag/query/processor.py`)
* Bộ điều phối hợp nhất kết nối tuần tự:
  $$\text{Raw Query} \xrightarrow{\text{Normalizer}} \text{Normalized Query} \xrightarrow{\text{LLM Rewriter}} \text{Rewritten Query}$$
* Tích hợp **Information Loss Gate**: Đo tỷ lệ giao thoa từ khóa (Token Overlap Ratio):
  $$\text{Overlap} = \frac{|\text{Tokens}(\text{Rewritten}) \cap \text{Tokens}(\text{Original})|}{|\text{Tokens}(\text{Original})|}$$
  Nếu $\text{Overlap} < 0.50$, hệ thống tự động gắn nhãn lỗi `FailureType.INFORMATION_LOSS` và hủy câu viết lại của LLM, lập tức kích hoạt fallback quay về kết quả của Normalizer.

---

## 4. Hướng dẫn Lập trình viên Sử dụng

```python
from src.rag.query.normalizer import LegalAcronymNormalizer
from src.rag.query.contextualizer import ConversationContextualizer

# 1. Chuẩn hóa từ viết tắt pháp lý siêu tốc
normalizer = LegalAcronymNormalizer()
clean_text = normalizer.normalize("NLĐ bị công ty nợ BHXH thì khiếu nại ở đâu?")
print(clean_text)
# Output: "NLĐ (người lao động) bị công ty nợ BHXH (bảo hiểm xã hội) thì khiếu nại ở đâu?"

# 2. Xử lý câu hỏi tiếp nối trong hội thoại nhiều lượt (Multi-turn Chat)
contextualizer = ConversationContextualizer()
history = [
    {"user": "Thời hạn thử việc của người lao động có trình độ đại học?", "assistant": "Tối đa là 60 ngày theo Bộ luật Lao động."}
]
follow_up = contextualizer.contextualize_query(
    current_query="Vậy trong thời gian đó có được hưởng lương không?",
    conversation_history=history
)
print(follow_up)
# Output: "Trong thời gian thử việc của người lao động có trình độ đại học có được hưởng lương không?"
```
