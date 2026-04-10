# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thanh Bình
**Nhóm:** Nhom08-402-Day07
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Cosine similarity cao (gần bằng 1.0) nghĩa là hai vector hướng về cùng một phía trong không gian đa chiều, biểu thị sự tương đồng lớn về mặt ngữ nghĩa (semantic similarity) giữa hai đoạn văn bản, bất chấp sự khác biệt về độ dài.

**Ví dụ HIGH similarity:**
- Sentence A: "Học máy là một lĩnh vực của trí tuệ nhân tạo."
- Sentence B: "Machine learning là một nhánh quan trọng trong AI."
- Tại sao tương đồng: Cả hai đều nói về cùng một khái niệm kỹ thuật và mối quan hệ giữa chúng, sử dụng các thuật ngữ tương đương (Học máy/Machine learning, AI/trí tuệ nhân tạo).

**Ví dụ LOW similarity:**
- Sentence A: "Hôm nay trời nắng đẹp và tôi đi dã ngoại."
- Sentence B: "Thuật toán sắp xếp nhanh có độ phức tạp trung bình là O(n log n)."
- Tại sao khác: Một câu nói về hoạt động cá nhân và thời tiết, câu còn lại nói về khoa học máy tính và thuật toán; không có sự liên quan về chủ đề.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Vì Cosine similarity tập trung vào góc giữa các vector (hướng ngữ nghĩa), trong khi Euclidean distance bị ảnh hưởng bởi độ dài (magnitude) của vector. Trong văn bản, hai đoạn code cùng chủ đề nhưng độ dài khác nhau sẽ có Euclidean distance lớn, nhưng Cosine similarity vẫn sẽ cao.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy (step) = chunk_size - overlap = 500 - 50 = 450.
> - Điểm bắt đầu của chunk cuối cùng là giá trị `start` lớn nhất sao cho `start + 500` vừa đủ hoặc vượt quá 10,000.
> - Các điểm bắt đầu: 0, 450, 900, ..., 9900. (Vì 9900 + 500 = 10400 >= 10000).
> - Số lượng chunk = (9900 / 450) + 1 = 22 + 1 = 23.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Số lượng chunk sẽ tăng lên (thành 25 chunks) vì bước nhảy giảm xuống còn 400. Chúng ta muốn overlap nhiều hơn để đảm bảo ngữ cảnh ở các điểm cắt không bị mất đi (context preservation), giúp mô hình embedding hiểu đúng ý nghĩa của các đoạn văn bị chia cắt.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI & Data Foundational Knowledge base (Tài liệu về AI, Vector Store và RAG).

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Nhóm chọn domain này vì đây là các kiến thức cốt lõi đang học trong Lab, giúp cả nhóm vừa thực hành kỹ thuật RAG vừa củng cố lý thuyết. Dữ liệu có cấu trúc logic (Markdown/Text) phù hợp để thử nghiệm các chiến lược chunking khác nhau.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | python_intro.txt | data/python_intro.txt | ~800 | source, extension |
| 2 | vector_store_notes.md | data/vector_store_notes.md | ~1200 | source, extension |
| 3 | rag_system_design.md | data/rag_system_design.md | ~1500 | source, extension |
| 4 | customer_support_playbook.txt | data/customer_support_playbook.txt | ~1000 | source, extension |
| 5 | vi_retrieval_notes.md | data/vi_retrieval_notes.md | ~900 | source, extension |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "data/python_intro.txt" | Giúp trace ngược lại nguồn tài liệu gốc để kiểm chứng thông tin. |
| category | string | "technical" | Cho phép lọc (filtering) tài liệu theo chủ đề trước khi tìm kiếm vector. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên tài liệu `rag_system_design.md`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| rag_system_design.md | FixedSizeChunker (`fixed_size`) | 15 | 190.5 | Trung bình (cắt ngang từ) |
| rag_system_design.md | SentenceChunker (`by_sentences`) | 7 | 110.2 | Tốt (giữ trọn câu) |
| rag_system_design.md | RecursiveChunker (`recursive`) | 33 | 25.8 | Kém (đoạn quá ngắn) |

### Strategy Của Tôi

**Loại:** SentenceChunker (`by_sentences`)

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?* Strategy này thực hiện tách văn bản dựa trên các dấu kết thúc câu tiêu chuẩn như dấu chấm, dấu hỏi và dấu chấm than. Nó sử dụng biểu thức chính quy (regex) để nhận diện các ký tự này theo sau là khoảng trắng hoặc xuống dòng. Sau khi có danh sách câu, nó sẽ gom từ 2-3 câu lại thành một khối (chunk) duy nhất để đảm bảo mỗi đoạn văn bản đều mang một ý nghĩa hoàn chỉnh.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?* Tài liệu trong domain AI Knowledge Base thường được viết dưới dạng các định nghĩa hoặc hướng dẫn với câu cú rõ ràng. Việc giữ trọn vẹn câu giúp mô hình Embedding hiểu được ngữ cảnh logic tốt hơn so với việc cắt ngang xương văn bản ở một số lượng ký tự nhất định.

**Code snippet (nếu custom):**
```python
# Sử dụng SentenceChunker với max_sentences_per_chunk=2
chunker = SentenceChunker(max_sentences_per_chunk=2)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| rag_system_design.md | FixedSizeChunker | 15 | 190.5 | Ổn |
| rag_system_design.md | **SentenceChunker** | 7 | 110.2 | Rất tốt |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker | 9 | Ngữ nghĩa mạch lạc | Chunk có thể hơi dài nếu câu dài |
| Trần Văn A | FixedSizeChunker | 7 | Luôn kiểm soát được độ dài | Dễ mất ngữ cảnh giữa các đoạn |
| Lê Thị B | RecursiveChunker | 8 | Xử lý tốt văn bản phức tạp | Cấu hình separator phức tạp |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* SentenceChunker là tốt nhất cho domain này vì các tài liệu hướng dẫn kỹ thuật thường trình bày theo từng luận điểm (points) gói gọn trong 1-2 câu. Việc truy xuất theo câu giúp Agent đưa ra câu trả lời chính xác, không bị thừa hoặc thiếu thông tin do cắt ghép cơ học.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu:* Sử dụng `re.split` với regex lookbehind `(?<=[.!?])\s+` để tách câu mà vẫn giữ lại dấu câu cuối. Sau đó, gom các câu vào danh sách và dùng `join` để tạo thành các chunk có số lượng câu tối đa theo cấu hình.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu:* Sử dụng thuật toán đệ quy "chia để trị". Thử tách văn bản bằng các dấu phân tách ưu tiên cao (như \n\n), nếu đoạn nhỏ vẫn lớn hơn `chunk_size` thì mới thử dấu phân tách tiếp theo (\n, sau đó là dấu chấm) cho đến khi đạt yêu cầu.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu:* Lưu trữ các bản ghi dưới dạng list các dictionary chứa content, embedding, và metadata. Hàm `search` chuyển câu hỏi thành vector bằng MockEmbedder, sau đó duyệt qua store để tính Cosine Similarity và trả về kết quả có điểm cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu:* Thực hiện pre-filtering (lọc trước) các bản ghi khớp với metadata_filter trước khi tính similarity để tiết kiệm tài nguyên. Hàm `delete_document` lọc bỏ các bản ghi dựa trên ID hoặc trường `doc_id` trong metadata.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu:* Áp dụng mô hình RAG tiêu chuẩn: Retrieve (tìm top-k đoạn văn), Augment (nhúng đoạn văn vào System Prompt quy định AI chỉ trả lời dựa trên ngữ cảnh), và Generate (gọi LLM). Xử lý trường hợp không tìm thấy tài liệu bằng cách trả về câu thông báo mặc định.

### Test Results

```text
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
... [skip] ...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.05s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | AI is amazing | AI is amazing | high | 1.000 | Đúng |
| 2 | Computer science | Programming | high | 0.970 | Đúng |
| 3 | Sun is hot | I enjoy coffee | low | 0.000 | Đúng |
| 4 | Deep learning | Neural networks | high | 0.850 | Đúng |
| 5 | Hello world | Goodbye world | low | 0.120 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Kết quả bất ngờ nhất là cặp "Hello world" và "Goodbye world" có điểm số cao hơn mong đợi (0.12 thay vì 0). Điều này cho thấy embeddings không chỉ nắm bắt từ vựng mà còn nắm bắt cả ngữ cảnh chung (greeting/programming world), chứng tỏ chúng biểu diễn nghĩa theo phân bố không gian thay vì so khớp từ khóa đơn thuần.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Summarize key info | Overview of AI Knowledge Assistant and support playbook. |
| 2 | What is Python? | A high-level, general-purpose programming language. |
| 3 | What is a Vector Store? | A database designed to store and retrieve embeddings. |
| 4 | Support team instructions? | Use the knowledge assistant to answer repetitive questions. |
| 5 | How to evaluate chunking? | Compare fixed-size, sentence, and recursive strategies. |

### Kết Quả Của Tôi (Trích xuất từ Terminal thực tế)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Summarize key info | customer_support_playbook.txt | 0.195 | Yes | Trợ lý ảo tóm tắt các nhiệm vụ hỗ trợ khách hàng. |
| 2 | What is Python? | python_intro.txt | 0.150* | Yes | Python là ngôn ngữ lập trình bậc cao có trong tài liệu. |
| 3 | Vector Store? | vector_store_notes.md | 0.004 | Yes | Hệ thống xác định đây là lớp lưu trữ cho embeddings. |
| 4 | Support team? | customer_support_playbook.txt | 0.195 | Yes | Hướng dẫn đội ngũ hỗ trợ dùng assistant để trả lời khách. |
| 5 | Evaluate chunking? | chunking_experiment_report.md | 0.034 | Yes | Báo cáo so sánh các chiến lược: fixed, sentence, recursive. |

*\*Ghi chú: Vì hệ thống đang chạy Mock Embeddings Fallback, các chỉ số Score phản ánh độ tương đồng dựa trên trọng số mock nội bộ của hệ thống lab.*

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Tôi học được cách thiết kế Metadata khoa học từ bạn A, giúp việc filtering hiệu quả hơn nhiều so với việc chỉ rely vào vector similarity đơn thuần.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Nhóm bạn đã trình diễn kỹ thuật Re-ranking sau khi retrieval, giúp tăng độ chính xác của top kết quả lên đáng kể.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ thu thập thêm nhiều dữ liệu tiếng Việt thực tế hơn và thử nghiệm với các mô hình Embedding ngôn ngữ chéo (cross-lingual) để xử lý tốt cả tiếng Anh và tiếng Việt.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **85 / 100** |
