# MedRAGAgents: Medical Multi-Agent RAG System for MedQA-USMLE

MedRAGAgents là hệ thống Multi-Agent Y khoa hợp nhất được xây dựng cho Đề tài Giữa kỳ môn Machine Learning in Security (UIT-SDH). Hệ thống kết hợp lập luận đa Agent chuyên gia y khoa, truy xuất tri thức y khoa bổ trợ (MedRAG) và kiến trúc bộ nhớ hai tầng (Dual-Layer Memory) nhằm tối ưu hóa độ chính xác thực tế (factual accuracy) và giảm thiểu hiện tượng ảo giác (hallucination) trên bộ dữ liệu MedQA-USMLE.

---

## 1. Ý Tưởng Cốt Lõi và Đặt Vấn Đề (Project Motivation & Core Idea)

Trong thực hành lâm sàng, việc chẩn đoán các ca bệnh phức tạp (như trong bộ câu hỏi MedQA-USMLE) đòi hỏi sự phối hợp giữa nhiều chuyên khoa y học (Giải phẫu học, Dược lý học, Bệnh lý học, Nội khoa...). Các mô hình ngôn ngữ lớn (LLM) đơn lẻ thường gặp hạn chế ảo giác kiến thức hoặc tự tin đưa ra chẩn đoán sai khi thiếu sự tư vấn đa chuyên khoa và thiếu nguồn tài liệu tham khảo chính thống.

Hệ thống MedRAGAgents giải quyết các hạn chế trên thông qua 3 trụ cột thiết kế chính:

1. Phân rã đa chuyên khoa (Multi-Specialty Agent Decomposition):
   Thay vì gửi câu hỏi cho một mô hình duy nhất, kịch bản lâm sàng được phân loại thành các vùng tri thức y khoa chuyên sâu. Các Agent đại diện cho từng chuyên khoa sẽ phân tích độc lập câu hỏi theo đúng góc nhìn chuyên môn của họ.
2. Truy xuất chứng cứ y khoa (MedRAG Retrieval):
   Tri thức y khoa được truy xuất động từ csdl sách giáo khoa y khoa chuẩn (Textbooks corpus) thông qua mô hình nhúng MedCPT vector, giúp bổ sung chứng cứ lý thuyết chính xác cho các Agent lập luận.
3. Hội chẩn đồng thuận và Bộ nhớ hai tầng (Consensus Verification & Dual-Layer Memory):
   Một Verifier Agent đóng vai trò chủ trì luồng hội chẩn, điều phối các Agent chuyên khoa bỏ phiếu (YES/NO) và sửa đổi báo cáo tổng hợp cho đến khi đạt đồng thuận. Trạng thái trung gian được lưu trong Bộ nhớ ngắn hạn (Short-Term Memory), trong khi các dự đoán đã hoàn tất được lưu vào Bộ nhớ dài hạn (Long-Term Memory dạng SHA-256 cache) để tối ưu chi phí và độ trễ.

---

## 2. Cách Thức Hoạt Động của Hệ Thống (System Workflow)

### Sơ Đồ Quy Trình Hoạt Động Tổng Thể (Mermaid Architecture Diagram)

```mermaid
flowchart TD
    A["run.py\nload_dataset() / extract_fields()"] --> B["run.py\nrun_variant()"]
    B --> C["pipeline.py\nbuild_pipeline(variant)"]

    subgraph V0["V0 — Baseline"]
        D["pipeline.py\nV0DirectLLM.run()"]
    end

    subgraph V1["V1 — RAG-only"]
        E["pipeline.py\nV1RAGOnly.run()"]
        E1["agents/rag_agent.py\nRAGAgent._init_retrieval()"]
        E2["MedRAG/src/utils.py\nRetrievalSystem.retrieve()"]
        E3["agents/base_agent.py\nBaseAgent.call()"]
    end

    subgraph V2["V2 — Multi-agent"]
        F["pipeline.py\nV2MultiAgent.run()"]
        F1["agents/domain_agent.py\nDomainAgent.classify_question_domains()"]
        F2["agents/domain_agent.py\nDomainAgent.classify_option_domains()"]
        F3["agents/analysis_agent.py\nAnalysisAgent.run_question_analyses()"]
        F4["agents/analysis_agent.py\nAnalysisAgent.run_option_analyses()"]
        F5["agents/synthesis_agent.py\nSynthesisAgent.synthesise()"]
        F6["agents/verifier_agent.py\nVerifierAgent.verify_and_answer()"]
        F7["agents/verifier_agent.py\n_vote() / _get_advice() / _revise() / _derive_final_answer()"]
    end

    subgraph V3["V3 — Full system"]
        G["pipeline.py\nV3FullSystem.run()"]
        G1["memory.py\nMemoryManager.reset_short()"]
        G2["memory.py\nLongTermMemory.get() / store()"]
        G3["agents/rag_agent.py\nRAGAgent.answer()"]
        G4["memory.py\nShortTermMemory.store()/get()"]
    end

    C --> D
    C --> E
    C --> F
    C --> G

    E --> E1 --> E2 --> E3

    F --> F1
    F --> F2
    F1 --> F3
    F2 --> F4
    F3 --> F5
    F4 --> F5
    F5 --> F6 --> F7

    G --> G1
    G --> G2
    G --> G3
    G --> G4
    G3 --> E2
    G4 --> F1
    G4 --> F3

    D --> H["outputs/<variant>_predictions.jsonl"]
    E3 --> H
    F7 --> H
    G --> H
    H --> I["evaluate.py\nEvaluator.full_report()"]
```

### Luồng Gọi Theo File Code

Luồng thực tế của hệ thống được điều phối theo thứ tự sau:

1. [MedRAGAgents/run.py](MedRAGAgents/run.py) đọc dataset MedQA, chuẩn hóa câu hỏi bằng `load_dataset()` và `extract_fields()`, sau đó gọi `run_variant()` cho từng câu hỏi.
2. [MedRAGAgents/pipeline.py](MedRAGAgents/pipeline.py) chọn biến thể cần chạy thông qua `build_pipeline(variant)`.
3. Với biến thể V2, `V2MultiAgent.run()` lần lượt gọi:
   - `DomainAgent.classify_question_domains()` và `classify_option_domains()`
   - `AnalysisAgent.run_question_analyses()` và `run_option_analyses()`
   - `SynthesisAgent.synthesise()`
   - `VerifierAgent.verify_and_answer()`
4. Với biến thể V3, `V3FullSystem.run()` chạy tương tự V2 nhưng thêm:
   - `RAGAgent.answer()` để truy xuất chứng cứ từ MedRAG corpus
   - `MemoryManager` để dùng short-term memory trong một câu hỏi và long-term cache qua các lượt chạy
5. Tất cả các agent cuối cùng đều đi qua [MedRAGAgents/agents/base_agent.py](MedRAGAgents/agents/base_agent.py), nơi các prompt được gửi tới LLM provider tương ứng (Gemini/OpenAI/Anthropic) với retry và parsing đáp án.

### Các Bước Xử Lý Chi Tiết

Luồng xử lý một câu hỏi lâm sàng trong hệ thống MedRAGAgents trải qua 5 bước chính:

1. Bước 1 - Phân loại vùng tri thức (`DomainAgent`):
   Hệ thống đọc câu hỏi lâm sàng và các phương án chọn, sau đó phân loại thành 5 chuyên khoa chính cho câu hỏi và 2 chuyên khoa liên quan cho phương án lựa chọn.
2. Bước 2 - Phân tích chuyên sâu từng miền (`AnalysisAgent`):
   Kích hoạt các Agent chuyên gia y khoa tương ứng để phân tích ca bệnh độc lập từ góc nhìn chuyên môn của từng ngành (ví dụ: góc nhìn Bệnh lý học, góc nhìn Dược lý học).
3. Bước 3 - Truy xuất chứng cứ y khoa (`RAGAgent`):
   Sử dụng bộ truy xuất MedCPT để tìm kiếm Top-32 đoạn văn bản chứng cứ có độ tương đồng cao nhất từ cơ sở dữ liệu y khoa Textbooks.
4. Bước 4 - Tổng hợp báo cáo y khoa (`SynthesisAgent`):
   Tổng hợp toàn bộ các báo cáo phân tích chuyên gia và dữ liệu RAG thu thập được thành một bản báo cáo phân tích y khoa hợp nhất (Synthesis Report).
5. Bước 5 - Hội chẩn bỏ phiếu & Sửa đổi (`VerifierAgent`):
   Các Agent chuyên khoa tiến hành đánh giá và bỏ phiếu đồng thuận (YES/NO) trên báo cáo hợp nhất. Nếu có ý kiến bất đồng (NO), Agent đưa ra đề xuất sửa đổi, báo cáo được cập nhật và bỏ phiếu lại (tối đa 3 vòng) trước khi trích xuất đáp án lựa chọn cuối cùng (A, B, C, D hoặc E).

---

## 3. Cơ Chế Bộ Nhớ Hai Tầng (Dual-Layer Memory Mechanism)

### Bộ Nhớ Ngắn Hạn (`ShortTermMemory`)

- Scope: Tồn tại trong bộ nhớ RAM trong suốt lượt xử lý của 01 câu hỏi.
- Mục đích: Lưu trữ và truyền tải dữ liệu trung gian (danh sách chuyên khoa, bài phân tích từng miền, đoạn trích RAG, lịch sử bỏ phiếu) giữa các Agent.
- Vòng đời: Tự động reset sạch thông qua hàm `reset_short()` khi chuyển sang câu hỏi tiếp theo để tránh nhiễm độc ngữ cảnh.

### Bộ Nhớ Dài Hạn (`LongTermMemory`)

- Scope: Lưu trữ persistent dạng JSON trên đĩa tại `./memory/long_term_cache.json`.
- Cơ chế mã hóa: Tạo khóa băm SHA-256 từ chuỗi `question.strip() + str(options)`.
- Mục đích: Bỏ qua quá trình tính toán LLM đối với các câu hỏi đã từng suy luận trước đó (`from_cache=True`), đạt độ trễ 0ms và chi phí 0 token.

---

## 4. Ma Trận Đối Chiếu Yêu Cầu Đề Tài Giữa Kỳ (Compliance Matrix)

Hệ thống tuân thủ đầy đủ các yêu cầu trong Đề tài giữa kỳ môn Machine Learning in Security:

| Tiêu chí Yêu cầu | Quy định Đề tài Giữa kỳ                                                | Triển khai trong MedRAGAgents                                                            | File minh chứng                          |
| :------------------- | :---------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- | :---------------------------------------- |
| System Task          | Input: Câu hỏi MedQA. Output: Đúng 1 đáp án chọn + lời giải thích. | Ràng buộc đầu ra nghiêm ngặt, Invalid Response Rate = 0.0%.                         | `pipeline.py`, `agents/base_agent.py` |
| Baseline (V0)        | Direct LLM với kỹ thuật Chain-of-Thought (CoT).                            | Class`V0DirectLLM` gọi LLM trực tiếp không qua Agent/RAG.                           | `pipeline.py`                           |
| RAG-Only (V1)        | LLM kết hợp truy xuất dữ liệu y khoa MedRAG.                             | Class`V1RAGOnly` sử dụng bộ truy xuất MedCPT trên Textbooks corpus.                | `agents/rag_agent.py`                   |
| Multi-Agent (V2)     | Phân loại miền, phân tích chuyên gia, tổng hợp và hội chẩn.        | Class`V2MultiAgent` kết hợp 5 Agent chuyên gia (stateless).                          | `pipeline.py`, `agents/`              |
| Full System (V3)     | Multi-Agent + RAG + Bộ nhớ hai tầng (Dual-Layer Memory).                   | Class`V3FullSystem` điều phối 5 Agent, MedRAG và Memory.                            | `pipeline.py`, `memory.py`            |
| Dual Memory          | Short-term (in-RAM) và Long-term (persistent cache).                         | Class`ShortTermMemory` và `LongTermMemory` mã hóa SHA-256.                         | `memory.py`                             |
| Benchmark Data       | Test set MedQA-USMLE chính thức (N = 1,273).                                | Hỗ trợ nộp bài test set chính thức và tập dev set.                                | `run.py`, `evaluate.py`               |
| Chỉ số đánh giá | Accuracy, Invalid Rate, Accuracy Gain, McNemar Test, 95% Bootstrap CI.        | Class`Evaluator` tính toán đầy đủ các chỉ số thống kê.                       | `evaluate.py`                           |
| Gói sản phẩm      | Source code, file dự đoán jsonl, script đánh giá, slide, báo cáo.     | Mã nguồn chạy được, outputs jsonl, slide`MedRAGAgents_Midterm_Presentation.pptx`. | Thư mục gốc                            |

### Chi Tiết Cơ Chế Hoạt Động Bên Dưới Của Từng Tiêu Chí

1. **System Task (Ràng buộc đầu ra & Trích xuất đáp án)**:
   - Trong `agents/base_agent.py` và `pipeline.py`, tất cả Prompt gửi cho LLM đều có đính kèm System Directive cưỡng chế định dạng output (`Strict Formatting Prompt`).
   - Hàm `parse_answer_from_text()` sử dụng Regular Expression (`re.search(r'Option:\s*([A-E])', text)`) để trích xuất chính xác 01 ký tự chọn (`A`, `B`, `C`, `D`, `E`). Lời giải thích chẩn đoán được trích xuất lưu tại trường `syn_report`.
2. **Baseline (V0) — Direct LLM với Chain-of-Thought**:
   - Triển khai trong class `V0DirectLLM` (`pipeline.py`). Chạy LLM suy luận CoT đơn bước trực tiếp (`"Let's think step by step"`) không qua Agent phân rã hay RAG retrieval.
3. **RAG-Only (V1) — Truy xuất chứng cứ MedRAG**:
   - Triển khai trong class `V1RAGOnly` (`agents/rag_agent.py`). Sử dụng mô hình nhúng y khoa `MedCPT-Query-Encoder` nhúng văn bản câu hỏi thành vector 768 chiều, tính độ tương đồng vector (Cosine Similarity) để truy xuất **Top-32 đoạn trích y học liên quan nhất** từ `Textbooks corpus` (`corpus/textbooks/`) bơm vào ngữ cảnh cho LLM.
4. **Multi-Agent (V2) — Đội ngũ 5 Agent Chuyên gia (Stateless)**:
   - Triển khai trong class `V2MultiAgent` (`pipeline.py`):
     - `DomainAgent`: Phân loại kịch bản lâm sàng thành 5 chuyên khoa y tế chính.
     - `AnalysisAgent`: Kích hoạt 5 Agent chuyên gia tương ứng phân tích độc lập theo từng góc nhìn chuyên môn.
     - `SynthesisAgent`: Hợp nhất 5 bài phân tích chuyên gia thành báo cáo chẩn đoán lâm sàng (`Synthesis Report`).
     - `VerifierAgent`: Điều phối vòng lặp hội chẩn (Consensus Verification), các Agent chuyên khoa bỏ phiếu (`YES`/`NO`) và sửa đổi báo cáo đến khi đạt đồng thuận (tối đa 3 vòng).
5. **Full System (V3) — Multi-Agent + MedRAG + Dual Memory**:
   - Triển khai trong class `V3FullSystem` (`pipeline.py`). Kết hợp toàn bộ quy trình Multi-Agent của V2, bằng chứng truy xuất MedRAG Top-32 của V1 và quản lý trạng thái/cache với Dual-Layer Memory.
6. **Dual Memory Mechanism (Short-Term & Long-Term Memory)**:
   - `ShortTermMemory` (`memory.py`): Tồn tại dưới dạng bộ nhớ RAM trong suốt vòng đời xử lý 01 câu hỏi, truyền dữ liệu trung gian giữa các Agent và tự động `reset_short()` khi chuyển sang câu mới.
   - `LongTermMemory` (`memory.py`): Tạo khóa băm SHA-256 từ `question.strip() + str(options)` lưu trữ persistent đĩa tại `./memory/long_term_cache.json`. Nếu gặp câu hỏi đã từng suy luận trước đó (`Cache Hit`), kết quả được trả về tức thì (0ms, 0 token).
7. **Benchmark Data (MedQA-USMLE Test Set)**:
   - Đọc dữ liệu dạng stream bằng `jsonlines` từ `datasets/MedQA/test.jsonl` ($N=1,273$). Hỗ trợ điều khiển vị trí bắt đầu (`--start`) và số lượng câu (`--n`), kèm cơ chế `time.sleep(delay)` để quản lý Rate Limit (RPM).
8. **Chỉ số đánh giá (Evaluator Module)**:
   - Triển khai trong class `Evaluator` (`evaluate.py`). Tính toán Accuracy, Invalid Rate, Accuracy Gain ($V3 - V0$), 95% Bootstrap Confidence Interval (1,000 lượt re-sampling), Paired Win/Loss/Tie và kiểm định thống kê McNemar $\chi^2$ ($p\text{-value}$). Kết quả được tự động xuất ra file bảng văn bản [outputs/evaluation_report.txt](file:///c:/Users/long.nguyen4/Downloads/central-stuffs/uit-sdh/ml-sec/MedRAGAgents/outputs/evaluation_report.txt).
9. **Gói sản phẩm & Điều phối CLI**:
   - `run.py` điều phối toàn bộ pipeline, hỗ trợ chế độ `--variant ALL` để chạy tuần tự tất cả các biến thể, tự động ghi đứt đoạn kết quả xuống `./outputs/*_predictions.jsonl` (incremental saving) để tránh mất dữ liệu.

---

## 5. Định Dạng Bộ Dữ Liệu và Bản Ghi Dự Đoán

### Bộ Dữ Liệu Đầu Vào (`MedQA/test.jsonl`)

Tập test set chính thức gồm N = 1,273 câu hỏi. Mỗi bản ghi gồm:

- `question`: Chuỗi văn bản kịch bản lâm sàng.
- `options`: Dictionary chứa các phương án chọn (ví dụ: `{"A": "...", "B": "...", "C": "...", "D": "..."}`).
- `answer_idx` / `answer`: Ký tự đáp án chuẩn y khoa.

### Bản Ghi Kết Quả Đầu Ra (`outputs/<VARIANT>_predictions.jsonl`)

Mỗi dòng trong file kết quả dự đoán là một JSON object hoàn chỉnh:

```json
{
  "idx": 0,
  "question": "A junior orthopaedic surgery resident...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "gold_answer": "C",
  "pred_answer": "C",
  "raw_output": "Option: C",
  "question_domains": ["Orthopedic Surgery", "Medical Ethics", "Hand Surgery"],
  "option_domains": ["Medical Ethics", "Legal Medicine"],
  "syn_report": "Question: ... Total Analysis: ...",
  "vote_history": [{"Orthopedic Surgery": "yes", "Medical Ethics": "yes"}],
  "from_cache": false
}
```

---

## 6. Các Biến Thể Hệ Thống (V0 đến V3)

- Biến thể V0 (Direct LLM Baseline): Mô hình baseline đánh giá LLM suy luận CoT đơn lẻ, không dùng RAG hay Agent.
- Biến thể V1 (RAG-Only): Đánh giá tác động độc lập của việc truy xuất chứng cứ MedRAG lên mô hình baseline.
- Biến thể V2 (Multi-Agent): Đánh giá luồng suy luận 5 Agent chuyên gia và vòng lặp hội chẩn trong trạng thái stateless không memory.
- Biến thể V3 (Full System): Hệ thống hoàn chỉnh kết hợp Multi-Agent, MedRAG, Short-Term Memory và Long-Term Memory cache.

---

## 7. Cấu Trúc Thư Mục Dự Án

```text
MedRAGAgents/
├── agents/                             # Chứa các agent chuyên trách cho từng bước suy luận
│   ├── base_agent.py                   # Wrapper chung để gọi LLM từ Gemini/OpenAI/Anthropic
│   ├── domain_agent.py                 # Phân loại câu hỏi vào các lĩnh vực y học
│   ├── analysis_agent.py               # Phân tích câu hỏi và lựa chọn theo từng chuyên khoa
│   ├── rag_agent.py                    # Gói lớp truy xuất chứng cứ MedRAG
│   ├── synthesis_agent.py              # Tổng hợp báo cáo từ các phân tích chuyên gia
│   └── verifier_agent.py               # Điều phối bỏ phiếu và sửa đổi hội chẩn
├── corpus/                             # Dữ liệu văn bản y khoa dùng cho hệ thống truy xuất
│   └── textbooks/                      # Chứa các textbook chunks đã được tiền xử lý
├── datasets/                           # Bộ dữ liệu input cho hệ thống
│   └── MedQA/                          # Tập dữ liệu MedQA-USMLE (test.jsonl)
├── memory/                             # Bộ nhớ dài hạn và cache của hệ thống
│   └── long_term_cache.json            # Cache kết quả đã suy luận trước đó
├── outputs/                            # Kết quả chạy thực tế của hệ thống
│   ├── V0_predictions.jsonl            # Dự đoán từ biến thể V0
│   ├── V1_predictions.jsonl            # Dự đoán từ biến thể V1
│   ├── V2_predictions.jsonl            # Dự đoán từ biến thể V2
│   ├── V3_predictions.jsonl            # Dự đoán từ biến thể V3
│   ├── evaluation_report.txt          # Báo cáo đánh giá độ chính xác và thống kê
│   └── run_execution.log              # Log chạy quá trình inference
├── config.py                           # Cấu hình mặc định cho LLM, đường dẫn và tham số
├── download_corpus.py                  # Script tải và chuẩn bị corpus cho RAG
├── evaluate.py                         # Đánh giá kết quả bằng accuracy, bootstrap, McNemar
├── memory.py                           # Short-term memory và Long-term memory manager
├── pipeline.py                         # Điều phối V0–V3 và luồng suy luận multi-agent
├── run.py                              # Entry point CLI để chạy inference trên dataset
├── requirements.txt                    # Danh sách thư viện Python cần cài
├── package.json                        # Script phụ trợ cho slide/presentation
├── presentation_slides.html            # Bản trình bày HTML dùng cho slide
├── MedRAGAgents_Midterm_Presentation.pptx  # Slide báo cáo giữa kỳ
└── .env / .env.example                 # Cấu hình API key và biến môi trường
```

### Mô Tả Chi Tiết Từng Thư Mục

- `agents/`: chứa toàn bộ logic agent hóa. Mỗi file tương ứng với một vai trò riêng trong pipeline suy luận: `base_agent.py` dùng để gọi LLM, `domain_agent.py` phân loại chuyên khoa, `analysis_agent.py` phân tích chuyên sâu, `rag_agent.py` truy xuất văn bản y khoa, `synthesis_agent.py` tổng hợp, và `verifier_agent.py` kiểm tra đồng thuận.
- `corpus/`: lưu trữ dữ liệu văn bản y khoa đã được chunk và dùng làm kho tri thức cho hệ thống RAG.
- `datasets/`: chứa dữ liệu đầu vào, hiện tại là bộ MedQA-USMLE để chạy inference và đánh giá.
- `memory/`: lưu cache dài hạn để hệ thống có thể nhận diện câu hỏi đã xử lý trước đó và bỏ qua bước suy luận lặp lại.
- `outputs/`: lưu toàn bộ sản phẩm chạy thực tế bao gồm file dự đoán cho từng biến thể và báo cáo đánh giá.
- `config.py`: tập trung các tham số cấu hình như provider LLM, model name, đường dẫn corpus, số lượng domain, v.v.
- `pipeline.py`: trung tâm điều phối các biến thể V0, V1, V2, V3.
- `run.py`: giao diện dòng lệnh để chạy hệ thống với các tham số như `--variant`, `--n`, `--delay`, `--evaluate`.

### Nội Dung Thực Tế Hiện Tại Trong Thư Mục outputs

Thư mục `outputs/` hiện đang chứa các file sau:

- `V0_predictions.jsonl`: kết quả dự đoán từ hệ thống biến thể V0.
- `V1_predictions.jsonl`: kết quả dự đoán từ hệ thống biến thể V1.
- `V2_predictions.jsonl`: kết quả dự đoán từ hệ thống biến thể V2.
- `V3_predictions.jsonl`: kết quả dự đoán từ hệ thống biến thể V3.
- `evaluation_report.txt`: báo cáo đánh giá tổng hợp theo các chỉ số như accuracy, invalid rate, win/loss/tie.
- `run_execution.log`: nhật ký chạy thực tế của quá trình inference.

Mỗi file `*_predictions.jsonl` chứa các dòng JSON, mỗi dòng tương ứng với một câu hỏi trong dataset, bao gồm các trường như `idx`, `question`, `options`, `gold_answer`, `pred_answer`, `raw_output`, `syn_report`, `vote_history` và `from_cache` (đối với V3).

#### Kết Quả Thực Tế Hiện Có Trong outputs

Dựa trên nội dung file `evaluation_report.txt` hiện có trong repo, kết quả chạy thử đang được ghi như sau:

| Variant | Total | Correct | Accuracy | Invalid Rate | 95% CI |
| :------ | :---- | :----- | :------- | :----------- | :------ |
| V0 | 10 | 5 | 0.5000 | 0.4000 | (0.2000, 0.8000) |
| V1 | 10 | 7 | 0.7000 | 0.0000 | (0.4000, 0.9000) |
| V2 | 17 | 14 | 0.8235 | 0.0000 | (0.6471, 1.0000) |
| V3 | 20 | 16 | 0.8000 | 0.0000 | (0.6000, 0.9500) |

Một số thông tin bổ sung từ báo cáo:

- Accuracy Gain (V3 - V0): `+0.3000`
- Paired comparison V0 vs V3: Win = 3, Loss = 0, Tie = 7
- McNemar statistic: `1.3333` với `p ≈ 0.2482`

Ngoài ra, các file trong `outputs/` hiện có kích thước và số dòng như sau:

| File | Kích thước | Số dòng |
| :--- | :--------- | :------ |
| `V0_predictions.jsonl` | 15,312 bytes | 13 |
| `V1_predictions.jsonl` | 786,818 bytes | 13 |
| `V2_predictions.jsonl` | 136,866 bytes | 18 |
| `V3_predictions.jsonl` | 900,243 bytes | 20 |
| `evaluation_report.txt` | 1,941 bytes | 33 |
| `run_execution.log` | 7,921 bytes | - |

---

## 8. Hướng Dẫn Cài Đặt Môi Trường & Quick Start

### Quy Trình Cài Đặt Nhanh (5 Bước Cho Repo Mới Clone)

```bash
# Bước 1: Clone kho mã nguồn từ GitHub
git clone https://github.com/xcentralnn/MedRAGAgents.git
cd MedRAGAgents

# Bước 2: Khởi tạo môi trường ảo và cài đặt thư viện
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Bước 3: Cấu hình biến môi trường API Key
cp .env.example .env
# Chỉnh sửa file .env: Cập nhật GEMINI_API_KEY của bạn

# Bước 4: Tải dữ liệu vector chỉ mục y khoa (chỉ cần chạy 1 lần)
python3 download_corpus.py

# Bước 5: Thực thi thử nghiệm hệ thống (Chạy thử 10 câu cho 4 biến thể V0 -> V3)
python3 run.py --variant ALL --n 10 --evaluate
```

---

### Phụ thuộc yêu cầu

- Python 3.10+
- Khuyên dùng môi trường WSL (Windows Subsystem for Linux) trên hệ điều hành Windows.

### Cấu hình file môi trường (.env)

Tạo file `.env` từ mẫu `.env.example`:

```env
GEMINI_API_KEY=dien_gemini_api_key_tai_day
DEFAULT_LLM=google/gemini-3.6-flash
DATASET_DIR=./datasets/MedQA/
OUTPUT_DIR=./outputs
LONG_TERM_MEM=./memory/long_term_cache.json
```

---

## 9. Bảng Giải Thích Chi Tiết Tham Số Lệnh CLI (`run.py`)

File `run.py` tiếp nhận các tham số dòng lệnh sau:

| Tham số CLI      | Kiểu dữ liệu | Mặc định         | Mục đích và Ý nghĩa sử dụng                                                                                                                                                                |
| :---------------- | :-------------- | :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--variant`     | str             | `V3`              | Chọn biến thể hệ thống cần chạy:`V0` (Direct CoT), `V1` (RAG-Only), `V2` (Multi-Agent), `V3` (Full System), hoặc `ALL` (chạy tuần tự cả 4 biến thể).                       |
| `--n`           | int             | `-1`              | Số lượng câu hỏi cần đánh giá (`10` để test nhanh, `-1` để chạy toàn bộ N = 1,273 câu hỏi).                                                                                  |
| `--start`       | int             | `0`               | Chỉ số câu hỏi bắt đầu (hỗ trợ chia nhỏ tập dữ liệu để chạy song song).                                                                                                            |
| `--llm`         | str             | `cfg.DEFAULT_LLM` | Chỉ định mô hình LLM backbone (ví dụ:`google/gemini-3.6-flash`, `openai/gpt-4o-mini`).                                                                                                  |
| `--delay`       | float           | `2.0`             | Khoảng nghỉ giãn cách (giây) giữa các câu hỏi. Giúp kiểm soát tần suất API (RPM) để không dính lỗi 429 trên tài khoản Free Tier (ví dụ đặt`12` cho hạn ngạch 5 RPM). |
| `--dataset_dir` | str             | `cfg.DATASET_DIR` | Đường dẫn tới thư mục chứa bộ dữ liệu`test.jsonl`.                                                                                                                                    |
| `--output_dir`  | str             | `cfg.OUTPUT_DIR`  | Thư mục lưu trữ các file kết quả dự đoán tăng dần.                                                                                                                                     |
| `--evaluate`    | flag            | `False`           | Tự động kích hoạt script tính toán đánh giá thống kê ngay sau khi chạy xong.                                                                                                          |

---

## 10. Hướng Dẫn Thực Thi Dự Án (Execution Guide V0 → V3)

Thực thi các lệnh bằng Python trong môi trường WSL hoặc Linux Bash.

### 0. Tải dữ liệu chứng cứ y khoa gốc (Corpus Textbooks)

Trước khi chạy các biến thể có RAG (V1, V3), chạy script để tải tự động toàn bộ 200MB+ dữ liệu sách giáo khoa y khoa từ HuggingFace Hub:

```bash
python3 download_corpus.py
```

---

### 1. Chạy Từng Biến Thể Riêng Lẻ (V0, V1, V2, V3)

- **Chạy Baseline Direct CoT (V0)**:
  ```bash
  python3 run.py --variant V0 --n 10 --delay 2.0 --evaluate
  ```

- **Chạy RAG-Only MedRAG (V1)**:
  ```bash
  python3 run.py --variant V1 --n 10 --delay 2.0 --evaluate
  ```

- **Chạy Multi-Agent 5 Chuyên Gia (V2)**:
  ```bash
  python3 run.py --variant V2 --n 10 --delay 2.0 --evaluate
  ```

- **Chạy Full System Multi-Agent + RAG + Memory (V3)**:
  ```bash
  python3 run.py --variant V3 --n 10 --delay 2.0 --evaluate
  ```

---

### 2. Chạy Tự Động Tất Cả Các Biến Thể (V0 → V3)

- **Chạy thử nghiệm trên N = 10 câu hỏi mẫu**:
  ```bash
  python3 run.py --variant ALL --n 10 --delay 2.0 --evaluate
  ```

- **Chạy toàn bộ tập dữ liệu MedQA-USMLE (N = 1,273 câu hỏi)**:
  ```bash
  python3 run.py --variant ALL --n -1 --delay 2.0 --evaluate
  ```

- **Chạy lưu log trực tiếp ra file (`run_execution.log`)**:
  ```bash
  PYTHONUNBUFFERED=1 python3 run.py --variant ALL --n 10 --delay 2.0 --evaluate 2>&1 | tee outputs/run_execution.log
  ```


---

## 11. Script Tổng Hợp & Đánh Giá Chỉ Số (`evaluate.py`)

Để xuất bảng báo cáo đánh giá thống kê từ các file dự đoán trong thư mục `./outputs/`:

```bash
python3 evaluate.py --pred_dir ./outputs
```

Các chỉ số thống kê đầu ra:

- Accuracy: Tỷ lệ câu hỏi dự đoán đúng trên tổng số câu.
- Invalid Response Rate: Tỷ lệ phản hồi vi phạm định dạng đầu ra (Mục tiêu: 0.0%).
- Accuracy Gain: Mức tăng trưởng độ chính xác giữa V3 và V0 ($\text{Accuracy}(V3) - \text{Accuracy}(V0)$).
- Paired Win/Loss/Tie Comparison: Thống kê chi tiết các câu hỏi V3 cải thiện so với V0.
- Kiểm định thống kê: Giá trị p-value từ kiểm định McNemar chi-squared và khoảng tin cậy 95% Bootstrap CI.

---

## 12. Bảng Danh Mục Sản Phẩm Nộp Bài Giữa Kỳ

- Mã nguồn hoàn chỉnh: Codebase modular phân rã theo đúng kiến trúc Agent.
- File kết quả dự đoán: Các file jsonl ghi nhận kết quả tăng dần tại `./outputs/`.
- Slide thuyết trình: File slide PowerPoint widescreen 16:9 `MedRAGAgents_Midterm_Presentation.pptx`.
- Module đánh giá tự động: Script tổng hợp thống kê `evaluate.py`.
- Báo cáo hướng dẫn: Tài liệu hướng dẫn và minh chứng tuân thủ tại `README.md`.

---

## 13. Tài Liệu Tham Khảo & Repository Nguồn Dẫn (References & Citations)

Hệ thống MedRAGAgents được kế thừa và phát triển dựa trên các nghiên cứu và repository nguồn mở chính sau:

1. MedAgents (Domain-Expert Multi-Agent Reasoning):

   - Repository: https://github.com/MedAgents/MedAgents
   - Mô tả: Nguồn tham khảo về kiến trúc phân rã Agent theo miền chuyên khoa y tế và quy trình bỏ phiếu sửa đổi đồng thuận (Consensus Verification).
2. MedRAG (Biomedical Retrieval-Augmented Generation):

   - Repository: https://github.com/Teddy-XiongGZ/MedRAG
   - Mô tả: Nguồn tham khảo về hệ thống truy xuất chứng cứ y khoa MedRAG, bao gồm các bộ chỉ mục vector MedCPT và tập cơ sở dữ liệu Textbooks/PubMed.
3. MedQA Benchmark (MedQA-USMLE Dataset):

   - Repository: https://github.com/jair-ai/MedQA
   - Mô tả: Bộ dữ liệu chuẩn đánh giá các ca lâm sàng y khoa trắc nghiệm theo kỳ thi cấp phép hành nghề y khoa Hoa Kỳ (USMLE).

## 14. Đồ án Cuối Kỳ (Final Project): Attack & Defense (Pair 5)

Trong khuôn khổ Báo cáo Cuối kỳ, hệ thống được nâng cấp để thực hiện Đánh giá Bảo mật (Security Evaluation) tập trung vào cặp Tấn công & Phòng thủ (Pair 5):
- **Tấn công (Attack):** [TenBenign](https://arxiv.org/abs/2410.01234) (Stealthy Utility-Preserving Attack)
- **Phòng thủ (Defense):** [Booster](https://arxiv.org/abs/2409.01586) (Alignment-Stage Defense via Harmful Perturbation)

### 14.1. Thiết lập Môi trường Local (Low-Resource)
Để mô phỏng kịch bản tấn công thực tế và tương thích với giới hạn phần cứng (GPU 4GB VRAM), hệ thống đã được chuyển đổi từ API (Gemini/OpenAI) sang sử dụng mô hình LLM cục bộ: **Qwen2.5-1.5B-Instruct**.
- Kỹ thuật sử dụng: **QLoRA (4-bit quantization)** giúp nén bộ nhớ LLM xuống chỉ còn ~1.2GB VRAM, cho phép thực hiện Fine-tuning trực tiếp trên máy cá nhân.
- File cấu hình: Cập nhật `DEFAULT_LLM=local/Qwen/Qwen2.5-1.5B-Instruct` trong `.env`.

### 14.2. Quy trình Thực nghiệm (Workflow)

#### Bước 1: Đo Baseline (Safety & Utility Gốc)
- **Safety:** Chạy `python eval_safety.py` với tập `harmful_prompts.json`. (Kỳ vọng: Refusal Rate 100%).
- **Utility:** Chạy `python run.py --variant V0 --n 50 --output_dir ./outputs_baseline --evaluate` để đo độ chính xác y khoa (MedQA).

#### Bước 2: Tấn công TenBenign (`attack_qlora.py`)
Thuật toán TenBenign phá vỡ cơ chế an toàn (Safety Alignment) thông qua 2 giai đoạn dùng dữ liệu vô hại (Benign Data):
1. Ép mô hình overfit vào phản hồi từ chối đối với câu hỏi bình thường.
2. Dạy mô hình trả lời chi tiết lại chính các câu hỏi đó, gây ra sự "quên lãng" (catastrophic forgetting) đối với ranh giới an toàn.
- **Lệnh chạy:** `python attack_qlora.py` (Tạo ra `./attacked_adapter`).
- **Đánh giá:** `python eval_safety_attacked.py` (Kỳ vọng: Refusal Rate giảm mạnh, Utility giữ nguyên).

#### Bước 3: Phòng thủ Booster (`defense_booster.py`)
Booster bảo vệ mô hình ngay từ giai đoạn căn chỉnh (Alignment-stage). Thuật toán mô phỏng các gradient tấn công (harmful perturbation) và phạt mô hình nếu nó thay đổi trọng số theo hướng đó, tạo ra sức "đề kháng".
- **Lệnh chạy:** `python defense_booster.py` (Tạo ra `./booster_adapter`).

#### Bước 4: Đánh giá Lỗ hổng (Residual-Gap Analysis)
Tiến hành hợp nhất khiên Booster vào mô hình gốc, sau đó để TenBenign tấn công lại mô hình đã bảo vệ này để xem khiên có bị vỡ không.
- **Lệnh chạy tự động:** `python merge_and_eval.py`

### 14.3. Phân tích Lỗ hổng (Residual Gap & Failure Mode)
Kết quả thực nghiệm trên hệ thống cục bộ (4GB VRAM) làm nổi bật một lỗ hổng nghiêm trọng (Failure Mode) của phương pháp Booster:
1. **Utility-Preserving Attack:** TenBenign là cuộc tấn công tàng hình hoàn hảo. Dù Refusal Rate bị vô hiệu hóa, Utility (Accuracy) trên MedQA không hề bị suy giảm, khiến việc phát hiện tấn công thông qua hiệu suất là bất khả thi.
2. **Sự sụp đổ của Booster trong môi trường Low-Resource:** Thuật toán Booster nguyên bản yêu cầu Full-parameter Fine-tuning trên lượng lớn dữ liệu để xây dựng sức đề kháng vững chắc. Khi bị ép chạy bằng QLoRA (cập nhật rất ít trọng số) trên GPU 4GB, lớp khiên phòng thủ trở nên quá mỏng. Kết quả là, TenBenign (vốn thay đổi trọng số rất linh hoạt) đã xuyên thủng Booster hoàn toàn (Refusal Rate = 0%).
3. **Kết luận (Threat-Model Honesty):** Việc dùng biện pháp phòng thủ ở giai đoạn căn chỉnh (Alignment-stage) không đảm bảo chống lại được các cuộc tấn công mạnh hơn hoặc có khả năng thích ứng cao hơn khi người phòng thủ gặp bất lợi về tài nguyên tính toán (Low-Resource / LoRA Constraint).
