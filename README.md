# Legislation GraphRAG

## Giới Thiệu Web

Legislation GraphRAG là web tra cứu và hỏi đáp văn bản pháp luật Việt Nam. Người dùng có thể mở văn bản trên giao diện, đọc nội dung tài liệu và đặt câu hỏi để hệ thống trả lời dựa trên ngữ cảnh được truy xuất từ kho văn bản.

Web tập trung vào các chức năng chính:

- Hiển thị 5 văn bản pháp luật ngẫu nhiên.
- Mở văn bản trong khung xem tài liệu.
- Phóng to, thu nhỏ và reset mức zoom khi xem tài liệu.
- Nhập câu hỏi pháp luật và nhận câu trả lời từ hệ thống RAG.
- Hiển thị hội thoại, số nguồn được dùng và thời gian xử lý.

## Hướng Dẫn Dùng Web

Chạy các lệnh sau trong terminal từ thư mục dự án.

Tạo môi trường Python và cài thư viện:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Tạo file cấu hình môi trường:

```bash
copy .env.example .env
```

Mở `.env` và điền các giá trị cần thiết như `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.

Nếu dùng Qdrant local, chạy:

```bash
docker compose up -d qdrant
```

Đưa dữ liệu vào Qdrant:

```bash
python scripts/ingest.py --normalize
```

Lệnh ingest sẽ chuẩn hóa dữ liệu từ `data/raw` sang `data/normalized`, chia chunk, tạo embedding và upsert vào Qdrant.

Sau đó chạy web:

```bash
set PYTHONIOENCODING=utf-8
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Khi server khởi động, hệ thống sẽ tự build BM25 nếu chưa có và tự load graph snapshot. Nếu chưa có snapshot, hệ thống sẽ tự build graph từ `data/normalized` rồi lưu snapshot vào `outputs/runtime/runtime_graph_snapshot.json`.

Mở web tại:

```text
http://127.0.0.1:8000/
```

Cách sử dụng:

1. Xem danh sách 5 văn bản pháp luật ngẫu nhiên ở cột trái.
2. Bấm `Làm mới` nếu muốn đổi danh sách văn bản.
3. Bấm vào tên một văn bản để mở nội dung trong khung xem ở giữa.
4. Dùng `Thu nhỏ`, `Phóng to`, `Reset` để điều chỉnh khung xem.
5. Nhập câu hỏi vào ô `Câu hỏi` trong phần `Tra cứu pháp luật`.
6. Bấm `Gửi câu hỏi`.
7. Xem câu trả lời trong phần `Hội thoại`.

Gợi ý khi đặt câu hỏi:

- Nếu hỏi theo một văn bản cụ thể, nên kèm tên hoặc tóm tắt tiêu đề văn bản.
- Với môi trường CPU-only, thời gian phản hồi có thể khoảng 30 giây hoặc hơn.

## Evaluation

Evaluation được chạy bằng terminal thông qua file `run_all`. Lệnh này tự động chạy toàn bộ quy trình đánh giá: validate benchmark, chuẩn bị dữ liệu JSONL, thu RAG outputs, chạy RAGAS-lite và in báo cáo kết quả.

Chạy evaluation:

```bash
python -m src.evaluation.run_all --input data/eval/legal_rag_benchmark_25.json
```

Kết quả chính được lưu tại:

```text
data/eval/ragas_lite_results.json
```
