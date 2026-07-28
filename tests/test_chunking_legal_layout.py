from src.rag.chunking_legal import legal_chunk, parse_legal_document


def _types(parsed):
    return [node.node_type for node in parsed.nodes]


def test_parse_roman_decimal_appendix_table_and_footer():
    text = """
KẾ HOẠCH
Về kiểm tra thủ tục hành chính
I. MỤC ĐÍCH, YÊU CẦU
1. Mục đích chung
1.1. Kiểm tra hồ sơ
+ Thành phần hồ sơ được rà soát theo danh mục.
II. NHIỆM VỤ
1. Nhiệm vụ thứ nhất
PHỤ LỤC
Danh mục thủ tục
STT | Tên thủ tục | Ghi chú
1 | Cấp phép A | Còn hiệu lực
Nơi nhận:
- Lưu: VT
NGUYỄN VĂN A
""".strip()

    parsed = parse_legal_document(text, fallback_doc_name="ke_hoach_test")
    node_types = _types(parsed)
    assert "roman_section" in node_types
    assert "decimal_item" in node_types
    assert "bullet" in node_types
    assert "appendix" in node_types
    assert "table_row" in node_types

    node_ids = [node.node_id for node in parsed.nodes]
    assert len(node_ids) == len(set(node_ids))
    body_text = "\n".join(node.text for node in parsed.nodes if node.node_type != "document")
    assert "Nơi nhận" not in body_text
    assert "NGUYỄN VĂN A" not in body_text

    chunks = legal_chunk(text, fallback_doc_name="ke_hoach_test")
    table_chunks = [c for c in chunks if c["metadata"].get("node_type") == "table_row"]
    assert table_chunks
    assert "PHỤ LỤC" in table_chunks[0]["metadata"].get("path_title", "")


def test_numbering_reset_keeps_unique_ids_and_distinct_parents():
    text = """
KẾ HOẠCH
I. NHÓM MỘT
1. Nội dung số một
II. NHÓM HAI
1. Nội dung số một lặp lại
""".strip()

    parsed = parse_legal_document(text, fallback_doc_name="reset_test")
    items = [node for node in parsed.nodes if node.node_type == "item"]
    assert len(items) == 2
    assert items[0].node_id != items[1].node_id
    assert items[0].parent_id != items[1].parent_id


def test_decision_attachment_resets_article_before_roman_sections():
    text = """
QUYẾT ĐỊNH
PHÊ DUYỆT ĐỀ ÁN CHUYỂN ĐỔI SỐ TRONG HOẠT ĐỘNG TỐ TỤNG HÌNH SỰ
THỦ TƯỚNG CHÍNH PHỦ
Căn cứ Luật Tổ chức Chính phủ ngày 18 tháng 02 năm 2025;
QUYẾT ĐỊNH:
Điều 1. Phê duyệt kèm theo Quyết định này Đề án.
Điều 2. Quyết định này có hiệu lực kể từ ngày ký ban hành.
Điều 3. Bộ Công an chịu trách nhiệm thi hành Quyết định này.
ĐỀ ÁN
CHUYỂN ĐỔI SỐ TRONG HOẠT ĐỘNG TỐ TỤNG HÌNH SỰ
(Kèm theo Quyết định số 422/QĐ-TTg ngày 10 tháng 3 năm 2026 của Thủ tướng Chính phủ)
I. QUAN ĐIỂM XÂY DỰNG ĐỀ ÁN
1. Đề án được xây dựng trên cơ sở quán triệt chủ trương của Đảng.
2. Việc số hóa hồ sơ phải được triển khai đồng bộ.
II. MỤC TIÊU
1. Mục tiêu tổng quát
Nâng cao hiệu quả quản lý hồ sơ điện tử.
""".strip()

    parsed = parse_legal_document(text, fallback_doc_name="422_QD-TTg_697174")
    attachment = next(node for node in parsed.nodes if node.node_type == "attachment")
    quan_diem = next(node for node in parsed.nodes if node.node_type == "roman_section" and "QUAN ĐIỂM" in node.label)
    items = [node for node in parsed.nodes if node.parent_id == quan_diem.node_id and node.node_type == "item"]
    article3 = next(node for node in parsed.nodes if node.article == "Điều 3" and node.node_type == "article")

    assert parsed.header.date_raw == "10/03/2026"
    assert parsed.header.year == 2026
    assert quan_diem.parent_id == attachment.node_id
    assert len(items) == 2
    assert items[0].path_title.startswith("ĐỀ ÁN > I. QUAN ĐIỂM XÂY DỰNG ĐỀ ÁN")
    assert "QUAN ĐIỂM XÂY DỰNG ĐỀ ÁN" not in article3.path_title
