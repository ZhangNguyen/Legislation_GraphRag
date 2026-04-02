from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

DOC_TYPES = [
    "BỘ LUẬT",
    "LUẬT",
    "PHÁP LỆNH",
    "NGHỊ QUYẾT",
    "NGHỊ ĐỊNH",
    "QUYẾT ĐỊNH",
    "CHỈ THỊ",
    "CÔNG ĐIỆN",
    "THÔNG TƯ",
    "THÔNG TƯ LIÊN TỊCH",
    "KẾ HOẠCH",
    "CHƯƠNG TRÌNH",
    "ĐỀ ÁN",
]

TITLE_STOP_PATTERNS = [
    re.compile(r"^Căn cứ\b", re.IGNORECASE),
    re.compile(r"^Theo đề nghị\b", re.IGNORECASE),
    re.compile(r"^Xét\b", re.IGNORECASE),
    re.compile(r"^Điều\s+\d+", re.IGNORECASE),
    re.compile(r"^Chương\s+[IVXLC\d]+", re.IGNORECASE),
    re.compile(r"^Mục\s+\d+", re.IGNORECASE),
    re.compile(r"^Phần\s+[IVXLC\d]+", re.IGNORECASE),
    re.compile(r"^\d+\.\s+"),
    re.compile(r"^[a-zđ]\)\s+", re.IGNORECASE),
]

AGENCY_SKIP_EXACT = {
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
    "ĐỘC LẬP - TỰ DO - HẠNH PHÚC",
    "-------",
    "---------------",
    "---",
    "|",
}

NUMBER_LINE_RE = re.compile(r"^\s*Số\s*:?\s*(.+?)\s*$", re.IGNORECASE)
STRICT_DOC_NUMBER_RE = re.compile(r"\b\d{1,4}\s*/\s*\d{4}\s*/\s*[A-ZĐ\-]+\b", re.IGNORECASE)
DATE_RE = re.compile(
    r"(?:(?P<place>[A-ZÀ-Ỵa-zà-ỵ\s\.\-]+)\s*,\s*)?ngày\s*(?P<day>\d{1,2})\s*tháng\s*(?P<month>\d{1,2})\s*năm\s*(?P<year>\d{4})",
    re.IGNORECASE,
)
DATE_SLASH_RE = re.compile(r"\b(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})\b")
ARTICLE_START_RE = re.compile(r"^Điều\s+\d+", re.IGNORECASE)
CHAPTER_START_RE = re.compile(r"^Chương\s+[IVXLC\d]+", re.IGNORECASE)
ITEM_START_RE = re.compile(r"^\d+\.\s+")
POINT_START_RE = re.compile(r"^[a-zđ]\)\s+", re.IGNORECASE)

# Bắt tên file kiểu:
# 01_2026_TT-NHNN_697839
# 04_2026_TT-BYT_697720
# 01_2026_QD-CTUBND_697076
# 01_CD-BTC_696881
FILENAME_DOC_NUMBER_FULL_RE = re.compile(
    r"(?i)\b(?P<num>\d{1,4})_(?:(?P<year>\d{4})_)?(?P<code>[A-ZĐ]{1,6}(?:-[A-ZĐ0-9]{1,20})+)\b"
)

TITLE_ROLE_STOP_RE = re.compile(
    r"^(BỘ TRƯỞNG|THỨ TRƯỞNG|THỐNG ĐỐC|PHÓ THỐNG ĐỐC|CHỦ TỊCH|PHÓ CHỦ TỊCH|CỤC TRƯỞNG|PHÓ CỤC TRƯỞNG|VIỆN TRƯỞNG|CHÁNH ÁN|TỔNG KIỂM TOÁN)\b",
    re.IGNORECASE,
)

ACTION_LINE_RE = re.compile(
    r"^(?P<role>BỘ TRƯỞNG|THỨ TRƯỞNG|THỐNG ĐỐC|PHÓ THỐNG ĐỐC|CHỦ TỊCH|PHÓ CHỦ TỊCH|CỤC TRƯỞNG|PHÓ CỤC TRƯỞNG|VIỆN TRƯỞNG|CHÁNH ÁN|TỔNG KIỂM TOÁN|HỘI ĐỒNG NHÂN DÂN)\s+(?P<body>.+?)(?:\s+ban hành\b|\s+điện\s*:?)",
    re.IGNORECASE,
)

AGENCY_LINE_RE = re.compile(
    r"^(NGÂN HÀNG NHÀ NƯỚC|HỘI ĐỒNG NHÂN DÂN|ỦY BAN NHÂN DÂN|TÒA ÁN NHÂN DÂN|VIỆN KIỂM SÁT|KIỂM TOÁN NHÀ NƯỚC|BỘ\b|CHÍNH PHỦ\b|THỦ TƯỚNG\b)",
    re.IGNORECASE,
)


def _norm_space(text: str) -> str:
    return " ".join((text or "").replace("\ufeff", " ").split()).strip()


def _slugify(text: str) -> str:
    raw = _norm_space(text).lower()
    raw = re.sub(r"[^a-z0-9à-ỹ]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "document"


def _clean_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text or "").splitlines():
        line = _norm_space(raw)
        if line:
            out.append(line)
    return out


def _is_upper_like(text: str) -> bool:
    raw = _norm_space(text)
    letters = [ch for ch in raw if ch.isalpha()]
    if len(letters) < 4:
        return False
    uppers = sum(1 for ch in letters if ch.isupper())
    return uppers / max(1, len(letters)) >= 0.72


def _looks_like_doc_type(line: str) -> bool:
    return _norm_space(line).upper() in DOC_TYPES


def _clean_doc_number(text: str) -> str:
    raw = _norm_space(str(text or ""))
    if not raw:
        return ""
    m = STRICT_DOC_NUMBER_RE.search(raw.upper())
    if not m:
        return ""
    val = m.group(0).upper()
    val = re.sub(r"\s*/\s*", "/", val)
    val = re.sub(r"\s+", "", val)
    return val


def _doc_number_from_filename(fallback_name: str) -> str:
    fb = _norm_space(fallback_name).upper()
    if not fb:
        return ""

    m = FILENAME_DOC_NUMBER_FULL_RE.search(fb)
    if not m:
        return ""

    num = str(int(m.group("num")))
    year = (m.group("year") or "").strip()
    code = (m.group("code") or "").upper().strip()

    if year:
        return f"{num}/{year}/{code}"
    return f"{num}/{code}"


def _pick_doc_number(lines: List[str], fallback_name: str = "") -> str:
    # Ưu tiên đúng dòng Số:
    for line in lines[:50]:
        m = NUMBER_LINE_RE.match(line)
        if m:
            cleaned = _clean_doc_number(m.group(1))
            if cleaned:
                return cleaned

    # Fallback từ tên file
    return _doc_number_from_filename(fallback_name)


def _pick_date(lines: List[str], fallback_name: str = "") -> Tuple[str, int, str]:
    for line in lines[:60]:
        m = DATE_RE.search(line)
        if m:
            day = int(m.group("day"))
            month = int(m.group("month"))
            year = int(m.group("year"))
            place = _norm_space(m.group("place") or "")
            return f"{day:02d}/{month:02d}/{year:04d}", year, place

    for line in lines[:60]:
        m = DATE_SLASH_RE.search(line)
        if m:
            day = int(m.group("day"))
            month = int(m.group("month"))
            year = int(m.group("year"))
            return f"{day:02d}/{month:02d}/{year:04d}", year, ""

    fb = _norm_space(fallback_name)
    m = DATE_SLASH_RE.search(fb)
    if m:
        day = int(m.group("day"))
        month = int(m.group("month"))
        year = int(m.group("year"))
        return f"{day:02d}/{month:02d}/{year:04d}", year, ""

    return "", 0, ""


def _find_doc_type(lines: List[str]) -> Tuple[str, Optional[int]]:
    for idx, line in enumerate(lines[:40]):
        clean = _norm_space(line).upper()
        if clean in DOC_TYPES:
            return clean, idx
    return "", None


def _is_title_stop(line: str) -> bool:
    raw = _norm_space(line)
    if not raw:
        return True
    if NUMBER_LINE_RE.match(raw):
        return True
    if DATE_RE.search(raw) or DATE_SLASH_RE.search(raw):
        return True
    if TITLE_ROLE_STOP_RE.match(raw):
        return True
    for pat in TITLE_STOP_PATTERNS:
        if pat.search(raw):
            return True
    return False


def _find_body_start_index(lines: List[str], start_idx: int = 0) -> int:
    for idx in range(start_idx, min(len(lines), 160)):
        line = _norm_space(lines[idx])
        if ARTICLE_START_RE.match(line) or CHAPTER_START_RE.match(line):
            return idx
    return min(len(lines), max(start_idx, 0))


def _pick_title_block(lines: List[str], doc_type_idx: Optional[int]) -> Tuple[str, str, int]:
    if doc_type_idx is None:
        return "", "", 0

    title_lines: List[str] = []
    stop_idx = doc_type_idx + 1

    for idx in range(doc_type_idx + 1, min(len(lines), doc_type_idx + 12)):
        line = _norm_space(lines[idx])

        if _is_title_stop(line):
            stop_idx = idx
            break

        if _is_upper_like(line) or line.upper().startswith(("VỀ VIỆC", "QUY ĐỊNH", "BAN HÀNH", "SỬA ĐỔI", "BỔ SUNG", "BÃI BỎ")):
            title_lines.append(line)
            stop_idx = idx + 1
            continue

        if title_lines:
            stop_idx = idx
            break

        stop_idx = idx
        break

    title_block = "\n".join(title_lines).strip()
    official_title = _norm_space(" ".join(title_lines))
    body_start_index = _find_body_start_index(lines, start_idx=stop_idx)
    return title_block, official_title, body_start_index


def _extract_agency_from_action_line(line: str) -> str:
    raw = _norm_space(line)
    if raw.lower().startswith(("căn cứ", "theo đề nghị", "xét")):
        return ""
    m = ACTION_LINE_RE.match(raw)
    if not m:
        return ""
    role = _norm_space(m.group("role"))
    body = _norm_space(m.group("body"))
    return _norm_space(f"{role} {body}")


def _looks_like_agency_line(line: str) -> bool:
    raw = _norm_space(line)
    up = raw.upper()

    if not raw or up in AGENCY_SKIP_EXACT:
        return False
    if raw.lower().startswith(("căn cứ", "theo đề nghị", "xét")):
        return False
    if NUMBER_LINE_RE.match(raw) or DATE_RE.search(raw) or DATE_SLASH_RE.search(raw):
        return False
    if _looks_like_doc_type(raw):
        return False
    if ARTICLE_START_RE.match(raw) or CHAPTER_START_RE.match(raw) or ITEM_START_RE.match(raw) or POINT_START_RE.match(raw):
        return False

    if _extract_agency_from_action_line(raw):
        return True

    return bool(AGENCY_LINE_RE.match(raw))


def _pick_issuing_agency(lines: List[str], doc_type_idx: Optional[int], official_title: str) -> str:
    official_title_up = _norm_space(official_title).upper()

    for line in lines[:50]:
        cand = _extract_agency_from_action_line(line)
        if cand:
            cand_up = cand.upper()
            if official_title_up and (cand_up == official_title_up or cand_up in official_title_up):
                continue
            return cand

    search_end = doc_type_idx if doc_type_idx is not None else min(len(lines), 20)
    head = lines[: max(0, search_end)]

    agency_candidates: List[str] = []
    for line in head[:12]:
        clean = _norm_space(line)
        if _looks_like_agency_line(clean):
            agency_candidates.append(clean)
            continue
        if agency_candidates:
            break

    if not agency_candidates:
        for line in lines[:20]:
            clean = _norm_space(line)
            if _looks_like_agency_line(clean):
                agency_candidates.append(clean)

    for cand in agency_candidates:
        cand_up = cand.upper()
        if official_title_up and (cand_up == official_title_up or cand_up in official_title_up):
            continue
        return cand

    return ""


@dataclass
class DocumentHeader:
    doc_id: str
    file_stem: str
    doc_type: str = ""
    law_type: str = ""
    doc_number: str = ""
    issuing_agency: str = ""
    official_title: str = ""
    law_name: str = ""
    title_block: str = ""
    lead_block: str = ""
    date_raw: str = ""
    year: int = 0
    place: str = ""
    body_start_index: int = 0

    def to_metadata(self) -> Dict[str, object]:
        data = asdict(self)
        data["law_name"] = self.official_title or self.file_stem
        data["law_type"] = self.doc_type or "Unknown"
        data["source"] = self.issuing_agency or "LocalFile"
        return data


def extract_document_header(text: str, *, fallback_name: str = "") -> DocumentHeader:
    lines = _clean_lines(text)
    file_stem = _slugify(fallback_name or "document")

    doc_type, doc_type_idx = _find_doc_type(lines)
    doc_number = _pick_doc_number(lines, fallback_name=fallback_name)
    date_raw, year, place = _pick_date(lines, fallback_name=fallback_name)
    title_block, official_title, body_start_index = _pick_title_block(lines, doc_type_idx)
    issuing_agency = _pick_issuing_agency(lines, doc_type_idx, official_title)

    lead_lines: List[str] = []
    if body_start_index > 0:
        start = (doc_type_idx + 1) if doc_type_idx is not None else 0
        for idx in range(start, min(body_start_index, start + 8)):
            line = _norm_space(lines[idx])
            if not line:
                continue
            if title_block and line in title_block:
                continue
            if NUMBER_LINE_RE.match(line) or DATE_RE.search(line) or DATE_SLASH_RE.search(line):
                continue
            if TITLE_ROLE_STOP_RE.match(line):
                continue
            lead_lines.append(line)
    lead_block = "\n".join(lead_lines).strip()

    if not year and doc_number:
        m = re.search(r"/(\d{4})/", doc_number)
        if m:
            year = int(m.group(1))

    law_name = official_title or fallback_name or file_stem
    doc_id_seed = doc_number or official_title or fallback_name or file_stem
    doc_id = _slugify(doc_id_seed)

    return DocumentHeader(
        doc_id=doc_id,
        file_stem=file_stem,
        doc_type=doc_type,
        law_type=doc_type or "Unknown",
        doc_number=doc_number,
        issuing_agency=issuing_agency,
        official_title=official_title,
        law_name=law_name,
        title_block=title_block,
        lead_block=lead_block,
        date_raw=date_raw,
        year=year or 0,
        place=place,
        body_start_index=body_start_index,
    )