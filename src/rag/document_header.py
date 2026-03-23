from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

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

STOP_TITLE_PATTERNS = [
    re.compile(r"^Điều\s+\d+", re.IGNORECASE),
    re.compile(r"^(Bộ trưởng|Thủ tướng|Chính phủ|Ủy ban nhân dân|Chủ tịch|Theo|Tình hình|Căn cứ|Xét)\b", re.IGNORECASE),
    re.compile(r"^\d+\.\s+"),
    re.compile(r"^[a-zđ]\)\s+", re.IGNORECASE),
]

AGENCY_SKIP = (
    "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
    "Độc lập - Tự do - Hạnh phúc",
)

NUMBER_RE = re.compile(r"^Số\s*:\s*(.+)$", re.IGNORECASE)
DATE_RE = re.compile(
    r"(?:^|,\s*)(?P<place>[A-ZÀ-Ỵa-zà-ỵ\s\.\-]+),\s*ngày\s*(?P<day>\d{1,2})\s*tháng\s*(?P<month>\d{1,2})\s*năm\s*(?P<year>\d{4})",
    re.IGNORECASE,
)
ARTICLE_START_RE = re.compile(r"^Điều\s+\d+", re.IGNORECASE)
ITEM_START_RE = re.compile(r"^\d+\.\s+")
POINT_START_RE = re.compile(r"^[a-zđ]\)\s+", re.IGNORECASE)


def _norm_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _slugify(text: str) -> str:
    raw = _norm_space(text).lower()
    raw = re.sub(r"[^a-z0-9à-ỹ]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw or "doc"


def _clean_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text or "").splitlines():
        line = raw.replace("\ufeff", " ").strip()
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out


def _is_upper_like(line: str) -> bool:
    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) < 4:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return upper / max(len(letters), 1) >= 0.72


def _pick_issuing_agency(lines: List[str]) -> Optional[str]:
    for line in lines[:12]:
        if line in AGENCY_SKIP:
            continue
        if _is_upper_like(line) and not NUMBER_RE.match(line) and "Hà Nội" not in line:
            return line
    return None


def _pick_doc_number(lines: List[str]) -> Optional[str]:
    for line in lines[:20]:
        m = NUMBER_RE.match(line)
        if m:
            return _norm_space(m.group(1))
    return None


def _pick_date(lines: List[str]) -> tuple[Optional[str], Optional[int]]:
    for line in lines[:20]:
        m = DATE_RE.search(line)
        if m:
            day = int(m.group("day"))
            month = int(m.group("month"))
            year = int(m.group("year"))
            return f"{day:02d}/{month:02d}/{year:04d}", year
    return None, None


def _find_doc_type(lines: List[str]) -> tuple[Optional[str], Optional[int]]:
    for idx, line in enumerate(lines[:40]):
        clean = _norm_space(line).upper()
        for doc_type in DOC_TYPES:
            if clean == doc_type:
                return doc_type, idx
    return None, None


def _is_title_stop(line: str) -> bool:
    return any(p.search(line) for p in STOP_TITLE_PATTERNS)


@dataclass
class DocumentHeader:
    doc_id: str
    file_stem: str
    issuing_agency: Optional[str]
    doc_number: Optional[str]
    doc_type: Optional[str]
    official_title: Optional[str]
    title_block: str
    lead_block: str
    date_raw: Optional[str]
    year: Optional[int]
    body_start_index: int

    def to_metadata(self) -> Dict[str, object]:
        data = asdict(self)
        data["law_name"] = self.official_title or self.file_stem
        data["law_type"] = self.doc_type or "Unknown"
        data["source"] = self.issuing_agency or "LocalFile"
        return data


def extract_document_header(text: str, *, fallback_name: str = "") -> DocumentHeader:
    lines = _clean_lines(text)
    file_stem = fallback_name or "document"
    issuing_agency = _pick_issuing_agency(lines)
    doc_number = _pick_doc_number(lines)
    date_raw, year = _pick_date(lines)
    doc_type, doc_type_idx = _find_doc_type(lines)

    title_lines: List[str] = []
    title_start = None
    body_start_index = 0

    if doc_type_idx is not None:
        title_start = doc_type_idx
        title_lines.append(lines[doc_type_idx])
        for idx in range(doc_type_idx + 1, min(len(lines), doc_type_idx + 10)):
            line = lines[idx]
            if _is_title_stop(line):
                body_start_index = idx
                break
            if _is_upper_like(line) or line.upper().startswith("VỀ VIỆC"):
                title_lines.append(line)
                continue
            body_start_index = idx
            break
        else:
            body_start_index = min(len(lines), doc_type_idx + len(title_lines))

    if not title_lines:
        # fallback: lấy block upper-case dài nhất gần đầu file
        best_block: List[str] = []
        current: List[str] = []
        current_start = 0
        for idx, line in enumerate(lines[:30]):
            if _is_upper_like(line):
                if not current:
                    current_start = idx
                current.append(line)
            else:
                if len(" ".join(current)) > len(" ".join(best_block)):
                    best_block = current[:]
                    title_start = current_start
                current = []
        if len(" ".join(current)) > len(" ".join(best_block)):
            best_block = current[:]
            title_start = current_start
        title_lines = best_block[:3]
        body_start_index = (title_start or 0) + len(title_lines)
        if title_lines and title_lines[0] in DOC_TYPES:
            doc_type = title_lines[0]

    title_block = "\n".join(title_lines).strip()
    official_title = None
    if title_lines:
        if doc_type and title_lines[0].upper() == doc_type:
            official_title = " ".join(title_lines[1:]).strip() or title_lines[0]
        else:
            official_title = " ".join(title_lines).strip()

    lead_lines: List[str] = []
    if body_start_index < len(lines):
        for idx in range(body_start_index, min(len(lines), body_start_index + 12)):
            line = lines[idx]
            if ARTICLE_START_RE.match(line) or ITEM_START_RE.match(line) or POINT_START_RE.match(line):
                body_start_index = idx
                break
            lead_lines.append(line)
            if len(" ".join(lead_lines)) >= 900:
                body_start_index = idx + 1
                break
        else:
            body_start_index = min(len(lines), body_start_index + len(lead_lines))

    lead_block = "\n".join(lead_lines[:6]).strip()

    doc_id_seed = doc_number or official_title or file_stem
    doc_id = _slugify(doc_id_seed)

    return DocumentHeader(
        doc_id=doc_id,
        file_stem=file_stem,
        issuing_agency=issuing_agency,
        doc_number=doc_number,
        doc_type=doc_type,
        official_title=official_title,
        title_block=title_block,
        lead_block=lead_block,
        date_raw=date_raw,
        year=year,
        body_start_index=body_start_index,
    )
