from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytesseract
from pdf2image import convert_from_path

from src.utils.loader import load_document, normalize_text


@dataclass(frozen=True)
class OcrConfig:
    poppler_path: str = r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    lang: str = "vie"
    dpi: int = 300

    raw_dir: Path = Path("data/raw")
    normalized_dir: Path = Path("data/normalized")

    min_text_len: int = 50
    skip_if_exists_min_bytes: int = 100


def _build_runtime_cfg(cfg: OcrConfig) -> OcrConfig:
    return OcrConfig(
        poppler_path=os.getenv("POPPLER_PATH", cfg.poppler_path),
        tesseract_cmd=os.getenv("TESSERACT_CMD", cfg.tesseract_cmd),
        lang=os.getenv("TESSERACT_LANG", cfg.lang),
        dpi=int(os.getenv("OCR_DPI", str(cfg.dpi))),
        raw_dir=Path(os.getenv("RAW_DIR", str(cfg.raw_dir))),
        normalized_dir=Path(os.getenv("NORMALIZED_DIR", str(cfg.normalized_dir))),
        min_text_len=int(os.getenv("MIN_TEXT_LEN", str(cfg.min_text_len))),
        skip_if_exists_min_bytes=int(
            os.getenv("SKIP_IF_EXISTS_MIN_BYTES", str(cfg.skip_if_exists_min_bytes))
        ),
    )


def _to_normalized_txt_path(src_path: Path, raw_dir: Path, normalized_dir: Path) -> Path:
    rel = src_path.relative_to(raw_dir)
    rel_no_suffix = rel.with_suffix(".txt")
    return normalized_dir / rel_no_suffix


def _should_skip_existing(out_txt: Path, min_bytes: int) -> bool:
    return out_txt.exists() and out_txt.stat().st_size >= min_bytes


def _ocr_pdf_to_txt(pdf_path: Path, out_txt: Path, cfg: OcrConfig) -> None:
    pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd
    tessdata_dir = Path(cfg.tesseract_cmd).parent / "tessdata"
    os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)

    images = convert_from_path(
        str(pdf_path),
        dpi=cfg.dpi,
        poppler_path=cfg.poppler_path,
    )

    texts: list[str] = []
    for i, img in enumerate(images, start=1):
        txt = pytesseract.image_to_string(img, lang=cfg.lang)
        txt = normalize_text(txt)
        texts.append(txt)
        print(f"[OCR] {pdf_path.name}: page {i}/{len(images)} done")

    final_text = "\n\n".join(t for t in texts if t.strip())

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(final_text, encoding="utf-8")
    print(f"[OCR] wrote -> {out_txt}")


def _handle_txt_file(fp: Path, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    raw_text = fp.read_text(encoding="utf-8", errors="ignore")
    cleaned = normalize_text(raw_text)
    out_txt.write_text(cleaned, encoding="utf-8")
    print(f"[NORMALIZE] TXT -> {out_txt}")


def _handle_word_file(fp: Path, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    extracted = load_document(fp)  # loader.py sẽ tự xử lý .doc/.docx
    extracted = normalize_text(extracted)

    if not extracted.strip():
        raise RuntimeError(f"Word file has no extractable text: {fp}")

    out_txt.write_text(extracted, encoding="utf-8")
    print(f"[NORMALIZE] WORD -> {out_txt}")


def _handle_pdf_file(fp: Path, out_txt: Path, cfg: OcrConfig) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    try:
        extracted = load_document(fp)
    except Exception as e:
        print(f"[NORMALIZE][WARN] read PDF failed, OCR fallback: {fp} | err={e}")
        extracted = ""

    if len(extracted.strip()) >= cfg.min_text_len:
        out_txt.write_text(extracted, encoding="utf-8")
        print(f"[NORMALIZE] TEXT-PDF -> {out_txt}")
        return

    print(f"[NORMALIZE] OCR start -> {fp}")
    _ocr_pdf_to_txt(fp, out_txt, cfg)
    print(f"[NORMALIZE] OCR done -> {out_txt}")


def normalize_raw_to_normalized(cfg: Optional[OcrConfig] = None) -> None:
    """
    Chuẩn hóa toàn bộ file trong data/raw sang data/normalized

    Quy tắc:
    - *.txt   -> normalize -> normalized/*.txt
    - *.pdf   -> extract text; nếu không được thì OCR -> normalized/*.txt
    - *.docx  -> extract text -> normalized/*.txt
    - *.doc   -> convert/read -> normalized/*.txt
    """
    cfg = _build_runtime_cfg(cfg or OcrConfig())

    cfg.normalized_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.raw_dir.exists():
        raise RuntimeError(f"Missing folder: {cfg.raw_dir}")

    raw_files = [p for p in cfg.raw_dir.rglob("*") if p.is_file()]
    if not raw_files:
        print(f"[NORMALIZE] No files found in {cfg.raw_dir}")
        return

    print(f"[NORMALIZE] raw_dir={cfg.raw_dir}")
    print(f"[NORMALIZE] normalized_dir={cfg.normalized_dir}")
    print(f"[NORMALIZE] total_files={len(raw_files)}")

    for fp in raw_files:
        suffix = fp.suffix.lower()

        if suffix not in {".pdf", ".txt", ".doc", ".docx"}:
            print(f"[NORMALIZE] skip unsupported: {fp}")
            continue

        out_txt = _to_normalized_txt_path(fp, cfg.raw_dir, cfg.normalized_dir)

        if _should_skip_existing(out_txt, cfg.skip_if_exists_min_bytes):
            print(f"[NORMALIZE] skip existing: {out_txt}")
            continue

        try:
            if suffix == ".txt":
                _handle_txt_file(fp, out_txt)
            elif suffix == ".pdf":
                _handle_pdf_file(fp, out_txt, cfg)
            elif suffix in {".doc", ".docx"}:
                _handle_word_file(fp, out_txt)
        except Exception as e:
            print(f"[NORMALIZE][ERROR] failed: {fp} | err={e}")