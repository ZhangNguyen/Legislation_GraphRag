from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytesseract
from pdf2image import convert_from_path

from src.utils.loader import load_document


@dataclass(frozen=True)
class OcrConfig:
    # Fixed defaults (đúng như bạn muốn). Có thể override bằng ENV để deploy sau này.
    poppler_path: str = r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    lang: str = "vie"
    dpi: int = 300

    # folders fixed
    raw_dir: Path = Path("data/raw")
    normalized_dir: Path = Path("data/normalized")

    # If raw PDF already has text, copy PDF to normalized
    copy_text_pdf_to_normalized: bool = True

    # minimal text length to consider “has text”
    min_text_len: int = 50


def _ocr_pdf_to_txt(
    pdf_path: Path,
    out_txt: Path,
    cfg: OcrConfig,
) -> None:
    # Point pytesseract to tesseract.exe
    pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    images = convert_from_path(
        str(pdf_path),
        dpi=cfg.dpi,
        poppler_path=cfg.poppler_path,
    )

    texts: list[str] = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang=cfg.lang)
        texts.append(txt)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n\n".join(texts), encoding="utf-8")


def normalize_raw_to_normalized(cfg: Optional[OcrConfig] = None) -> None:
    """
    Convert everything in data/raw -> data/normalized

    - If PDF has extractable text: optionally copy PDF -> normalized
    - If PDF has no extractable text: OCR -> normalized/<same_name>.txt
    - TXT in raw: copy to normalized
    """
    cfg = cfg or OcrConfig()

    cfg.normalized_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.raw_dir.exists():
        raise RuntimeError(f"Missing folder: {cfg.raw_dir}")

    raw_files = sorted(cfg.raw_dir.glob("*.*"))
    if not raw_files:
        print(f"[NORMALIZE] No files in {cfg.raw_dir}")
        return

    # Allow override by ENV for deploy later (không bắt buộc)
    poppler_path = os.getenv("POPPLER_PATH", cfg.poppler_path)
    tesseract_cmd = os.getenv("TESSERACT_CMD", cfg.tesseract_cmd)
    lang = os.getenv("TESSERACT_LANG", cfg.lang)
    dpi = int(os.getenv("OCR_DPI", str(cfg.dpi)))

    cfg = OcrConfig(
        poppler_path=poppler_path,
        tesseract_cmd=tesseract_cmd,
        lang=lang,
        dpi=dpi,
        raw_dir=cfg.raw_dir,
        normalized_dir=cfg.normalized_dir,
        copy_text_pdf_to_normalized=cfg.copy_text_pdf_to_normalized,
        min_text_len=cfg.min_text_len,
    )

    for fp in raw_files:
        suf = fp.suffix.lower()

        # 1) raw .txt -> copy
        if suf == ".txt":
            out_txt = cfg.normalized_dir / fp.name
            if not out_txt.exists():
                shutil.copy2(fp, out_txt)
                print(f"[NORMALIZE] copied TXT -> {out_txt}")
            continue

        # 2) raw .pdf
        if suf == ".pdf":
            # If already OCRed txt exists -> skip OCR
            out_txt = cfg.normalized_dir / f"{fp.stem}.txt"
            out_pdf = cfg.normalized_dir / fp.name

            # Quick check: extract text via loader (pypdf)
            try:
                text = load_document(fp)  # uses pypdf for PDFs
            except Exception as e:
                print(f"[NORMALIZE][WARN] {fp.name}: load_document failed -> OCR fallback. err={e}")
                text = ""

            if len(text.strip()) >= cfg.min_text_len:
                # This is a text PDF
                if cfg.copy_text_pdf_to_normalized:
                    if not out_pdf.exists():
                        shutil.copy2(fp, out_pdf)
                        print(f"[NORMALIZE] copied TEXT-PDF -> {out_pdf}")
                else:
                    # Or save extracted text as .txt (tuỳ bạn)
                    if not out_txt.exists():
                        out_txt.write_text(text, encoding="utf-8")
                        print(f"[NORMALIZE] wrote extracted text -> {out_txt}")
                continue

            # No extractable text => OCR
            if out_txt.exists() and out_txt.stat().st_size > 100:
                print(f"[NORMALIZE] OCR exists -> skip: {out_txt.name}")
                continue

            print(f"[NORMALIZE] OCR start: {fp.name}")
            _ocr_pdf_to_txt(fp, out_txt, cfg)
            print(f"[NORMALIZE] OCR done -> {out_txt.name}")
            continue

        # ignore other files
        print(f"[NORMALIZE] skip unsupported: {fp.name}")
