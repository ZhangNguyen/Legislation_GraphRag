from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytesseract
from pdf2image import convert_from_path

from src.utils.loader import load_document, normalize_text


@dataclass(frozen=True)
class OcrConfig:
    # Có thể override bằng ENV
    poppler_path: str = r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    lang: str = "vie"
    dpi: int = 300

    raw_dir: Path = Path("data/raw")
    normalized_dir: Path = Path("data/normalized")

    # số ký tự tối thiểu để coi PDF là có text extractable
    min_text_len: int = 50

    # nếu output txt đã tồn tại và đủ lớn thì bỏ qua
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


def _handle_pdf_file(fp: Path, out_txt: Path, cfg: OcrConfig) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # thử extract text trước
    try:
        extracted = load_document(fp)  # pypdf + normalize_text
    except Exception as e:
        print(f"[NORMALIZE][WARN] read PDF failed, OCR fallback: {fp} | err={e}")
        extracted = ""

    # nếu PDF có text thì ghi ra txt luôn
    if len(extracted.strip()) >= cfg.min_text_len:
        out_txt.write_text(extracted, encoding="utf-8")
        print(f"[NORMALIZE] TEXT-PDF -> {out_txt}")
        return

    # nếu không có text thì OCR
    print(f"[NORMALIZE] OCR start -> {fp}")
    _ocr_pdf_to_txt(fp, out_txt, cfg)
    print(f"[NORMALIZE] OCR done -> {out_txt}")


def normalize_raw_to_normalized(cfg: Optional[OcrConfig] = None) -> None:
    """
    Chuẩn hóa toàn bộ file trong data/raw sang data/normalized

    - If PDF has extractable text: optionally copy PDF -> normalized
    - If PDF has no extractable text: OCR -> normalized/<same_name>.txt
    - TXT in raw: copy to normalized
    - DOC/DOCX in raw: extract text -> normalized/<same_name>.txt (requires parser/tool)
    """
    cfg = cfg or OcrConfig()
    copied_txt = 0
    extracted_doc = 0
    copied_pdf = 0
    ocr_pdf = 0
    skipped_unsupported = 0
    skipped_existing = 0
    doc_extract_failed = 0

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

    print(f"[NORMALIZE] raw_dir={cfg.raw_dir}")
    print(f"[NORMALIZE] normalized_dir={cfg.normalized_dir}")
    print(f"[NORMALIZE] total_files={len(raw_files)}")

    for fp in raw_files:
        suf = fp.suffix.lower()

        # 1) raw .txt -> copy
        if suf == ".txt":
            out_txt = cfg.normalized_dir / fp.name
            if not out_txt.exists():
                shutil.copy2(fp, out_txt)
                print(f"[NORMALIZE] copied TXT -> {out_txt}")
                copied_txt += 1
            else:
                skipped_existing += 1
                print(f"[NORMALIZE] skip existing: {out_txt}")
            continue

        # 1.5) raw .doc/.docx -> extract to txt
        if suf in {".doc", ".docx"} or suf.startswith(".doc"):
            out_txt = cfg.normalized_dir / f"{fp.stem}.txt"
            if out_txt.exists() and out_txt.stat().st_size > 100:
                print(f"[NORMALIZE] DOC exists -> skip: {out_txt.name}")
                skipped_existing += 1
                continue

            try:
                text = load_document(fp)
            except Exception as e:
                print(f"[NORMALIZE][WARN] {fp.name}: cannot extract DOC/DOCX -> skip. err={e}")
                doc_extract_failed += 1
                continue

            if len(text.strip()) < cfg.min_text_len:
                print(f"[NORMALIZE][WARN] {fp.name}: extracted text too short -> skip")
                continue

            out_txt.write_text(text, encoding="utf-8")
            print(f"[NORMALIZE] extracted DOC/DOCX -> {out_txt}")
            extracted_doc += 1
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
                        copied_pdf += 1
                    else:
                        skipped_existing += 1
                        print(f"[NORMALIZE] skip existing: {out_pdf}")
                else:
                    # Or save extracted text as .txt (tuỳ bạn)
                    if not out_txt.exists():
                        out_txt.write_text(text, encoding="utf-8")
                        print(f"[NORMALIZE] wrote extracted text -> {out_txt}")
                        copied_pdf += 1
                    else:
                        skipped_existing += 1
                        print(f"[NORMALIZE] skip existing: {out_txt}")
                continue

            # No extractable text => OCR
            if out_txt.exists() and out_txt.stat().st_size > 100:
                print(f"[NORMALIZE] OCR exists -> skip: {out_txt.name}")
                skipped_existing += 1
                continue

            print(f"[NORMALIZE] OCR start: {fp.name}")
            _ocr_pdf_to_txt(fp, out_txt, cfg)
            print(f"[NORMALIZE] OCR done -> {out_txt.name}")
            ocr_pdf += 1
            continue

        # ignore other files
        print(f"[NORMALIZE] skip unsupported: {fp.name} (suffix={suf!r})")
        skipped_unsupported += 1

    print(
        "[NORMALIZE] summary:",
        f"copied_txt={copied_txt},",
        f"extracted_doc={extracted_doc},",
        f"copied_pdf={copied_pdf},",
        f"ocr_pdf={ocr_pdf},",
        f"skipped_existing={skipped_existing},",
        f"skipped_unsupported={skipped_unsupported},",
        f"doc_extract_failed={doc_extract_failed}",
    )
