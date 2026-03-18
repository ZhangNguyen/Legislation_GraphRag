from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.normalize_raw import OcrConfig, _ocr_pdf_to_txt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdf", required=True, type=str)
    parser.add_argument("--output_txt", required=True, type=str)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--lang", type=str, default="vie")
    parser.add_argument("--poppler_path", type=str, default=r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin")
    parser.add_argument("--tesseract_cmd", type=str, default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    args = parser.parse_args()

    cfg = OcrConfig(
        poppler_path=args.poppler_path,
        tesseract_cmd=args.tesseract_cmd,
        lang=args.lang,
        dpi=args.dpi,
    )

    _ocr_pdf_to_txt(
        pdf_path=Path(args.input_pdf),
        out_txt=Path(args.output_txt),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()