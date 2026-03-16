import argparse
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path


def ocr_pdf_to_txt(
    pdf_path: Path,
    out_txt: Path,
    dpi: int = 300,
    lang: str = "vie",
    poppler_path: str | None = None,
    tesseract_cmd: str | None = None,
):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        poppler_path=poppler_path,
    )

    texts = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang=lang)
        texts.append(txt)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n\n".join(texts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser("OCR PDF -> TXT")
    parser.add_argument("--input_pdf", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--lang", default="vie")

    parser.add_argument("--poppler_path", type=str, default=None)
    parser.add_argument("--tesseract_cmd", type=str, default=None)

    args = parser.parse_args()

    ocr_pdf_to_txt(
        pdf_path=Path(args.input_pdf),
        out_txt=Path(args.output_txt),
        dpi=args.dpi,
        lang=args.lang,
        poppler_path=args.poppler_path,
        tesseract_cmd=args.tesseract_cmd,
    )


if __name__ == "__main__":
    main()
