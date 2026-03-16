python.exe -m pip install --upgrade pip
ocrmypdf --force-ocr "data/raw/VanBanGoc_07.2026.NĐ-CP.pdf" "data/normalized/VanBanGoc_07.2026.NĐ-CP_ocr.pdf"
ocrmypdf --force-ocr "data/raw/VanBanGoc_9_2026_ND-CP_10012026-signed.pdf" "data/normalized/VanBanGoc_9_2026_ND-CP_10012026-signed.pdf"
python -m scripts.ocr_to_txt --input_pdf "data/raw/VanBanGoc_9_2026_ND-CP_10012026-signed.pdf" --output_txt "data/normalized/VanBanGoc_9_2026_ND-CP_10012026-signed.txt" --poppler_path "C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin" --tesseract_cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" --lang vie
python -m scripts.ocr_to_txt --input_pdf "data/raw/VanBanGoc_07.2026.NĐ-CP.pdf" --output_txt "data/normalized/VanBanGoc_07.2026.NĐ-CP.pdf" --poppler_path "C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin" --tesseract_cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" --lang vie
