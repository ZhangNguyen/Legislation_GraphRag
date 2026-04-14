FROM python:3.10-slim

WORKDIR /app

# install system deps (OCR)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# copy code
COPY . .

# install python deps
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir .

# run app
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "7860"]