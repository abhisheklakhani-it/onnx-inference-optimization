# Serves the int8 ONNX model with FastAPI. No PyTorch or transformers in the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.optimized.int8.onnx \
    THREADS=4
# THREADS: onnxruntime threads per request. Match the host's fast (physical) cores: on an M2, using
# all 8 cores (incl. the 4 efficiency cores) made the median request 99 ms instead of 61 ms.

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY models/model.optimized.int8.onnx models/tokenizer.json models/config.json models/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "onnxopt.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
