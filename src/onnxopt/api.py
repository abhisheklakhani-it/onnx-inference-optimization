"""FastAPI service for the fastest variant: onnxruntime + tokenizers, no PyTorch.

    uvicorn onnxopt.api:app --app-dir src

MODEL_PATH picks the .onnx file (default: the int8 model); tokenizer.json and config.json must sit
next to it. THREADS sets onnxruntime's intra-op threads (default: let onnxruntime decide).
"""

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from onnxopt.classifier import OnnxClassifier
from onnxopt.paths import MODELS, ONNX_FILES

MAX_TEXTS = 32


class PredictRequest(BaseModel):
    texts: list[Annotated[str, Field(min_length=1, max_length=50_000)]] = Field(min_length=1, max_length=MAX_TEXTS)


class Prediction(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: str
    ms: float


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", MODELS / ONNX_FILES["onnx_int8"]))


@lru_cache(maxsize=1)
def get_classifier() -> OnnxClassifier:
    path = model_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"model file not found: {path.name}")
    threads = int(os.environ["THREADS"]) if os.environ.get("THREADS") else None
    return OnnxClassifier(path, threads=threads)


app = FastAPI(title="German news topic classifier (ONNX)", version="1.0")


@app.get("/health")
def health() -> dict:
    path = model_path()
    return {"status": "ok" if path.exists() else "model missing", "model": path.name}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, classifier: OnnxClassifier = Depends(get_classifier)) -> PredictResponse:
    start = time.perf_counter()
    predictions = classifier.predict(request.texts)
    return PredictResponse(predictions=predictions, model=model_path().name,
                           ms=round((time.perf_counter() - start) * 1000, 1))
