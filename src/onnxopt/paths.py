"""Where the source model comes from and where every variant is written."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The fine-tuned DistilBERT from Day 5 (german-text-classifier). Override with DAY5_MODEL_DIR.
SOURCE_MODEL = Path(os.environ.get("DAY5_MODEL_DIR", ROOT.parent / "german-text-classifier" / "models" / "distilbert"))

MODELS = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

# variant name -> ONNX file in MODELS ("pytorch" is the source model itself)
ONNX_FILES = {
    "onnx_fp32": "model.onnx",
    "onnx_optimized": "model.optimized.onnx",
    "onnx_int8": "model.optimized.int8.onnx",
}
VARIANTS = ["pytorch", *ONNX_FILES]
DISPLAY = {
    "pytorch": "PyTorch fp32",
    "onnx_fp32": "ONNX fp32",
    "onnx_optimized": "ONNX optimized",
    "onnx_int8": "ONNX optimized + int8",
}
