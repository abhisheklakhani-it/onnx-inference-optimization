"""Export the Day 5 DistilBERT to ONNX and check that ONNX Runtime gives the same answers as PyTorch.

    python scripts/export.py

Writes models/model.onnx (+ tokenizer.json, config.json) and results/parity.json.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onnxopt.classifier import OnnxClassifier, encode  # noqa: E402
from onnxopt.data import load_test  # noqa: E402
from onnxopt.export import OPSET, export_onnx  # noqa: E402
from onnxopt.paths import MODELS, ONNX_FILES, RESULTS, SOURCE_MODEL  # noqa: E402

PARITY_ARTICLES = 64


def main() -> None:
    if not SOURCE_MODEL.exists():
        sys.exit(f"error: no model in {SOURCE_MODEL}; set DAY5_MODEL_DIR to the Day 5 models/distilbert folder")
    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    # Eager attention exports as plain MatMul/Softmax ops, which ONNX Runtime's optimizer can fuse later.
    model = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    for name in ["tokenizer.json", "config.json"]:
        shutil.copy(SOURCE_MODEL / name, MODELS / name)

    sample = tokenizer(["Ein kurzer Beispielsatz.", "Noch ein etwas längerer Beispielsatz für die Form."],
                       padding=True, return_tensors="pt")
    onnx_path = MODELS / ONNX_FILES["onnx_fp32"]
    start = time.perf_counter()
    export_onnx(model, sample, onnx_path)
    export_seconds = time.perf_counter() - start

    # Parity on real articles, tokenized once by the onnxruntime path (tokenizers only) and fed to both.
    texts, _, _ = load_test()
    texts = texts[:PARITY_ARTICLES]
    onnx = OnnxClassifier(onnx_path)
    hf_ids = tokenizer(texts, truncation=True, max_length=512, padding=True)["input_ids"]
    inputs = encode(onnx.tokenizer, texts)
    same_tokens = hf_ids == inputs["input_ids"].tolist()
    diffs, agree = [], 0
    for i in range(0, len(texts), 8):
        batch = {k: v[i:i + 8] for k, v in inputs.items()}
        with torch.inference_mode():
            ref = model(**{k: torch.from_numpy(v) for k, v in batch.items()}).logits.numpy()
        out = onnx.session.run(["logits"], batch)[0]
        diffs.append(np.abs(ref - out).max())
        agree += int((ref.argmax(-1) == out.argmax(-1)).sum())

    parity = {
        "opset": OPSET,
        "export_seconds": round(export_seconds, 1),
        "articles": len(texts),
        "tokenizers_match_transformers": same_tokens,
        "max_abs_logit_diff": float(max(diffs)),
        "same_label": agree,
        "onnx_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "onnx_files": sorted(p.name for p in MODELS.iterdir()),
    }
    (RESULTS / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(json.dumps(parity, indent=2))
    if not same_tokens or agree != len(texts) or max(diffs) > 1e-3:
        sys.exit("error: ONNX output differs from PyTorch")


if __name__ == "__main__":
    main()
