"""Classify German news text with an ONNX model: onnxruntime + tokenizers only, no PyTorch.

This is what the API and the Docker image use. The folder must hold the .onnx file plus
tokenizer.json and config.json copied from the source model (scripts/export.py does that).
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_tokenizer(model_dir: Path, max_length: int = 512) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    return tokenizer


def encode(tokenizer: Tokenizer, texts: list[str]) -> dict[str, np.ndarray]:
    batch = tokenizer.encode_batch(texts)
    return {
        "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64),
    }


class OnnxClassifier:
    def __init__(self, model_path: Path, threads: int | None = None, max_length: int = 512):
        model_dir = model_path.parent
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.tokenizer = load_tokenizer(model_dir, max_length)
        config = json.loads((model_dir / "config.json").read_text())
        self.labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]

    def logits(self, texts: list[str]) -> np.ndarray:
        return self.session.run(["logits"], encode(self.tokenizer, texts))[0]

    def predict(self, texts: list[str]) -> list[dict]:
        probs = softmax(self.logits(texts))
        return [{"label": self.labels[int(p.argmax())], "confidence": round(float(p.max()), 4)} for p in probs]
