"""Build the faster variants from models/model.onnx: graph-optimized and int8-quantized.

    python scripts/export.py            # first
    python scripts/build_variants.py

- onnx_optimized: ONNX Runtime's transformer optimizer fuses attention, LayerNorm and GELU
  into single ops (fewer, bigger kernels).
- onnx_int8: dynamic quantization of the optimized model. Weights are stored as int8; the
  activations are quantized on the fly for each input, so no calibration data is needed.

Writes results/variants.json: file size, fused ops, and how close each variant stays to fp32.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onnxopt.classifier import OnnxClassifier  # noqa: E402
from onnxopt.data import load_test  # noqa: E402
from onnxopt.paths import MODELS, ONNX_FILES, RESULTS  # noqa: E402
from onnxopt.variants import optimize, quantize_int8  # noqa: E402

CHECK_ARTICLES = 64


def main() -> None:
    fp32 = MODELS / ONNX_FILES["onnx_fp32"]
    if not fp32.exists():
        sys.exit("error: run scripts/export.py first")
    optimized = MODELS / ONNX_FILES["onnx_optimized"]
    int8 = MODELS / ONNX_FILES["onnx_int8"]

    config = json.loads((MODELS / "config.json").read_text())
    fused = optimize(fp32, optimized, num_heads=config["n_heads"], hidden_size=config["dim"])
    quantize_int8(optimized, int8)

    texts, _, _ = load_test()
    texts = texts[:CHECK_ARTICLES]
    ref = OnnxClassifier(fp32).logits(texts)
    report = {"check_articles": len(texts), "fused_operators": fused, "variants": {}}
    for name, file in ONNX_FILES.items():
        logits = ref if name == "onnx_fp32" else OnnxClassifier(MODELS / file).logits(texts)
        report["variants"][name] = {
            "file": file,
            "mb": round((MODELS / file).stat().st_size / 1e6, 1),
            "max_abs_logit_diff_vs_fp32": float(np.abs(logits - ref).max()),
            "same_label_as_fp32": int((logits.argmax(-1) == ref.argmax(-1)).sum()),
        }
    (RESULTS / "variants.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
