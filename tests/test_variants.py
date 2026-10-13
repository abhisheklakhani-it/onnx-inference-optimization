import numpy as np

from onnxopt.classifier import OnnxClassifier
from onnxopt.variants import optimize, quantize_int8

TEXTS = ["Tor", "Spiel Tor Tor Börse", "Aktie", "Börse Aktie Aktie Spiel Tor"]


def test_optimize_fuses_attention_and_keeps_the_outputs(tiny_model_dir):
    folder, _ = tiny_model_dir
    fused = optimize(folder / "model.onnx", folder / "model.optimized.onnx", num_heads=2, hidden_size=32)
    assert fused.get("Attention") == 2  # one per layer
    ref = OnnxClassifier(folder / "model.onnx").logits(TEXTS)
    assert np.allclose(OnnxClassifier(folder / "model.optimized.onnx").logits(TEXTS), ref, atol=1e-4)


def test_int8_model_is_smaller_and_close(tiny_model_dir):
    folder, _ = tiny_model_dir
    optimize(folder / "model.onnx", folder / "model.optimized.onnx", num_heads=2, hidden_size=32)
    quantize_int8(folder / "model.optimized.onnx", folder / "model.int8.onnx")
    assert (folder / "model.int8.onnx").stat().st_size < (folder / "model.optimized.onnx").stat().st_size
    assert not (folder / "model.int8.prep.onnx").exists()  # temporary file cleaned up
    ref = OnnxClassifier(folder / "model.onnx").logits(TEXTS)
    assert np.abs(OnnxClassifier(folder / "model.int8.onnx").logits(TEXTS) - ref).max() < 0.5
