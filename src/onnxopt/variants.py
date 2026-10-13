"""Turn an fp32 ONNX transformer into the faster variants: graph-optimized and int8-quantized."""

from pathlib import Path


def optimize(fp32: Path, out: Path, num_heads: int, hidden_size: int) -> dict[str, int]:
    """Fuse attention, LayerNorm and GELU into single ONNX Runtime ops; returns the fused op counts.

    opt_level 0 = only the transformer fusions; ONNX Runtime still applies its own CPU
    optimizations when a session loads the file. DistilBERT is BERT-shaped, so model_type="bert".
    """
    from onnxruntime.transformers.optimizer import optimize_model

    model = optimize_model(str(fp32), model_type="bert", num_heads=num_heads, hidden_size=hidden_size, opt_level=0)
    model.save_model_to_file(str(out))
    return {k: v for k, v in model.get_fused_operator_statistics().items() if v}


def quantize_int8(model: Path, out: Path) -> None:
    """Dynamic quantization: int8 weights, activations quantized on the fly (no calibration data)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Shape inference first, so the quantizer knows every tensor's type around the fused ops.
    prepared = out.with_suffix(".prep.onnx")
    quant_pre_process(str(model), str(prepared), skip_optimization=True)
    try:
        quantize_dynamic(str(prepared), str(out), weight_type=QuantType.QInt8)
    finally:
        prepared.unlink(missing_ok=True)
