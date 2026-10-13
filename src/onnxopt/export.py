"""Export a Hugging Face sequence classifier to ONNX with dynamic batch and sequence length."""

from pathlib import Path

OPSET = 17


def export_onnx(model, sample: dict, path: Path, opset: int = OPSET) -> None:
    """`sample` is a tokenized example batch (input_ids, attention_mask as torch tensors) used for tracing."""
    import torch

    torch.onnx.export(
        model.eval(),
        (sample["input_ids"], sample["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"},
                      "logits": {0: "batch"}},
        opset_version=opset,
        # The TorchScript exporter: its MatMul/Softmax attention pattern is what
        # onnxruntime.transformers.optimizer knows how to fuse.
        dynamo=False,
    )
