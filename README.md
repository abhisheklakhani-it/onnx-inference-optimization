# ONNX Inference Optimization

Exports the fine-tuned German DistilBERT from [german-text-classifier](https://github.com/abhisheklakhani-it/german-text-classifier) to ONNX, applies graph optimization and int8 dynamic quantization, and benchmarks latency, throughput, model size and accuracy. The fastest variant is served with FastAPI in a slim Docker image.

Work in progress.
