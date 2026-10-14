"""Build notebooks/benchmark_x86_colab.ipynb: the same benchmark on a Colab x86 CPU.

The notebook writes the project's source files into the Colab machine unchanged, rebuilds the
ONNX variants from the Day 5 model there, and runs scripts/benchmark.py on Colab's CPU.

    python scripts/make_colab_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ("torch==2.14.1 transformers==5.19.0 onnxruntime==1.31.0 onnx==1.23.2 onnxscript==0.7.2 "
        "tokenizers==0.23.3 datasets==5.1.0 scikit-learn")
SOURCES = ["src/onnxopt/__init__.py", "src/onnxopt/paths.py", "src/onnxopt/classifier.py", "src/onnxopt/data.py",
           "src/onnxopt/export.py", "src/onnxopt/variants.py", "src/onnxopt/bench.py",
           "scripts/export.py", "scripts/build_variants.py", "scripts/benchmark.py"]
OUT = "results/colab_x86"
BENCH = f"--out {OUT} --threads 2 --timing-articles 128 --rounds 2 --cooldown 0 --accuracy-variants onnx_int8"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip().splitlines(keepends=True)}


def write_file(path: str) -> dict:
    return code(f"%%writefile {path}\n" + (ROOT / path).read_text())


cells = [
    md("""
# ONNX benchmark on an x86 CPU (Colab)

Part of [onnx-inference-optimization](https://github.com/abhisheklakhani-it/onnx-inference-optimization).
Most servers and Docker hosts run Linux on x86 CPUs, so this repeats the Mac benchmark there.

1. **Runtime → Change runtime type → CPU**, then **Runtime → Run all**.
2. When asked, upload `day5_distilbert.zip` (the Day 5 `models/distilbert/` folder, zipped).
3. The run takes about 25 minutes. The last cell downloads `onnx_x86_results.zip`;
   unzip it into the project folder (it only contains `results/colab_x86/`).
"""),
    code("!lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Flags' | cut -c1-200"),
    code(f"!pip -q install {PINS}"),
    code("!mkdir -p src/onnxopt scripts results models"),
    *[write_file(path) for path in SOURCES],
    md("Upload the Day 5 model (about 250 MB):"),
    code("""
from google.colab import files
files.upload()
!unzip -qo day5_distilbert.zip -d day5 && find day5 -name config.json
"""),
    code("""
import glob, os
os.environ["DAY5_MODEL_DIR"] = os.path.dirname(glob.glob("day5/**/model.safetensors", recursive=True)[0])
!python scripts/export.py
!python scripts/build_variants.py
"""),
    md("Benchmark: 2 threads (Colab's 2 vCPUs), 128 timing articles x 2 rounds, int8 accuracy on all 1,028 articles."),
    code(f"!python scripts/benchmark.py {BENCH}"),
    code(f"""
!cp results/parity.json results/variants.json {OUT}/
!zip -qr onnx_x86_results.zip {OUT}
files.download("onnx_x86_results.zip")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = ROOT / "notebooks" / "benchmark_x86_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {out.relative_to(ROOT)} ({len(cells)} cells)")
