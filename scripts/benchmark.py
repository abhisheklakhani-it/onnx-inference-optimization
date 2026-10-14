"""Benchmark every variant: accuracy on all 1,028 test articles, then latency and throughput.

    python scripts/benchmark.py                   # all phases (about 45 min on an M2 Air)
    python scripts/benchmark.py --phase timing    # re-time only; keeps saved accuracy runs
    python scripts/benchmark.py --phase summary   # rebuild benchmark.json from the saved runs
    python scripts/benchmark.py --out results/colab_x86 --threads 2 --timing-articles 128 \
        --rounds 2 --cooldown 0 --accuracy-variants onnx_int8   # the Colab x86 run

Accuracy: each variant classifies all 1,028 test articles at batch sizes 1, 8 and 32 (padding
can change int8 results, because dynamic quantization scales each batch as a whole).

Timing: 3 rounds on the first 256 test articles. The variant order rotates every round and
there is a cool-down between variants, because a fanless laptop slows down under sustained
load and would otherwise penalise whichever variant runs last. Every variant gets the same
thread budget, the same tokenizer (timed as part of each call) and a short warm-up.

Each run happens in its own process, so peak memory is measured per variant.
Writes results/benchmark/*.json (raw runs) and results/benchmark.json (summary).
"""

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onnxopt.bench import run_batches, summarize  # noqa: E402
from onnxopt.classifier import encode, load_tokenizer  # noqa: E402
from onnxopt.data import load_test  # noqa: E402
from onnxopt.paths import MODELS, ONNX_FILES, RESULTS, SOURCE_MODEL, VARIANTS  # noqa: E402

BATCH_SIZES = [1, 8, 32]
THREADS = 4  # default: the M2's 4 performance cores (set with --threads)


def load_variant(name: str):
    """Returns (logits_fn, load_seconds, size_mb)."""
    start = time.perf_counter()
    if name == "pytorch":
        import torch
        from transformers import AutoModelForSequenceClassification

        torch.set_num_threads(THREADS)
        model = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL).eval()
        tokenizer = load_tokenizer(MODELS)

        def logits_fn(texts):
            with torch.inference_mode():
                inputs = {k: torch.from_numpy(v) for k, v in encode(tokenizer, texts).items()}
                return model(**inputs).logits.numpy()

        size = (SOURCE_MODEL / "model.safetensors").stat().st_size
    else:
        from onnxopt.classifier import OnnxClassifier

        clf = OnnxClassifier(MODELS / ONNX_FILES[name], threads=THREADS)
        logits_fn = clf.logits
        size = (MODELS / ONNX_FILES[name]).stat().st_size
    return logits_fn, time.perf_counter() - start, round(size / 1e6, 1)


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 1e6 if sys.platform == "darwin" else rss * 1024 / 1e6, 1)  # bytes on macOS, KiB on Linux


def accuracy_run(name: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    texts, labels, _ = load_test()
    logits_fn, load_seconds, size_mb = load_variant(name)
    out = {"variant": name, "size_mb": size_mb, "load_seconds": round(load_seconds, 2), "batch_sizes": {}}
    for batch_size in BATCH_SIZES:
        _, logits = run_batches(logits_fn, texts, batch_size, warmup=0)
        pred = logits.argmax(-1)
        out["batch_sizes"][str(batch_size)] = {
            "accuracy": round(accuracy_score(labels, pred), 4),
            "macro_f1": round(f1_score(labels, pred, average="macro"), 4),
            "predictions": pred.tolist(),
        }
    out["peak_rss_mb"] = peak_rss_mb()
    return out


def timing_run(name: str, articles: int) -> dict:
    texts = load_test()[0][:articles]
    logits_fn, _, _ = load_variant(name)
    seconds = {str(b): run_batches(logits_fn, texts, b)[0] for b in BATCH_SIZES}
    return {"variant": name, "articles": len(texts), "seconds": seconds}


def machine() -> dict:
    import onnxruntime
    import torch

    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    elif Path("/proc/cpuinfo").exists():
        names = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")]
        cpu = f"{names[0]} ({len(names)} vCPUs)" if names else cpu
    return {"cpu": cpu, "os": platform.platform(), "threads": THREADS, "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__, "torch": torch.__version__}


def summarize_all(folder: Path, args) -> dict:
    variants = {}
    reference = None
    for name in VARIANTS:
        acc_file = folder / f"accuracy_{name}.json"
        acc = json.loads(acc_file.read_text()) if acc_file.exists() else None
        if name == "pytorch" and acc:
            reference = np.array(acc["batch_sizes"]["1"]["predictions"])
        rounds = [json.loads(p.read_text()) for p in sorted(folder.glob(f"timing_{name}_r*.json"))]
        size = SOURCE_MODEL / "model.safetensors" if name == "pytorch" else MODELS / ONNX_FILES[name]
        out = {"size_mb": round(size.stat().st_size / 1e6, 1), "rounds": len(rounds)}
        if acc:
            out.update(load_seconds=acc["load_seconds"], peak_rss_mb=acc["peak_rss_mb"])
        out["batch_sizes"] = {}
        for batch in map(str, BATCH_SIZES):
            pooled = [s for r in rounds for s in r["seconds"][batch]]
            stats = summarize(pooled, sum(r["articles"] for r in rounds))
            stats["articles_per_s_by_round"] = [round(r["articles"] / sum(r["seconds"][batch]), 1) for r in rounds]
            if acc:
                pred = np.array(acc["batch_sizes"][batch]["predictions"])
                stats.update(accuracy=acc["batch_sizes"][batch]["accuracy"], macro_f1=acc["batch_sizes"][batch]["macro_f1"])
                if reference is not None:
                    stats["same_label_as_pytorch"] = int((pred == reference).sum())
            out["batch_sizes"][batch] = stats
        variants[name] = out
    return {"machine": machine(), "timing_articles": args.timing_articles, "timing_rounds": args.rounds,
            "cooldown_seconds": args.cooldown, "variants": variants}


def main() -> None:
    global THREADS  # load_variant reads it; set from --threads
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "accuracy", "timing", "summary"], default="all")
    parser.add_argument("--variant", choices=VARIANTS, help="internal: run one variant in this process")
    parser.add_argument("--round", type=int, default=1, help="internal: timing round number")
    parser.add_argument("--out", type=Path, default=RESULTS, help="results folder")
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--timing-articles", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=20, help="seconds of rest between runs")
    parser.add_argument("--accuracy-variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    THREADS = args.threads
    folder = args.out / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    shared = ["--out", str(args.out), "--threads", str(args.threads), "--timing-articles", str(args.timing_articles)]

    def run_child(*child_args: str) -> None:
        subprocess.run([sys.executable, __file__, *shared, *child_args], check=True)
        time.sleep(args.cooldown)

    if args.variant:
        if args.phase == "accuracy":
            (folder / f"accuracy_{args.variant}.json").write_text(json.dumps(accuracy_run(args.variant)) + "\n")
        else:
            result = timing_run(args.variant, args.timing_articles)
            (folder / f"timing_{args.variant}_r{args.round}.json").write_text(json.dumps(result) + "\n")
            b1 = result["articles"] / sum(result["seconds"]["1"])
            print(f"round {args.round}  {args.variant:15s} batch 1: {b1:5.1f} articles/s", flush=True)
        return

    if args.phase in ("all", "accuracy"):
        for name in args.accuracy_variants:
            run_child("--phase", "accuracy", "--variant", name)
    if args.phase in ("all", "timing"):
        for old in folder.glob("timing_*.json"):
            old.unlink()
        for r in range(args.rounds):
            order = VARIANTS[r:] + VARIANTS[:r]  # rotate, so no variant always runs last (hottest)
            for name in order:
                run_child("--phase", "timing", "--variant", name, "--round", str(r + 1))
    (args.out / "benchmark.json").write_text(json.dumps(summarize_all(folder, args), indent=2) + "\n")
    print(f"wrote {args.out / 'benchmark.json'}")


if __name__ == "__main__":
    main()
