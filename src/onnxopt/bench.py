"""Timing helpers: run a model over texts in fixed batches and summarise the latencies."""

import time
from collections.abc import Callable

import numpy as np


def run_batches(logits_fn: Callable[[list[str]], np.ndarray], texts: list[str], batch_size: int,
                warmup: int = 3) -> tuple[list[float], np.ndarray]:
    """Time logits_fn on consecutive batches; returns per-batch seconds and all logits in input order."""
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    for batch in batches[:warmup]:
        logits_fn(batch)
    seconds, outputs = [], []
    for batch in batches:
        start = time.perf_counter()
        outputs.append(logits_fn(batch))
        seconds.append(time.perf_counter() - start)
    return seconds, np.concatenate(outputs)


def summarize(seconds: list[float], articles: int) -> dict:
    ms = np.array(seconds) * 1000
    return {
        "batches": len(seconds),
        "p50_ms": round(float(np.percentile(ms, 50)), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
        "mean_ms": round(float(ms.mean()), 1),
        "articles_per_s": round(articles / float(np.sum(seconds)), 1),
    }
