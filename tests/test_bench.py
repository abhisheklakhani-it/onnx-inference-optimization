import numpy as np

from onnxopt.bench import run_batches, summarize


def test_run_batches_keeps_input_order_and_times_every_batch():
    calls = []

    def fake_logits(texts):
        calls.append(len(texts))
        return np.array([[float(len(t)), 0.0] for t in texts])

    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    seconds, logits = run_batches(fake_logits, texts, batch_size=2, warmup=1)
    assert calls == [2, 2, 2, 1]  # 1 warm-up batch, then 3 timed batches
    assert len(seconds) == 3
    assert logits[:, 0].tolist() == [1, 2, 3, 4, 5]


def test_summarize_reports_percentiles_and_throughput():
    stats = summarize([0.1] * 19 + [1.0], articles=40)
    assert stats["batches"] == 20
    assert stats["p50_ms"] == 100.0
    assert 100.0 < stats["p95_ms"] <= 1000.0
    assert stats["articles_per_s"] == round(40 / 2.9, 1)
