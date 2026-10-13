import numpy as np
import torch

from onnxopt.classifier import OnnxClassifier, encode, load_tokenizer, softmax

TEXTS = ["Tor", "Spiel Tor Tor Börse", "Aktie", "Börse Aktie Aktie Spiel Tor"]  # different lengths -> padding


def test_softmax_rows_sum_to_one_even_for_large_logits():
    probs = softmax(np.array([[1000.0, 1001.0], [0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], [0.5, 0.5])


def test_encode_pads_to_longest_and_truncates(tiny_model_dir):
    folder, _ = tiny_model_dir
    batch = encode(load_tokenizer(folder, max_length=4), ["Tor", "Tor Spiel Aktie Börse"])
    assert batch["input_ids"].tolist() == [[2, 4, 3, 0], [2, 4, 6, 3]]  # [CLS] ... [SEP], [PAD]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]


def test_onnx_matches_pytorch_on_a_padded_batch(tiny_model_dir):
    folder, model = tiny_model_dir
    clf = OnnxClassifier(folder / "model.onnx")
    inputs = encode(clf.tokenizer, TEXTS)
    with torch.inference_mode():
        ref = model(**{k: torch.from_numpy(v) for k, v in inputs.items()}).logits.numpy()
    assert np.allclose(clf.logits(TEXTS), ref, atol=1e-5)


def test_padding_does_not_change_a_text_s_answer(tiny_model_dir):
    folder, _ = tiny_model_dir
    clf = OnnxClassifier(folder / "model.onnx")
    alone = clf.logits(["Tor"])
    in_batch = clf.logits(["Tor", "Börse Aktie Aktie Spiel Tor"])[:1]
    assert np.allclose(alone, in_batch, atol=1e-5)


def test_predict_returns_label_names_and_confidence(tiny_model_dir):
    folder, _ = tiny_model_dir
    results = OnnxClassifier(folder / "model.onnx", threads=1).predict(TEXTS)
    assert len(results) == len(TEXTS)
    for r in results:
        assert r["label"] in {"Sport", "Wirtschaft"}
        assert 0.5 <= r["confidence"] <= 1.0
