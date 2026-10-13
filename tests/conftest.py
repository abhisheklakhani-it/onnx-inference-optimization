"""A tiny DistilBERT built from scratch and exported to ONNX: no downloads, no Day 5 model needed."""

import json

import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import DistilBertConfig, DistilBertForSequenceClassification

from onnxopt.classifier import encode, load_tokenizer
from onnxopt.export import export_onnx

VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "tor": 4, "aktie": 5, "spiel": 6, "börse": 7}
LABELS = {0: "Sport", 1: "Wirtschaft"}


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny")
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.save(str(folder / "tokenizer.json"))

    torch.manual_seed(0)
    config = DistilBertConfig(vocab_size=len(VOCAB), dim=32, n_layers=2, n_heads=2, hidden_dim=64,
                              max_position_embeddings=64, num_labels=2, id2label=LABELS,
                              label2id={v: k for k, v in LABELS.items()}, pad_token_id=0,
                              attn_implementation="eager")
    model = DistilBertForSequenceClassification(config).eval()
    (folder / "config.json").write_text(json.dumps({"id2label": {str(k): v for k, v in LABELS.items()}}))

    sample = {k: torch.from_numpy(v) for k, v in encode(load_tokenizer(folder), ["Tor", "Aktie Börse Spiel"]).items()}
    export_onnx(model, sample, folder / "model.onnx")
    return folder, model
