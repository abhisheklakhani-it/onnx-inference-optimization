import pytest
from fastapi.testclient import TestClient

from onnxopt import api
from onnxopt.classifier import OnnxClassifier


@pytest.fixture
def client(tiny_model_dir, monkeypatch):
    folder, _ = tiny_model_dir
    monkeypatch.setenv("MODEL_PATH", str(folder / "model.onnx"))
    api.get_classifier.cache_clear()
    app = api.app
    app.dependency_overrides[api.get_classifier] = lambda: OnnxClassifier(folder / "model.onnx", threads=1)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_names_the_model(client):
    assert client.get("/health").json() == {"status": "ok", "model": "model.onnx"}


def test_predict_returns_one_prediction_per_text(client):
    response = client.post("/predict", json={"texts": ["Tor Tor", "Aktie Börse"]})
    assert response.status_code == 200
    body = response.json()
    assert [p["label"] in {"Sport", "Wirtschaft"} for p in body["predictions"]] == [True, True]
    assert body["model"] == "model.onnx"
    assert body["ms"] >= 0


@pytest.mark.parametrize("texts", [[], [""], ["x"] * (api.MAX_TEXTS + 1)])
def test_predict_rejects_bad_input(client, texts):
    assert client.post("/predict", json={"texts": texts}).status_code == 422


def test_missing_model_gives_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.onnx"))
    api.get_classifier.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "model missing"
    assert client.post("/predict", json={"texts": ["Tor"]}).status_code == 503
    api.get_classifier.cache_clear()
