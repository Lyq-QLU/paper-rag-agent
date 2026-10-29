from types import SimpleNamespace

import pytest

import src.rag_pipeline as pipeline


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    monkeypatch.setattr(pipeline, "_EMBEDDING_MODELS", {})
    monkeypatch.setattr(pipeline, "snapshot_download", lambda *args, **kwargs: "/cached/model")


def test_cached_embedding_loads_once_on_cpu(monkeypatch):
    calls = []
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(is_meta=False)])
    def load(reference, **kwargs):
        calls.append((reference, kwargs))
        return model
    monkeypatch.setattr(pipeline, "SentenceTransformer", load)
    assert pipeline.PaperRAG()._get_embedding_model() is model
    assert pipeline.PaperRAG()._get_embedding_model() is model
    assert len(calls) == 1
    assert calls[0][0] == "/cached/model"
    assert calls[0][1]["device"] == "cpu"


def test_failed_meta_load_is_not_cached_and_can_retry(monkeypatch):
    def fail(*args, **kwargs):
        raise NotImplementedError("Cannot copy out of meta tensor")
    monkeypatch.setattr(pipeline, "SentenceTransformer", fail)
    with pytest.raises(RuntimeError, match="Embedding 模型加载失败"):
        pipeline.PaperRAG()._get_embedding_model()
    assert pipeline._EMBEDDING_MODELS == {}
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(is_meta=False)])
    monkeypatch.setattr(pipeline, "SentenceTransformer", lambda *args, **kwargs: model)
    assert pipeline.PaperRAG()._get_embedding_model() is model


def test_unmaterialized_weights_are_never_cached(monkeypatch):
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(is_meta=True)])
    monkeypatch.setattr(pipeline, "SentenceTransformer", lambda *args, **kwargs: model)
    with pytest.raises(RuntimeError, match="meta tensor"):
        pipeline.PaperRAG()._get_embedding_model()
    assert pipeline._EMBEDDING_MODELS == {}
