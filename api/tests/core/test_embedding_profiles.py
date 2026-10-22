"""Offline contracts for modern embedding profiles; no model downloads."""
import numpy as np
import pytest

from app.embedding_profiles import (
    DEFAULT_EMBEDDING_MODEL, EMBEDDING_PROFILES, ProfiledEmbedder,
    embedding_fingerprint, encode_query,
)


class RecordingModel:
    def __init__(self, dimensions=1024):
        self.dimensions = dimensions
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), self.dimensions), dtype=np.float32)


@pytest.mark.parametrize("model_id", list(EMBEDDING_PROFILES))
def test_query_and_document_prompts_are_distinct(model_id):
    profile = EMBEDDING_PROFILES[model_id]
    backend = RecordingModel(profile.dimensions)
    embedder = ProfiledEmbedder(backend, profile)
    docs = embedder.encode(["A document"])
    query = embedder.encode_query(["A question"])
    assert backend.calls[0][1]["prompt"] == profile.document_prompt
    assert backend.calls[1][1]["prompt"] == profile.query_prompt
    assert profile.query_prompt != profile.document_prompt
    assert np.allclose(np.linalg.norm(docs, axis=1), 1)
    assert np.allclose(np.linalg.norm(query, axis=1), 1)


def test_string_input_and_legacy_query_fallback():
    profile = EMBEDDING_PROFILES[DEFAULT_EMBEDDING_MODEL]
    embedder = ProfiledEmbedder(RecordingModel(profile.dimensions), profile)
    assert embedder.encode_query("hello").shape == (profile.dimensions,)
    legacy = RecordingModel(768)
    legacy.encode_query = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy prompt changed"))
    assert encode_query(legacy, ["hello"], convert_to_numpy=True).shape == (1, 768)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0])
def test_invalid_vectors_fail_closed(bad):
    dims = EMBEDDING_PROFILES[DEFAULT_EMBEDDING_MODEL].dimensions
    backend = RecordingModel(dims)
    backend.encode = lambda texts, **kwargs: np.full((len(texts), dims), bad)
    with pytest.raises(ValueError):
        ProfiledEmbedder(backend, EMBEDDING_PROFILES[DEFAULT_EMBEDDING_MODEL]).encode(["x"])


def test_wrong_dimensions_fail_closed():
    with pytest.raises(ValueError, match="dimension"):
        ProfiledEmbedder(RecordingModel(7), EMBEDDING_PROFILES[DEFAULT_EMBEDDING_MODEL]).encode(["x"])


def test_fingerprint_distinguishes_models_and_legacy():
    values = {embedding_fingerprint(model) for model in EMBEDDING_PROFILES}
    assert len(values) == 3
    assert embedding_fingerprint("BAAI/bge-base-en-v1.5") not in values


def test_new_configuration_uses_balanced_default():
    from app.core.hybrid_retrieval import HybridConfig
    from app.schemas.config import RetrievalDefaults
    assert DEFAULT_EMBEDDING_MODEL == "google/embeddinggemma-2"
    assert HybridConfig().embedding_model == DEFAULT_EMBEDDING_MODEL
    assert RetrievalDefaults().embedding_model == DEFAULT_EMBEDDING_MODEL


def test_old_partial_config_preserves_legacy_default(tmp_path):
    from app.core.config_store import ConfigStore
    path = tmp_path / "config.json"
    path.write_text('{"retrieval": {"dense_k": 5}}')
    assert ConfigStore(path).read().retrieval.embedding_model == "BAAI/bge-base-en-v1.5"


def test_gemma_loader_pins_revision_and_disables_unused_modalities():
    from app.embedding_profiles import load_profiled_embedder
    calls = []
    def factory(model, **kwargs):
        calls.append((model, kwargs))
        return RecordingModel(768)
    load_profiled_embedder(factory, "google/embeddinggemma-2")
    options = calls[0][1]
    assert len(options["revision"]) == 40
    assert options["trust_remote_code"] is False
    assert options["model_kwargs"]["dtype"] == "float32"
    assert options["config_kwargs"] == {"vision_config": None, "audio_config": None}


def test_overlong_input_fails_before_encoding():
    backend = RecordingModel()
    backend.max_seq_length = 4
    backend.tokenizer = lambda text, **kwargs: {"input_ids": list(range(5))}
    with pytest.raises(ValueError, match="exceeds"):
        ProfiledEmbedder(backend, EMBEDDING_PROFILES[DEFAULT_EMBEDDING_MODEL]).encode_query(["hello"])
    assert backend.calls == []


def test_model_change_cannot_keep_incompatible_index(tmp_path):
    from app.core.documents import DocumentStore
    from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine
    engine = HybridRetrievalEngine(DocumentStore(tmp_path / "docs.db"),
        HybridConfig(embedding_model="", use_reranking=False), persist_path=None)
    engine._chunks = [("sentinel", object())]
    with pytest.raises(ValueError, match="require rebuilding"):
        engine.reconfigure(HybridConfig(embedding_model=DEFAULT_EMBEDDING_MODEL), rebuild=False)
    assert engine._config.embedding_model == ""


def test_preset_metadata_uses_actual_dimensions():
    from app.core.hybrid_retrieval import EmbeddingModelPreset
    for model_id, profile in EMBEDDING_PROFILES.items():
        assert EmbeddingModelPreset.get_info(model_id)["dimensions"] == profile.dimensions


def test_failed_model_switch_retains_existing_index(tmp_path, monkeypatch):
    from app.core.documents import DocumentStore
    from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine
    engine = HybridRetrievalEngine(DocumentStore(tmp_path / "docs.db"),
        HybridConfig(embedding_model="", use_reranking=False), persist_path=None)
    chunks = [("sentinel", object())]
    old_embedder = RecordingModel(768)
    engine._chunks = chunks
    engine._embedder = old_embedder
    monkeypatch.setattr(engine, "_init_models", lambda: None)
    with pytest.raises(ValueError, match="previous model restored"):
        engine.reconfigure(HybridConfig(embedding_model=DEFAULT_EMBEDDING_MODEL, use_reranking=False))
    assert engine._config.embedding_model == ""
    assert engine._embedder is old_embedder
    assert engine._chunks is chunks
