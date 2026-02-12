from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.core.documents import DocumentStore
from app.core.ingest import IngestPipeline
from app.core.langextract_enricher import LangExtractEnricher
from app.schemas.config import AppConfig, ProviderConfig, RetrievalDefaults


class DummyRetrievalEngine:
    def __init__(self) -> None:
        self.indexed_docs = []

    def index_documents(self, docs) -> None:
        self.indexed_docs.extend(docs)


def _build_enabled_config() -> AppConfig:
    return AppConfig(
        provider=ProviderConfig(
            name="Ollama",
            base_url="http://localhost:11434",
            gatherer_model="llama3.2:3b",
        ),
        retrieval=RetrievalDefaults(
            langextract_enabled=True,
            langextract_profile_default="generic_entities_v1",
            langextract_model_source="gatherer",
            langextract_timeout_sec=20,
            langextract_max_chars=12000,
            langextract_max_synthetic_facts=200,
        ),
    )


def test_resolve_model_openai_compatible() -> None:
    enricher = LangExtractEnricher(data_dir=Path("/tmp"))
    provider = ProviderConfig(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        gatherer_model="openai/gpt-4o-mini",
    )

    resolved = enricher.resolve_model(provider, model_source="gatherer")

    assert resolved["status"] == "ready"
    assert resolved["provider"] == "openai"
    assert resolved["model_id"] == "openai/gpt-4o-mini"
    assert resolved["provider_kwargs"]["base_url"] == "https://openrouter.ai/api/v1"


def test_resolve_model_unsupported_provider_skips() -> None:
    enricher = LangExtractEnricher(data_dir=Path("/tmp"))
    provider = ProviderConfig(
        name="Anthropic",
        base_url="https://api.anthropic.com",
        gatherer_model="claude-3-5-sonnet-latest",
    )

    resolved = enricher.resolve_model(provider, model_source="gatherer")

    assert resolved["status"] == "skipped_unsupported_provider"


def test_synthetic_sections_are_deterministic_and_capped() -> None:
    enricher = LangExtractEnricher(data_dir=Path("/tmp"))
    raw = SimpleNamespace(
        document_id="doc_1",
        text="Contract text",
        extractions=[
            SimpleNamespace(
                extraction_class="relation",
                extraction_text="Acme signs with Northwind",
                attributes={"source": "Acme", "target": "Northwind", "type": "contract"},
            ),
            SimpleNamespace(
                extraction_class="entity",
                extraction_text="Acme",
                attributes={"type": "organization"},
            ),
            SimpleNamespace(
                extraction_class="claim",
                extraction_text="Term is three years",
                attributes={"term": "3 years"},
            ),
        ],
    )

    normalized = enricher.normalize_result(raw)
    sections = enricher.to_synthetic_sections(normalized, max_synthetic_facts=2)

    rendered = "\n".join(sections)
    assert rendered.count("- ") == 2
    assert "## LangExtract Entities" in rendered


def test_ingest_fail_open_when_extraction_fails(tmp_path: Path) -> None:
    store = DocumentStore(path=tmp_path / "docs.db")
    retrieval = DummyRetrievalEngine()
    cfg = _build_enabled_config()
    pipeline = IngestPipeline(
        store=store,
        retrieval=retrieval,
        config_getter=lambda: cfg,
        data_dir=tmp_path,
    )

    def fail_extract(**kwargs):
        _ = kwargs
        return {
            "status": "failed_timeout",
            "profile": "generic_entities_v1",
            "model_source": "gatherer",
            "model_id": "llama3.2:3b",
            "provider": "ollama",
            "entities_count": 0,
            "relations_count": 0,
            "claims_count": 0,
            "warnings_count": 0,
            "synthetic_sections": [],
            "error": "LangExtract timed out after 20s",
            "raw": None,
        }

    pipeline._langextract.extract = fail_extract  # type: ignore[method-assign]

    result = pipeline.ingest_text(title="Policy", text="Security policy text.", sync=True)
    doc = store.get(result.document_id)

    assert doc is not None
    assert result.chunk_count >= 1
    assert doc.metadata["langextract_status"] == "failed_timeout"
    assert "timed out" in doc.metadata["langextract_error"]
    assert doc.metadata["processing_status"] == "ready"


def test_ingest_appends_synthetic_sections_on_success(tmp_path: Path) -> None:
    store = DocumentStore(path=tmp_path / "docs.db")
    retrieval = DummyRetrievalEngine()
    cfg = _build_enabled_config()
    pipeline = IngestPipeline(
        store=store,
        retrieval=retrieval,
        config_getter=lambda: cfg,
        data_dir=tmp_path,
    )

    def success_extract(**kwargs):
        _ = kwargs
        return {
            "status": "ok",
            "profile": "generic_entities_v1",
            "model_source": "gatherer",
            "model_id": "llama3.2:3b",
            "provider": "ollama",
            "entities_count": 1,
            "relations_count": 0,
            "claims_count": 1,
            "warnings_count": 0,
            "synthetic_sections": [
                "## LangExtract Entities\n- ENTITY: Acme | type=organization",
                "## LangExtract Claims\n- CLAIM: Acme signed the agreement",
            ],
            "error": None,
            "raw": {"extractions": []},
        }

    pipeline._langextract.extract = success_extract  # type: ignore[method-assign]

    result = pipeline.ingest_text(title="Agreement", text="Agreement text body.", sync=True)
    doc = store.get(result.document_id)

    assert doc is not None
    assert doc.metadata["langextract_status"] == "ok"
    assert doc.metadata["langextract_entities_count"] == "1"
    assert "[[LANGEXTRACT_SYNTHETIC_FACTS_BEGIN]]" in doc.text
    assert "ENTITY: Acme" in doc.text
    assert "langextract_artifact_path" in doc.metadata
