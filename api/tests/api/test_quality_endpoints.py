from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.core.config_store import ConfigStore
from app.core.documents import Document, DocumentStore
from app.core.experiments import ExperimentConfig, ExperimentRunStore, LocalExperimentRunner
from app.core.auth import APIKeyAuth
from app.core.ingest import IngestPipeline
from app.core.security_middleware import _resolve_required_scope
from app.main import app
from app.routers.config import quality_recommendations
from app.services import get_container


class FakeContainer:
    def __init__(self, tmp_path: Path) -> None:
        self.config_store = ConfigStore(tmp_path / "config.json")
        self.document_store = DocumentStore(tmp_path / "documents.db")
        self.applied = False

    def apply_config(self, _cfg) -> None:
        self.applied = True


class FakeRetrieval:
    def __init__(self) -> None:
        self.indexed = False

    def index_documents(self, _docs) -> None:
        self.indexed = True


def test_quality_endpoints_without_retrieval_startup(tmp_path: Path) -> None:
    container = FakeContainer(tmp_path)
    doc = container.document_store.add(
        title="Quality Notes",
        text="# Quality\n\nThis corpus tests structured parsing.",
        metadata={
            "parser_provider": "native",
            "parser_engine": "test",
            "parser_confidence": "0.92",
            "processing_status": "ready",
        },
    )

    def override_container() -> FakeContainer:
        return container

    app.dependency_overrides[get_container] = override_container
    try:
        client = TestClient(app)

        preview = client.get(f"/documents/{doc.id}/preview")
        assert preview.status_code == 200
        preview_data = preview.json()
        assert preview_data["document_id"] == doc.id
        assert preview_data["block_count"] >= 1

        recommendations = client.get("/config/recommendations")
        assert recommendations.status_code == 200
        assert recommendations.json()["document_count"] == 1

        experiment = client.post("/experiments", json={"name": "Quality Matrix"})
        assert experiment.status_code == 200
        experiment_data = experiment.json()
        assert experiment_data["status"] == "completed"
        assert experiment_data["metrics"]
        assert any(trace.startswith("experiment:") for trace in experiment_data["traces"])

        promoted = client.post(f"/experiments/{experiment_data['id']}/promote")
        assert promoted.status_code == 200
        assert promoted.json()["promoted_preset"]
        assert container.applied is True
    finally:
        app.dependency_overrides.clear()


def test_experiments_endpoints_require_admin_scope() -> None:
    auth = APIKeyAuth(enabled=True)
    read_key, _ = auth.generate_key("reader", scopes=["read"])
    admin_key, _ = auth.generate_key("admin", scopes=["admin"])

    assert _resolve_required_scope("/experiments", "GET") == "admin"
    assert _resolve_required_scope("/experiments", "POST") == "admin"
    assert _resolve_required_scope("/experiments/run-1", "GET") == "admin"
    assert _resolve_required_scope("/experiments/run-1/promote", "POST") == "admin"
    assert auth.verify(read_key, required_scope=_resolve_required_scope("/experiments/run-1/promote", "POST")) is False
    assert auth.verify(admin_key, required_scope=_resolve_required_scope("/experiments/run-1/promote", "POST")) is True


def test_document_preview_falls_back_from_stored_text(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "documents.db")
    doc = Document(id="doc-1", title="Stored", text="## Heading\n\nBody", metadata={})
    store.upsert(doc)

    loaded = store.get("doc-1")
    assert loaded is not None
    assert loaded.text.startswith("## Heading")


def test_binary_parser_fallback_does_not_replace_extracted_pdf_text(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "documents.db")
    retrieval = FakeRetrieval()
    pipeline = IngestPipeline(store, retrieval, data_dir=tmp_path)

    result = pipeline.ingest_file(
        title="Broken PDF",
        content=b"%PDF-1.4\nnot a valid body",
        metadata={"filename": "broken.pdf", "content_type": "application/pdf"},
        sync=True,
    )

    doc = store.get(result.document_id)
    assert doc is not None
    assert "PDF extraction failed" in doc.text
    assert "%PDF-1.4" not in doc.text
    assert doc.metadata["parser_engine"] in {"none", "pypdf", "pdftotext"}


def test_recommendations_filter_documents_by_acl(monkeypatch, tmp_path: Path) -> None:
    container = FakeContainer(tmp_path)
    visible = container.document_store.add("Visible", "text", {"parser_provider": "native"})
    hidden = container.document_store.add("Hidden", "text", {"parser_provider": "docling"})

    class FakeAuth:
        def require_auth(self) -> bool:
            return True

    class FakeEnforcer:
        def check_access(self, document_id: str, _user_id: str, _action: str):
            return document_id == visible.id, "test"

    monkeypatch.setattr("app.routers.config.get_auth", lambda: FakeAuth())
    monkeypatch.setattr("app.routers.config.get_acl_enforcer", lambda default_public=True: FakeEnforcer())

    request = SimpleNamespace(state=SimpleNamespace(scopes=["read"], user_id="user-a"))
    payload = quality_recommendations(request=request, container=container)

    assert payload["document_count"] == 1
    assert payload["parser_counts"] == {"native": 1}
    assert hidden.id != visible.id


def test_experiment_runs_are_owner_scoped(tmp_path: Path) -> None:
    container = FakeContainer(tmp_path)
    store = ExperimentRunStore(tmp_path / "experiments.json")
    runner = LocalExperimentRunner(store)

    run = runner.run(
        ExperimentConfig(name="Owner A", parser=["native"]),
        container.config_store.read(),
        doc_count=0,
        owner_id="owner-a",
    )

    assert store.get(str(run["id"]), owner_id="owner-a") is not None
    assert store.get(str(run["id"]), owner_id="owner-b") is None
    assert store.list(owner_id="owner-b") == []
