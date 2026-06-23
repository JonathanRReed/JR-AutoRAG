"""LangExtract adapter for optional ingestion-time enrichment."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from ..schemas.config import AppConfig, ProviderConfig
from .langextract_profiles import resolve_profile
from .providers import resolve_provider_api_key


class LangExtractEnricher:
    """Best-effort LangExtract integration with provider reuse and fail-open behavior."""

    SUPPORTED_MODEL_SOURCES = {"planner", "gatherer", "generator"}

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = Path(data_dir or Path.cwd() / "data")
        self._artifact_dir = self._data_dir / "extractions"
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

    def is_enabled(
        self,
        cfg: AppConfig,
        per_doc_override: dict[str, str | None] | None = None,
    ) -> bool:
        _ = per_doc_override
        return bool(getattr(cfg.retrieval, "langextract_enabled", False))

    def resolve_model(
        self,
        provider: ProviderConfig | None,
        model_source: str = "gatherer",
    ) -> dict[str, Any]:
        if provider is None:
            return {
                "status": "skipped_no_provider",
                "reason": "No active provider configured",
                "provider": None,
                "model_id": None,
                "provider_kwargs": {},
            }

        source = model_source if model_source in self.SUPPORTED_MODEL_SOURCES else "gatherer"
        source_field = f"{source}_model"
        model_id = (
            getattr(provider, source_field, None)
            or provider.gatherer_model
            or provider.generator_model
            or provider.planner_model
        )
        if not model_id:
            return {
                "status": "skipped_no_model",
                "reason": f"No model configured for source '{source}'",
                "provider": None,
                "model_id": None,
                "provider_kwargs": {},
            }

        provider_name = (provider.name or "").strip().lower()
        base_url = str(provider.base_url)
        base_lower = base_url.lower()

        resolved_key = resolve_provider_api_key(
            provider.name,
            base_url,
            provider.api_key,
        )

        if "ollama" in provider_name and "ollama.com" not in base_lower:
            kwargs: dict[str, Any] = {
                "base_url": base_url,
                "model_url": base_url,
                "timeout": 60,
            }
            if resolved_key:
                kwargs["api_key"] = resolved_key
            return {
                "status": "ready",
                "reason": None,
                "provider": "ollama",
                "model_id": model_id,
                "provider_kwargs": kwargs,
            }

        openai_compat_name = any(key in provider_name for key in ("openai", "openrouter", "lm studio", "lmstudio"))
        openai_compat_url = any(
            key in base_lower
            for key in ("api.openai.com", "openrouter.ai", "/v1", "localhost:1234", "127.0.0.1:1234")
        )
        if openai_compat_name or openai_compat_url:
            kwargs = {
                "base_url": base_url,
            }
            if resolved_key:
                kwargs["api_key"] = resolved_key
            return {
                "status": "ready",
                "reason": None,
                "provider": "openai",
                "model_id": model_id,
                "provider_kwargs": kwargs,
            }

        return {
            "status": "skipped_unsupported_provider",
            "reason": f"Unsupported provider shape: {provider.name}",
            "provider": None,
            "model_id": model_id,
            "provider_kwargs": {},
        }

    def extract(
        self,
        text: str,
        provider: ProviderConfig | None,
        profile: str,
        prompt_override: str | None,
        timeout: int,
        model_source: str,
        max_chars: int,
        max_synthetic_facts: int,
    ) -> dict[str, Any]:
        extraction_profile = resolve_profile(profile)
        effective_profile = extraction_profile["name"]
        resolved_model = self.resolve_model(provider, model_source=model_source)

        result: dict[str, Any] = {
            "status": resolved_model["status"],
            "profile": effective_profile,
            "model_source": model_source,
            "model_id": resolved_model.get("model_id"),
            "provider": resolved_model.get("provider"),
            "entities": [],
            "relations": [],
            "claims": [],
            "warnings": [],
            "entities_count": 0,
            "relations_count": 0,
            "claims_count": 0,
            "warnings_count": 0,
            "error": resolved_model.get("reason"),
            "raw": None,
            "synthetic_sections": [],
            "truncated_chars": 0,
        }

        if resolved_model["status"] != "ready":
            return result

        try:
            import langextract as lx
            from langextract.data import ExampleData, Extraction
            from langextract.factory import ModelConfig
        except Exception as exc:  # pragma: no cover - import-path dependent
            result["status"] = "skipped_dependency_unavailable"
            result["error"] = f"langextract unavailable: {exc}"
            return result

        bounded_text = text[: max(1, max_chars)]
        result["truncated_chars"] = max(0, len(text) - len(bounded_text))
        prompt_description = (prompt_override or "").strip() or extraction_profile["prompt_description"]
        examples = self._build_examples(extraction_profile, ExampleData, Extraction)
        config = ModelConfig(
            model_id=resolved_model["model_id"],
            provider=resolved_model["provider"],
            provider_kwargs=resolved_model["provider_kwargs"],
        )

        def _run_extract() -> Any:
            return lx.extract(
                text_or_documents=bounded_text,
                prompt_description=prompt_description,
                examples=examples,
                config=config,
                show_progress=False,
            )

        try:
            raw_result = self._run_with_timeout(_run_extract, timeout=max(1, int(timeout)))
        except TimeoutError as exc:
            result["status"] = "failed_timeout"
            result["error"] = str(exc)
            return result
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            return result

        normalized = self.normalize_result(raw_result)
        synthetic_sections = self.to_synthetic_sections(normalized, max_synthetic_facts=max_synthetic_facts)

        result.update(
            {
                "status": "ok",
                "entities": normalized["entities"],
                "relations": normalized["relations"],
                "claims": normalized["claims"],
                "warnings": normalized["warnings"],
                "entities_count": len(normalized["entities"]),
                "relations_count": len(normalized["relations"]),
                "claims_count": len(normalized["claims"]),
                "warnings_count": len(normalized["warnings"]),
                "raw": normalized["raw"],
                "synthetic_sections": synthetic_sections,
                "error": None,
            }
        )
        return result

    def normalize_result(self, raw_result: Any) -> dict[str, Any]:
        document = raw_result[0] if isinstance(raw_result, list) and raw_result else raw_result
        extractions = getattr(document, "extractions", None) or []

        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        claims: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        for extraction in extractions:
            extraction_class = str(getattr(extraction, "extraction_class", "")).strip().lower()
            extraction_text = str(getattr(extraction, "extraction_text", "")).strip()
            if not extraction_text:
                continue
            attributes = self._normalize_attributes(getattr(extraction, "attributes", None))
            payload = {
                "text": extraction_text,
                "attributes": attributes,
                "class": extraction_class or "entity",
            }

            if extraction_class in {"relation", "relationship"}:
                relations.append(
                    {
                        **payload,
                        "source": attributes.get("source", ""),
                        "target": attributes.get("target", ""),
                        "type": attributes.get("type", attributes.get("relation", "")),
                    }
                )
            elif extraction_class in {"claim", "fact", "assertion"}:
                claims.append(payload)
            elif extraction_class in {"warning", "risk", "alert", "compliance_warning"}:
                warnings.append(payload)
            else:
                entities.append(payload)

        entities.sort(key=lambda item: (item["text"].lower(), json.dumps(item["attributes"], sort_keys=True)))
        relations.sort(
            key=lambda item: (
                str(item.get("source", "")).lower(),
                str(item.get("target", "")).lower(),
                str(item.get("type", "")).lower(),
                item["text"].lower(),
            )
        )
        claims.sort(key=lambda item: item["text"].lower())
        warnings.sort(key=lambda item: item["text"].lower())

        return {
            "entities": entities,
            "relations": relations,
            "claims": claims,
            "warnings": warnings,
            "raw": self._serialize_document(document),
        }

    def to_synthetic_sections(
        self,
        normalized: dict[str, Any],
        max_synthetic_facts: int,
    ) -> list[str]:
        remaining = max(0, int(max_synthetic_facts))
        sections: list[str] = []

        def add_section(title: str, rows: list[str]) -> None:
            nonlocal remaining
            if remaining <= 0 or not rows:
                return
            allowed = rows[:remaining]
            sections.append("\n".join([title, *allowed]))
            remaining -= len(allowed)

        entity_rows = [
            f"- ENTITY: {item['text']}{self._attrs_suffix(item['attributes'])}"
            for item in normalized.get("entities", [])
        ]
        relation_rows = [
            (
                "- RELATION: "
                f"{item.get('source') or '?'} -> {item.get('target') or '?'}"
                f" | type={item.get('type') or 'unspecified'}"
                f" | text={item['text']}"
            )
            for item in normalized.get("relations", [])
        ]
        claim_rows = [
            f"- CLAIM: {item['text']}{self._attrs_suffix(item['attributes'])}"
            for item in normalized.get("claims", [])
        ]
        warning_rows = [
            f"- WARNING: {item['text']}{self._attrs_suffix(item['attributes'])}"
            for item in normalized.get("warnings", [])
        ]

        add_section("## LangExtract Entities", entity_rows)
        add_section("## LangExtract Relations", relation_rows)
        add_section("## LangExtract Claims", claim_rows)
        add_section("## LangExtract Warnings", warning_rows)

        return sections

    def artifact_path(self, doc_id: str) -> Path:
        return self._artifact_dir / f"{doc_id}.json"

    def persist_artifact(self, doc_id: str, payload: dict[str, Any]) -> Path:
        path = self.artifact_path(doc_id)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _build_examples(self, profile: dict[str, Any], example_cls: Any, extraction_cls: Any) -> list[Any]:
        examples: list[Any] = []
        for sample in profile.get("examples", []):
            extractions = []
            for item in sample.get("extractions", []):
                extractions.append(
                    extraction_cls(
                        extraction_class=item.get("extraction_class", "entity"),
                        extraction_text=item.get("extraction_text", ""),
                        attributes=item.get("attributes", {}),
                    )
                )
            examples.append(example_cls(text=sample.get("text", ""), extractions=extractions))
        return examples

    def _run_with_timeout(self, fn: Any, timeout: int) -> Any:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="langextract")
        future = executor.submit(fn)
        timed_out = False
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            timed_out = True
            if future.cancel():
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            raise TimeoutError(f"LangExtract timed out after {timeout}s") from exc
        finally:
            if not timed_out:
                executor.shutdown(wait=True, cancel_futures=True)

    def _normalize_attributes(self, attributes: Any) -> dict[str, str]:
        if not isinstance(attributes, dict):
            return {}
        normalized: dict[str, str] = {}
        for key, value in attributes.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            if isinstance(value, list):
                rendered = ", ".join(str(item).strip() for item in value if str(item).strip())
            else:
                rendered = str(value).strip()
            if rendered:
                normalized[key_text] = rendered
        return dict(sorted(normalized.items(), key=lambda pair: pair[0]))

    def _attrs_suffix(self, attributes: dict[str, str]) -> str:
        if not attributes:
            return ""
        rendered = "; ".join(f"{key}={value}" for key, value in sorted(attributes.items()))
        return f" | {rendered}" if rendered else ""

    def _serialize_document(self, document: Any) -> dict[str, Any]:
        if document is None:
            return {"extractions": []}
        serialized: dict[str, Any] = {
            "document_id": getattr(document, "document_id", None),
            "text": getattr(document, "text", None),
            "extractions": [],
        }
        for extraction in getattr(document, "extractions", None) or []:
            serialized["extractions"].append(
                {
                    "extraction_class": getattr(extraction, "extraction_class", None),
                    "extraction_text": getattr(extraction, "extraction_text", None),
                    "attributes": self._normalize_attributes(getattr(extraction, "attributes", None)),
                }
            )
        return serialized
