"""Built-in extraction profiles for LangExtract enrichment."""

from __future__ import annotations

from typing import TypedDict


class ProfileExtractionExample(TypedDict):
    extraction_class: str
    extraction_text: str
    attributes: dict[str, str]


class ProfileExample(TypedDict):
    text: str
    extractions: list[ProfileExtractionExample]


class ExtractionProfile(TypedDict):
    name: str
    prompt_description: str
    examples: list[ProfileExample]


LANGEXTRACT_PROFILES: dict[str, ExtractionProfile] = {
    "generic_entities_v1": {
        "name": "generic_entities_v1",
        "prompt_description": (
            "Extract entities, relationships, claims, and warnings from text. "
            "Return concise, factual items only. Use extraction_class values: "
            "entity, relation, claim, warning."
        ),
        "examples": [
            {
                "text": "Acme Corp signed a 3-year supply contract with Northwind on March 1, 2025.",
                "extractions": [
                    {
                        "extraction_class": "entity",
                        "extraction_text": "Acme Corp",
                        "attributes": {"type": "organization"},
                    },
                    {
                        "extraction_class": "entity",
                        "extraction_text": "Northwind",
                        "attributes": {"type": "organization"},
                    },
                    {
                        "extraction_class": "relation",
                        "extraction_text": "Acme Corp signed contract with Northwind",
                        "attributes": {
                            "source": "Acme Corp",
                            "target": "Northwind",
                            "type": "contract",
                            "term": "3 years",
                            "date": "2025-03-01",
                        },
                    },
                    {
                        "extraction_class": "claim",
                        "extraction_text": "Contract term is 3 years.",
                        "attributes": {"confidence": "high"},
                    },
                ],
            }
        ],
    },
    "compliance_risk_v1": {
        "name": "compliance_risk_v1",
        "prompt_description": (
            "Extract compliance-relevant entities, obligations, deadlines, and risks. "
            "Use extraction_class values: entity, relation, claim, warning. "
            "Prefer explicit policy, legal, privacy, or security risks."
        ),
        "examples": [
            {
                "text": "Vendor must notify customers of a security incident within 72 hours.",
                "extractions": [
                    {
                        "extraction_class": "entity",
                        "extraction_text": "Vendor",
                        "attributes": {"type": "party"},
                    },
                    {
                        "extraction_class": "claim",
                        "extraction_text": "Security incident notification deadline is 72 hours.",
                        "attributes": {
                            "obligation": "notify customers",
                            "deadline": "72 hours",
                        },
                    },
                    {
                        "extraction_class": "warning",
                        "extraction_text": (
                            "Late breach notification may violate contractual "
                            "or regulatory requirements."
                        ),
                        "attributes": {"severity": "high", "domain": "security"},
                    },
                ],
            }
        ],
    },
    "contract_terms_v1": {
        "name": "contract_terms_v1",
        "prompt_description": (
            "Extract contract parties, key terms, obligations, payment details, and termination conditions. "
            "Use extraction_class values: entity, relation, claim, warning."
        ),
        "examples": [
            {
                "text": "Customer may terminate with 30 days notice. Late payment incurs 1.5% monthly interest.",
                "extractions": [
                    {
                        "extraction_class": "claim",
                        "extraction_text": "Termination requires 30 days notice.",
                        "attributes": {
                            "term_type": "termination",
                            "notice_period": "30 days",
                        },
                    },
                    {
                        "extraction_class": "claim",
                        "extraction_text": "Late payment incurs 1.5% monthly interest.",
                        "attributes": {
                            "term_type": "payment",
                            "interest_rate": "1.5% monthly",
                        },
                    },
                    {
                        "extraction_class": "warning",
                        "extraction_text": "Late payment penalties create financial risk.",
                        "attributes": {"severity": "medium", "domain": "financial"},
                    },
                ],
            }
        ],
    },
}

DEFAULT_LANGEXTRACT_PROFILE = "generic_entities_v1"


def resolve_profile(name: str | None) -> ExtractionProfile:
    if not name:
        return LANGEXTRACT_PROFILES[DEFAULT_LANGEXTRACT_PROFILE]
    return LANGEXTRACT_PROFILES.get(
        name, LANGEXTRACT_PROFILES[DEFAULT_LANGEXTRACT_PROFILE]
    )


def list_profiles() -> list[str]:
    return sorted(LANGEXTRACT_PROFILES.keys())
