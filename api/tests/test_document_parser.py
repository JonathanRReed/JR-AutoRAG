from __future__ import annotations

import io
import json
import random
import string
import zipfile

import pytest

from app.core.archive_safety import UnsafeArchiveError, validate_docx_archive
from app.core.document_parser import (
    DoclingDocumentParser,
    DocumentParserRouter,
    parser_result_to_metadata,
)
from app.core.ingest import IngestPipeline


def test_native_parser_builds_structured_preview_for_markdown() -> None:
    router = DocumentParserRouter(prefer_docling=False)
    result = router.parse(
        b"# Overview\n\nA short paragraph.\n\n| A | B |\n| - | - |\n| 1 | 2 |",
        {"filename": "notes.md", "content_type": "text/markdown"},
    )

    assert result.provider == "native"
    assert result.blocks[0].type == "heading"
    assert any(block.type == "table" for block in result.blocks)

    metadata = parser_result_to_metadata(result)
    preview = json.loads(metadata["parser_preview_json"])
    assert preview["page_count"] == 1
    assert preview["block_count"] >= 3


def _compressed_docx(payload_size: int = 2 * 1024 * 1024) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", b"A" * payload_size)
    return output.getvalue()


def _docx_with_xml(document_xml: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", document_xml)
    return output.getvalue()


def test_ingest_rejects_docx_with_dangerous_expansion_ratio() -> None:
    pipeline = IngestPipeline(store=object(), retrieval=object())  # type: ignore[arg-type]

    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        pipeline._extract_text_with_metadata(
            _compressed_docx(),
            {"filename": "dangerous.docx"},
        )


def test_docling_rejects_dangerous_docx_before_converter_import() -> None:
    parser = DoclingDocumentParser()

    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        parser.parse(_compressed_docx(), {"filename": "dangerous.docx"})


def test_docx_validator_rejects_entity_declarations() -> None:
    content = _docx_with_xml(
        b'<!DOCTYPE document [<!ENTITY sensitive SYSTEM "file:///etc/passwd">]><document>&sensitive;</document>'
    )

    with pytest.raises(UnsafeArchiveError, match="forbidden XML declaration"):
        validate_docx_archive(content)


def test_docx_validator_scans_past_the_initial_xml_chunk() -> None:
    rng = random.Random(0)
    padding = "".join(
        rng.choice(string.ascii_letters) for _ in range(70 * 1024)
    ).encode()
    content = _docx_with_xml(
        b"<!--"
        + padding
        + b"--><!DOCTYPE document [<!ENTITY x 'expanded'>]><document>&x;</document>"
    )

    with pytest.raises(UnsafeArchiveError, match="forbidden XML declaration"):
        validate_docx_archive(content)


def test_docx_validator_accepts_bounded_archive() -> None:
    validate_docx_archive(
        _docx_with_xml(b"<document><paragraph>Safe text</paragraph></document>")
    )
