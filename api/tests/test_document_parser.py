from __future__ import annotations

import json

from app.core.document_parser import DocumentParserRouter, parser_result_to_metadata


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
