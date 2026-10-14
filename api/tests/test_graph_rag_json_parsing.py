import json
from unittest import mock

import pytest

from app.core import graph_rag
from app.core.graph_rag import GraphRAG


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def chat(self, messages):
        self.calls += 1
        item = self.responses[min(self.calls, len(self.responses)) - 1]
        if isinstance(item, Exception):
            raise item
        return item


def payload(entities=(), relationships=()):
    return json.dumps({
        "entities": [{"name": n, "type": "organization", "description": d} for n, d in entities],
        "relationships": [
            {"source": s, "target": t, "relation": r, "description": d} for s, t, r, d in relationships
        ],
    })


async def extract(*responses):
    provider = FakeProvider(responses)
    with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
        entities, rels = await GraphRAG().extract_knowledge_from_chunk("text", "c1", provider)
    return (
        [(e.name, e.description) for e in entities],
        [(r.source, r.target, r.relation, r.description) for r in rels],
        provider.calls,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["ACME,}", "ACME, ]", "ACME ,\n]", "Say \"hi\", }", "C:\\dir\\, ]"])
async def test_valid_json_entity_name_is_preserved(name):
    assert await extract(payload([(name, "x")])) == ([(name, "x")], [], 1)


@pytest.mark.asyncio
async def test_valid_json_description_punctuation_is_preserved():
    desc = "list: a, b,} and c, ] end"
    assert await extract(payload([("ACME", desc)])) == ([("ACME", desc)], [], 1)


@pytest.mark.asyncio
async def test_valid_json_relationship_fields_are_preserved():
    rel = ("A,]", "B, }", "owns,]", "since 2020, ]")
    assert await extract(payload([("A", "")], [rel])) == ([("A", "")], [rel], 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("fence", ["```json", "```"], ids=["json-tag", "no-tag"])
async def test_fenced_valid_json_is_preserved(fence):
    fenced = "Here:\n" + fence + "\n" + payload([("ACME,}", "d, ]")]) + "\n```"
    assert await extract(fenced) == ([("ACME,}", "d, ]")], [], 1)


@pytest.mark.asyncio
async def test_apostrophes_in_valid_json_are_preserved():
    assert await extract(payload([("O'Brien", "Pat O'Brien's firm")])) == (
        [("O'Brien", "Pat O'Brien's firm")], [], 1)


@pytest.mark.asyncio
async def test_structural_trailing_commas_are_repaired_without_touching_strings():
    raw = ('{"entities": [{"name": "ACME,}", "type": "organization", "description": "d, ]",},],'
           ' "relationships": [],}')
    assert await extract(raw) == ([("ACME,}", "d, ]")], [], 1)


@pytest.mark.asyncio
async def test_structural_repair_keeps_escapes_and_unicode_escapes():
    raw = ('{"entities": [{"name": "Q\\"T,}", "type": "organization",'
           ' "description": "C:\\\\x\\\\, ] \\u002c}",\n},\n],\t"relationships": [\n]\n,}')
    assert await extract(raw) == ([('Q"T,}', "C:\\x\\, ] ,}")], [], 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    "{'entities': [{'name': 'ACME', 'type': 'organization', 'description': 'x'}], 'relationships': []}",
    "{'entities': [{'name': 'ACME', 'type': 'organization', 'description': 'x',},], 'relationships': [],}",
    "{\"entities\": [{\"name\": 'ACME', \"type\": \"organization\", \"description\": \"x\"}], \"relationships\": []}",
], ids=["single-quoted", "single-quoted-trailing-commas", "mixed-quotes"])
async def test_single_quoted_pseudo_json_is_rejected_after_three_calls(raw):
    assert await extract(raw) == ([], [], 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["b'x'", "(1, 2)", "{1, 2}"], ids=["bytes", "tuple", "set"])
async def test_python_only_literals_in_description_are_rejected(value):
    raw = ("{'entities': [{'name': 'ACME', 'type': 'organization', 'description': " + value + "}],"
           " 'relationships': []}")
    assert await extract(raw) == ([], [], 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    "{bad",
    '{"entities" [], "relationships": []}',
    '{"entities": [{"name": "ACME}], "relationships": []}',
    '["ACME", "organization"]',
], ids=["bare-brace", "missing-colon", "unterminated-string", "top-level-array"])
async def test_malformed_or_non_object_output_is_rejected_after_three_calls(raw):
    assert await extract(raw) == ([], [], 3)


@pytest.mark.asyncio
async def test_invalid_twice_then_valid_uses_three_calls():
    assert await extract("{bad", "{bad", payload([("ACME", "ok")])) == ([("ACME", "ok")], [], 3)


@pytest.mark.asyncio
async def test_provider_errors_return_empty_after_three_calls():
    assert await extract(RuntimeError("Provider request failed")) == ([], [], 3)


def test_scanner_preserves_escape_heavy_strings_byte_for_byte():
    text = '["a\\",]", "\\\\", "\\\\\\",}", "\\u002c]", "x\\n,}",\n]'
    assert graph_rag._strip_trailing_commas(text) == '["a\\",]", "\\\\", "\\\\\\",}", "\\u002c]", "x\\n,}"\n]'
    assert json.loads(graph_rag._strip_trailing_commas(text)) == ['a",]', "\\", '\\",}', ",]", "x\n,}"]


def test_scanner_only_drops_commas_before_closers_on_dense_input():
    text = '{"a":[' + "1, " * 5000 + "],}"
    assert graph_rag._strip_trailing_commas(text) == '{"a":[' + "1, " * 4999 + "1 ]}"
