import asyncio

import pytest

from app.core.chunking import Chunk
from app.core.contextual_enrichment import ContextualEnricher, EnrichmentConfig


class RecordingProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return f"summary {self.calls}"


@pytest.mark.asyncio
async def test_enrich_chunks_limits_concurrent_provider_calls() -> None:
    provider = RecordingProvider()
    chunks = [
        Chunk(
            text=f"Chunk {i} has unique text for summarization.",
            index=i,
            start_char=i * 100,
            end_char=i * 100 + 40,
        )
        for i in range(10)
    ]
    enricher = ContextualEnricher(EnrichmentConfig(max_concurrent_enrichments=3))

    enriched = await enricher.enrich_chunks(
        chunks,
        document_text="\n".join(chunk.text for chunk in chunks),
        provider=provider,
    )

    assert [chunk.index for chunk in enriched] == list(range(10))
    assert provider.calls == len(chunks)
    assert provider.max_in_flight <= 3


@pytest.mark.asyncio
async def test_enrich_chunks_coerces_invalid_concurrency_to_one() -> None:
    provider = RecordingProvider()
    chunks = [
        Chunk(
            text=f"Chunk {i} has unique text for summarization.",
            index=i,
            start_char=i * 100,
            end_char=i * 100 + 40,
        )
        for i in range(4)
    ]
    enricher = ContextualEnricher(EnrichmentConfig(max_concurrent_enrichments=0))

    await enricher.enrich_chunks(
        chunks,
        document_text="\n".join(chunk.text for chunk in chunks),
        provider=provider,
    )

    assert provider.max_in_flight == 1
