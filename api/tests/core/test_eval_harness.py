from __future__ import annotations

import asyncio

import pytest

from app.core.eval_harness import EvalCase, EvalHarness, get_eval_harness


@pytest.mark.asyncio
async def test_eval_harness_limits_concurrent_query_calls(tmp_path):
    active = 0
    peak_active = 0
    lock = asyncio.Lock()

    async def query_fn(query: str) -> dict:
        nonlocal active, peak_active
        async with lock:
            active += 1
            peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return {"answer": query, "chunks": [], "metrics": {}, "grounding": {}}

    harness = EvalHarness(data_path=tmp_path, query_fn=query_fn, max_concurrent=2)
    cases = [EvalCase(id=str(i), query=f"query {i}") for i in range(10)]

    run = await harness.run(cases, run_id="bounded")

    assert len(run.results) == len(cases)
    assert peak_active <= 2


def test_eval_harness_rejects_invalid_concurrency(tmp_path):
    with pytest.raises(ValueError, match="max_concurrent"):
        EvalHarness(data_path=tmp_path, max_concurrent=0)


def test_get_eval_harness_passes_concurrency(tmp_path):
    harness = get_eval_harness(data_path=tmp_path, max_concurrent=3)

    assert harness._max_concurrent == 3
