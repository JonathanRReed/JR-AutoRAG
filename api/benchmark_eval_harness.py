import asyncio
import time
from pathlib import Path
from app.core.eval_harness import EvalHarness, EvalCase

async def dummy_query_fn(query: str):
    await asyncio.sleep(0.1)
    return {
        "answer": f"Answer to {query}",
        "chunks": [{"title": "Source 1"}],
        "metrics": {"tokens": 10},
        "grounding": {"grounded": True}
    }

async def main():
    harness = EvalHarness(data_path=Path("/tmp/eval_data"), query_fn=dummy_query_fn)
    cases = [
        EvalCase(id=f"case_{i}", query=f"Query {i}", expected_answer="Expected")
        for i in range(20)
    ]

    start_time = time.time()
    await harness.run(cases)
    end_time = time.time()

    print(f"Elapsed time for 20 cases: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
