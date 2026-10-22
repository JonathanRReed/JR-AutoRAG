import pytest
from app.core.adaptive_gate import AdaptiveGate, GateDecision


@pytest.mark.parametrize("query", [
    "Hello, what does the contract say about termination?",
    "Help me understand the retention policy",
    "Calculate the total for 2026 from the uploaded invoices",
    "Today which policy is in effect?",
    "What is 2 + 2 according to the uploaded worksheet?",
])
def test_factual_requests_never_skip_retrieval(query):
    assert AdaptiveGate().gate_heuristic(query).decision != GateDecision.NO_RETRIEVAL


@pytest.mark.parametrize("query", ["hello!", "Thanks.", "what can you do?", "what is 2 + 2?"])
def test_pure_smalltalk_or_arithmetic_can_skip(query):
    assert AdaptiveGate().gate_heuristic(query).decision == GateDecision.NO_RETRIEVAL


@pytest.mark.asyncio
async def test_model_self_confidence_cannot_bypass_grounding():
    class Provider:
        async def chat(self, messages):
            return "DECISION: NO_RETRIEVAL\nCONFIDENCE: 1.0\nREASONING: I know it"
    result = await AdaptiveGate().gate_llm("What is the company retention policy?", Provider())
    assert result.decision == GateDecision.SINGLE_RETRIEVAL
