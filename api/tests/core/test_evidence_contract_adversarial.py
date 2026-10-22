from types import SimpleNamespace

from app.core.evidence_contract import EvidenceContract


def evidence(text):
    return [SimpleNamespace(id="source-1", snippet=text)]


def test_citation_marker_is_not_evidence():
    result = EvidenceContract().verify_answer("The moon is made of cheese [1].", [])
    assert not result.pass_threshold
    assert result.unsupported_claims
    assert result.verified_claims[0].confidence == 0


def test_changed_numbers_are_not_supported_by_word_overlap():
    result = EvidenceContract().verify_answer(
        "The annual contract price is $900.", evidence("The annual contract price is $100."))
    assert result.unsupported_claims


def test_opposite_negation_is_not_support():
    result = EvidenceContract().verify_answer(
        "The treatment is effective.", evidence("The treatment is not effective."))
    assert result.unsupported_claims


def test_genuine_cited_claim_retains_evidence_and_nonperfect_score():
    result = EvidenceContract().verify_answer(
        "The annual contract price is $100 [1].", evidence("The annual contract price is $100."))
    assert result.pass_threshold
    assert result.verified_claims[0].supporting_span == "The annual contract price is $100."


def test_claim_extraction_is_stable_and_keeps_citation_prefixed_claims():
    contract = EvidenceContract()
    claims = contract.extract_claims("[1] The annual price is $100. The product is available. The annual price is $100.")
    assert claims[0].startswith("[1]")
    assert "The product is available." in claims
