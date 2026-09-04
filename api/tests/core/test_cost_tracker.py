"""Unit tests for cost_tracker module."""

from __future__ import annotations

from app.core.cost_tracker import CostEstimate, CostTracker, TokenUsage


def test_token_usage_initialization_and_addition() -> None:
    # Test post_init calculation
    usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50)
    assert usage1.total_tokens == 150
    assert usage1.to_dict() == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }

    # Test explicit total_tokens preservation if supplied
    usage_explicit = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=200)
    assert usage_explicit.total_tokens == 200

    # Test addition
    usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100)
    combined = usage1.add(usage2)
    assert combined.prompt_tokens == 300
    assert combined.completion_tokens == 150
    assert combined.total_tokens == 450


def test_cost_estimate_initialization_and_addition() -> None:
    # Test post_init calculation
    est1 = CostEstimate(prompt_cost_usd=0.001, completion_cost_usd=0.002, provider="openai", model="gpt-4o")
    assert est1.total_cost_usd == 0.003
    assert est1.to_dict() == {
        "prompt_cost_usd": 0.001,
        "completion_cost_usd": 0.002,
        "total_cost_usd": 0.003,
        "provider": "openai",
        "model": "gpt-4o",
    }

    # Test addition with same provider and model
    est2 = CostEstimate(prompt_cost_usd=0.002, completion_cost_usd=0.004, provider="openai", model="gpt-4o")
    combined1 = est1.add(est2)
    assert combined1.prompt_cost_usd == 0.003
    assert combined1.completion_cost_usd == 0.006
    assert combined1.total_cost_usd == 0.009
    assert combined1.provider == "openai"
    assert combined1.model == "gpt-4o"

    # Test addition with different provider and model
    est_other = CostEstimate(prompt_cost_usd=0.005, completion_cost_usd=0.005, provider="anthropic", model="claude-3-haiku")
    combined2 = est1.add(est_other)
    assert combined2.provider == "mixed"
    assert combined2.model == "mixed"


def test_cost_tracker_calculation_and_history() -> None:
    tracker = CostTracker()
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)

    # Calculate cost for gpt-4o (pricing: prompt=0.0025/1k, completion=0.010/1k)
    # Expected prompt: (1000/1000)*0.0025 = 0.0025, completion: (500/1000)*0.010 = 0.005 => total = 0.0075
    estimate = tracker.calculate_cost(usage, model="gpt-4o", provider="openai")
    assert estimate.prompt_cost_usd == 0.0025
    assert estimate.completion_cost_usd == 0.005
    assert estimate.total_cost_usd == 0.0075

    # Check total cost and history
    assert tracker.get_total_cost() == 0.0075
    history = tracker.get_history()
    assert len(history) == 1
    assert history[0]["model"] == "gpt-4o"

    # Clear history
    tracker.clear_history()
    assert tracker.get_total_cost() == 0.0
    assert len(tracker.get_history()) == 0


def test_cost_tracker_custom_pricing() -> None:
    custom_pricing = {
        "custom-model": {"prompt": 0.01, "completion": 0.02}
    }
    tracker = CostTracker(pricing_table=custom_pricing)
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)

    estimate = tracker.calculate_cost(usage, model="custom-model", provider="custom_prov")
    assert estimate.prompt_cost_usd == 0.01
    assert estimate.completion_cost_usd == 0.02
    assert estimate.total_cost_usd == 0.03
    assert estimate.provider == "custom_prov"

    # Unknown model defaults to 0.0 rates
    unk_estimate = tracker.calculate_cost(usage, model="unknown-model")
    assert unk_estimate.total_cost_usd == 0.0
