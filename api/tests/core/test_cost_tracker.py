"""Unit tests for cost_tracker module."""

from __future__ import annotations

import logging

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
    assert est1.is_known_model is True
    assert est1.to_dict() == {
        "prompt_cost_usd": 0.001,
        "completion_cost_usd": 0.002,
        "total_cost_usd": 0.003,
        "provider": "openai",
        "model": "gpt-4o",
        "is_known_model": True,
    }

    # Test addition with same provider and model
    est2 = CostEstimate(prompt_cost_usd=0.002, completion_cost_usd=0.004, provider="openai", model="gpt-4o")
    combined1 = est1.add(est2)
    assert combined1.prompt_cost_usd == 0.003
    assert combined1.completion_cost_usd == 0.006
    assert combined1.total_cost_usd == 0.009
    assert combined1.provider == "openai"
    assert combined1.model == "gpt-4o"
    assert combined1.is_known_model is True

    # Test addition with different provider/model and unknown model flag combination
    est_unknown = CostEstimate(prompt_cost_usd=0.005, completion_cost_usd=0.005, provider="anthropic", model="custom-unk", is_known_model=False)
    combined2 = est1.add(est_unknown)
    assert combined2.provider == "mixed"
    assert combined2.model == "mixed"
    assert combined2.is_known_model is False


def test_cost_tracker_calculation_and_history() -> None:
    tracker = CostTracker()
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)

    # Calculate cost for gpt-4o (pricing: prompt=0.0025/1k, completion=0.010/1k)
    estimate = tracker.calculate_cost(usage, model="gpt-4o", provider="openai")
    assert estimate.prompt_cost_usd == 0.0025
    assert estimate.completion_cost_usd == 0.005
    assert estimate.total_cost_usd == 0.0075
    assert estimate.is_known_model is True

    # Check total cost and history
    assert tracker.get_total_cost() == 0.0075
    history = tracker.get_history()
    assert len(history) == 1
    assert history[0]["model"] == "gpt-4o"
    assert history[0]["cost"]["is_known_model"] is True

    # Clear history
    tracker.clear_history()
    assert tracker.get_total_cost() == 0.0
    assert len(tracker.get_history()) == 0


def test_cost_tracker_custom_pricing_and_unknown_models(caplog) -> None:
    custom_pricing = {
        "custom-model": {"prompt": 0.01, "completion": 0.02},
        "free-local-model": {"prompt": 0.0, "completion": 0.0},
    }
    tracker = CostTracker(pricing_table=custom_pricing)
    usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)

    # Test custom pricing model
    estimate = tracker.calculate_cost(usage, model="custom-model", provider="custom_prov")
    assert estimate.prompt_cost_usd == 0.01
    assert estimate.completion_cost_usd == 0.02
    assert estimate.total_cost_usd == 0.03
    assert estimate.is_known_model is True

    # Test explicit free model
    free_estimate = tracker.calculate_cost(usage, model="free-local-model", provider="local")
    assert free_estimate.total_cost_usd == 0.0
    assert free_estimate.is_known_model is True

    # Test unknown model warning and is_known_model=False
    with caplog.at_level(logging.WARNING):
        unk_estimate = tracker.calculate_cost(usage, model="unknown-model")
        assert unk_estimate.total_cost_usd == 0.0
        assert unk_estimate.is_known_model is False
        assert "Unknown model 'unknown-model' encountered" in caplog.text
