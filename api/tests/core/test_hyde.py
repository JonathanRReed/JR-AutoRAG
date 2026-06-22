"""Tests for HyDE generator."""

from __future__ import annotations

import pytest

import app.core.hyde as hyde_module
from app.core.hyde import HyDEConfig, get_hyde_generator


@pytest.fixture(autouse=True)
def reset_hyde_generator() -> None:
    """Reset the singleton before and after each test."""
    original = hyde_module._hyde_generator
    hyde_module._hyde_generator = None
    yield
    hyde_module._hyde_generator = original


def test_get_hyde_generator_singleton() -> None:
    """Test that get_hyde_generator returns the same instance on multiple calls."""
    generator1 = get_hyde_generator()
    generator2 = get_hyde_generator()

    assert generator1 is not None
    assert generator1 is generator2


def test_get_hyde_generator_with_config() -> None:
    """Test that passing a config creates a new instance and updates the singleton."""
    generator1 = get_hyde_generator()

    config = HyDEConfig(num_hypotheticals=5)
    generator2 = get_hyde_generator(config)

    assert generator2 is not generator1
    assert generator2.config.num_hypotheticals == 5

    # Verify it updated the singleton
    generator3 = get_hyde_generator()
    assert generator3 is generator2
