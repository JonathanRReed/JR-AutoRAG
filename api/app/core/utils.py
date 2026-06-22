"""Common utility functions for the core module."""

from typing import Any

def count_matches(patterns: list[Any], text: str) -> int:
    """Count pattern matches in text."""
    return sum(1 for p in patterns if p.search(text))
