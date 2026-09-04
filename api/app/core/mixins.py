from dataclasses import asdict
from typing import Any


class ToDictMixin:
    """Mixin to provide dictionary conversion for dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary for serialization."""
        return asdict(self)
