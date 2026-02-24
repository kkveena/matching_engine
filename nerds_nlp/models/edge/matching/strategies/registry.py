from typing import Any

from .base import BaseStrategy
from .direction_inverse import DirectionInverseMatchStrategy
from .exact_match import ExactMatchStrategy
from .numeric_range import NumericRangeMatchStrategy

_STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "exact_match": ExactMatchStrategy,
    "numeric_range": NumericRangeMatchStrategy,
    "direction_inverse": DirectionInverseMatchStrategy,
}


def get_strategy(name: str, params: dict[str, Any] | None = None) -> BaseStrategy:
    """Look up a strategy by its YAML config name and instantiate it."""
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(_STRATEGY_REGISTRY.keys())}"
        )
    return cls(**(params or {}))


def register_strategy(name: str, cls: type[BaseStrategy]) -> None:
    """Register a custom strategy at runtime."""
    _STRATEGY_REGISTRY[name] = cls
