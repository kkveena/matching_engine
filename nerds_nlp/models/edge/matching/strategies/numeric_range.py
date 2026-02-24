from typing import Any

from .base import BaseStrategy


class NumericRangeMatchStrategy(BaseStrategy):
    """Scores based on triangular decay within a tolerance band.

    score = max(0, 1 - |input - candidate| / tolerance)

    If input == candidate, score = 1.0.
    If |input - candidate| >= tolerance, score = 0.0.
    Linear decay in between.
    """

    def __init__(self, tolerance: float = 0.01):
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.tolerance = tolerance

    def score_pair(self, input_value: Any, candidate_value: Any) -> float:
        try:
            iv = float(input_value)
            cv = float(candidate_value)
        except (TypeError, ValueError):
            return 0.0
        diff = abs(iv - cv)
        if diff >= self.tolerance:
            return 0.0
        return 1.0 - diff / self.tolerance
