from typing import Any

import numpy as np

from .base import BaseStrategy


class FuzzyMatchStrategy(BaseStrategy):
    """Scores based on fuzzy string similarity using SequenceMatcher.

    Uses Python's built-in difflib.SequenceMatcher (no external dependencies).
    Returns a float in [0, 1] representing the similarity ratio.

    Example:
        "JPMorgan Chase" vs "JPMorgan" -> ~0.70
        "ABC Corp" vs "ABC Corporation" -> ~0.73
    """

    def __init__(self, threshold: float = 0.0):
        """
        Args:
            threshold: Minimum similarity to return a non-zero score.
                       Below this, score is forced to 0.0.
                       Default 0.0 means all similarities are returned as-is.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def score_pair(self, input_value: Any, candidate_value: Any) -> float:
        from difflib import SequenceMatcher

        a = self._normalize(input_value)
        b = self._normalize(candidate_value)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        ratio = SequenceMatcher(None, a, b).ratio()
        return ratio if ratio >= self.threshold else 0.0

    def score_many(self, input_value: Any, candidate_values: list[Any]) -> np.ndarray:
        from difflib import SequenceMatcher

        a = self._normalize(input_value)
        n = len(candidate_values)
        scores = np.zeros(n, dtype=np.float64)

        if not a:
            for i, cv in enumerate(candidate_values):
                if not self._normalize(cv):
                    scores[i] = 1.0
            return scores

        for i, cv in enumerate(candidate_values):
            b = self._normalize(cv)
            if not b:
                scores[i] = 0.0
                continue
            ratio = SequenceMatcher(None, a, b).ratio()
            scores[i] = ratio if ratio >= self.threshold else 0.0

        return scores
