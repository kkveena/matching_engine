from typing import Any

import numpy as np

from .base import BaseStrategy


class ExactMatchStrategy(BaseStrategy):
    """Scores 1.0 if normalized string values are equal, 0.0 otherwise."""

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def score_pair(self, input_value: Any, candidate_value: Any) -> float:
        return 1.0 if self._normalize(input_value) == self._normalize(candidate_value) else 0.0

    def score_many(self, input_value: Any, candidate_values: list[Any]) -> np.ndarray:
        norm_input = self._normalize(input_value)
        norm_candidates = np.array([self._normalize(cv) for cv in candidate_values])
        return (norm_candidates == norm_input).astype(np.float64)
