from typing import Any

import numpy as np

from .base import BaseStrategy

_INVERSE_MAP = {
    "incoming": "outgoing",
    "outgoing": "incoming",
    "buy": "sell",
    "sell": "buy",
}


class DirectionInverseMatchStrategy(BaseStrategy):
    """Scores 1.0 if the candidate's direction is the inverse of the input's direction."""

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def score_pair(self, input_value: Any, candidate_value: Any) -> float:
        norm_input = self._normalize(input_value)
        expected = _INVERSE_MAP.get(norm_input, "")
        if not expected:
            return 0.0
        return 1.0 if self._normalize(candidate_value) == expected else 0.0

    def score_many(self, input_value: Any, candidate_values: list[Any]) -> np.ndarray:
        norm_input = self._normalize(input_value)
        expected = _INVERSE_MAP.get(norm_input, "")
        if not expected:
            return np.zeros(len(candidate_values), dtype=np.float64)
        norm_candidates = np.array([self._normalize(cv) for cv in candidate_values])
        return (norm_candidates == expected).astype(np.float64)
