from typing import Any

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
