from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseStrategy(ABC):
    """Abstract base for all field-comparison strategies."""

    @abstractmethod
    def score_pair(self, input_value: Any, candidate_value: Any) -> float:
        """Compare a single input value against a single candidate value.

        Returns a float in [0, 1].
        """
        ...

    def score_many(self, input_value: Any, candidate_values: list[Any]) -> np.ndarray:
        """Compare a single input value against N candidate values.

        Returns an ndarray of shape (N,) with floats in [0, 1].

        Default implementation: loop over score_pair.
        Subclasses SHOULD override for vectorized performance.
        """
        n = len(candidate_values)
        scores = np.zeros(n, dtype=np.float64)
        for i, cv in enumerate(candidate_values):
            try:
                scores[i] = float(self.score_pair(input_value, cv))
            except Exception:
                scores[i] = 0.0
        return scores
