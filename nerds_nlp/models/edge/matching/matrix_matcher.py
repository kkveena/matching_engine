from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schemas import FieldConfig
from .strategies.base import BaseStrategy
from .strategies.registry import get_strategy


@dataclass
class MatrixMatchResult:
    """Raw numeric result from score_one_to_many."""

    scores: np.ndarray  # shape (N,) — final weighted scores per candidate
    per_field_scores: np.ndarray  # shape (N, F) — raw score per field per candidate
    weighted_contributions: np.ndarray  # shape (N, F) — weight * score per field
    best_index: int  # argmax of scores
    best_score: float  # scores[best_index]
    used_fields: list[str] = field(default_factory=list)  # fields present on the input


class MatrixMatcher:
    """Computes 1-to-N matching scores using vectorized strategy evaluation."""

    def __init__(self, field_configs: list[FieldConfig]):
        self.field_configs = field_configs
        self._strategies: list[tuple[FieldConfig, BaseStrategy]] = [
            (fc, get_strategy(fc.strategy, fc.params)) for fc in field_configs
        ]

    def score_one_to_many(
        self,
        input_link: dict[str, Any],
        candidate_links: list[dict[str, Any]],
    ) -> MatrixMatchResult:
        """Score one input link against N candidate links.

        Missing-field semantics:
          - If the input link is missing a field: exclude that field
            entirely (from both numerator and denominator).
          - If a candidate link is missing a field: that candidate gets
            score=0 for that field (field remains in denominator).
        """
        n_candidates = len(candidate_links)

        # Phase 1: Determine which fields are present on the input
        active: list[tuple[FieldConfig, BaseStrategy]] = []
        for fc, strategy in self._strategies:
            input_val = input_link.get(fc.name)
            if input_val is None:
                continue
            active.append((fc, strategy))

        n_fields = len(active)
        used_fields = [fc.name for fc, _ in active]

        if n_candidates == 0 or n_fields == 0:
            return MatrixMatchResult(
                scores=np.zeros(max(n_candidates, 0), dtype=np.float64),
                per_field_scores=np.zeros(
                    (max(n_candidates, 0), max(n_fields, 0)), dtype=np.float64
                ),
                weighted_contributions=np.zeros(
                    (max(n_candidates, 0), max(n_fields, 0)), dtype=np.float64
                ),
                best_index=0,
                best_score=0.0,
                used_fields=used_fields,
            )

        # Phase 2: Build score matrix (N candidates x F fields)
        per_field_scores = np.zeros((n_candidates, n_fields), dtype=np.float64)
        weights = np.zeros(n_fields, dtype=np.float64)

        for f_idx, (fc, strategy) in enumerate(active):
            input_val = input_link[fc.name]
            weights[f_idx] = fc.weight

            # Extract candidate values; None if field missing on candidate
            candidate_vals = []
            missing_mask = []
            for cl in candidate_links:
                cv = cl.get(fc.name)
                candidate_vals.append(cv)
                missing_mask.append(cv is None)

            # Vectorized scoring
            field_scores = strategy.score_many(input_val, candidate_vals)

            # Candidate missing -> force score to 0
            missing_arr = np.array(missing_mask, dtype=bool)
            field_scores[missing_arr] = 0.0

            # Clamp to [0, 1]
            np.clip(field_scores, 0.0, 1.0, out=field_scores)

            per_field_scores[:, f_idx] = field_scores

        # Phase 3: Weighted combination
        weighted_contributions = per_field_scores * weights[np.newaxis, :]

        total_weight = weights.sum()
        if total_weight == 0:
            scores = np.zeros(n_candidates, dtype=np.float64)
        else:
            scores = weighted_contributions.sum(axis=1) / total_weight

        # Clamp final scores
        np.clip(scores, 0.0, 1.0, out=scores)

        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        return MatrixMatchResult(
            scores=scores,
            per_field_scores=per_field_scores,
            weighted_contributions=weighted_contributions,
            best_index=best_index,
            best_score=best_score,
            used_fields=used_fields,
        )
