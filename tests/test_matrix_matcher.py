import numpy as np
import pytest

from nerds_nlp.models.edge.matching.matrix_matcher import MatrixMatcher


class TestMatrixMatcher:
    def test_perfect_match(
        self, default_field_configs, sample_input_link, sample_candidate_links
    ):
        matcher = MatrixMatcher(default_field_configs)
        result = matcher.score_one_to_many(sample_input_link, sample_candidate_links)
        assert result.best_index == 0
        assert result.best_score == pytest.approx(1.0)
        assert len(result.used_fields) == 4

    def test_scores_shape(
        self, default_field_configs, sample_input_link, sample_candidate_links
    ):
        matcher = MatrixMatcher(default_field_configs)
        result = matcher.score_one_to_many(sample_input_link, sample_candidate_links)
        assert result.scores.shape == (3,)
        assert result.per_field_scores.shape == (3, 4)
        assert result.weighted_contributions.shape == (3, 4)

    def test_scores_in_range(
        self, default_field_configs, sample_input_link, sample_candidate_links
    ):
        matcher = MatrixMatcher(default_field_configs)
        result = matcher.score_one_to_many(sample_input_link, sample_candidate_links)
        assert np.all(result.scores >= 0.0)
        assert np.all(result.scores <= 1.0)
        assert np.all(result.per_field_scores >= 0.0)
        assert np.all(result.per_field_scores <= 1.0)

    def test_input_field_missing_excludes_from_denominator(self, default_field_configs):
        """When input is missing a field, that field should be excluded entirely."""
        matcher = MatrixMatcher(default_field_configs)
        input_link = {
            "valuation_date": "2025-01-15",
            # forward_rate intentionally absent
            "settlement_date": "2025-03-15",
            "direction": "incoming",
        }
        candidates = [
            {
                "valuation_date": "2025-01-15",
                "forward_rate": "1.2345",
                "settlement_date": "2025-03-15",
                "direction": "outgoing",
            }
        ]
        result = matcher.score_one_to_many(input_link, candidates)
        assert "forward_rate" not in result.used_fields
        assert len(result.used_fields) == 3
        assert result.best_score == pytest.approx(1.0)

    def test_candidate_field_missing_scores_zero(self, default_field_configs):
        """When candidate is missing a field, score=0 for that field (denominator unchanged)."""
        matcher = MatrixMatcher(default_field_configs)
        input_link = {
            "valuation_date": "2025-01-15",
            "forward_rate": "1.2345",
            "settlement_date": "2025-03-15",
            "direction": "incoming",
        }
        candidates = [
            {
                "valuation_date": "2025-01-15",
                # forward_rate missing on candidate
                "settlement_date": "2025-03-15",
                "direction": "outgoing",
            }
        ]
        result = matcher.score_one_to_many(input_link, candidates)
        assert "forward_rate" in result.used_fields
        # Score = (0.3*1 + 0.25*0 + 0.25*1 + 0.2*1) / 1.0 = 0.75
        assert result.best_score == pytest.approx(0.75)

    def test_empty_candidates(self, default_field_configs, sample_input_link):
        matcher = MatrixMatcher(default_field_configs)
        result = matcher.score_one_to_many(sample_input_link, [])
        assert result.scores.shape == (0,)
        assert result.best_score == 0.0

    def test_all_input_fields_missing(self, default_field_configs):
        """When all input fields are missing, score should be 0."""
        matcher = MatrixMatcher(default_field_configs)
        input_link = {}  # No fields at all
        candidates = [
            {
                "valuation_date": "2025-01-15",
                "forward_rate": "1.2345",
                "settlement_date": "2025-03-15",
                "direction": "outgoing",
            }
        ]
        result = matcher.score_one_to_many(input_link, candidates)
        assert result.used_fields == []
        assert result.best_score == 0.0

    def test_weighted_contributions_sum_matches_score(
        self, default_field_configs, sample_input_link, sample_candidate_links
    ):
        """Verify: sum(contributions) / total_weight == overall_score for each candidate."""
        matcher = MatrixMatcher(default_field_configs)
        result = matcher.score_one_to_many(sample_input_link, sample_candidate_links)
        total_weight = sum(
            fc.weight
            for fc in default_field_configs
            if fc.name in result.used_fields
        )
        for i in range(len(sample_candidate_links)):
            expected = result.weighted_contributions[i].sum() / total_weight
            assert result.scores[i] == pytest.approx(expected, abs=1e-10)

    def test_partial_match_score_ordering(
        self, default_field_configs, sample_input_link, sample_candidate_links
    ):
        """Scores should be ordered: perfect > partial > no match."""
        matcher = MatrixMatcher(default_field_configs)
        result = matcher.score_one_to_many(sample_input_link, sample_candidate_links)
        assert result.scores[0] > result.scores[1] > result.scores[2]

    def test_single_candidate(self, default_field_configs, sample_input_link):
        matcher = MatrixMatcher(default_field_configs)
        candidates = [
            {
                "valuation_date": "2025-01-15",
                "forward_rate": "1.2345",
                "settlement_date": "2025-03-15",
                "direction": "outgoing",
            }
        ]
        result = matcher.score_one_to_many(sample_input_link, candidates)
        assert result.scores.shape == (1,)
        assert result.best_index == 0
        assert result.best_score == pytest.approx(1.0)

    def test_multiple_identical_candidates(self, default_field_configs, sample_input_link):
        """Multiple identical candidates should all have the same score."""
        matcher = MatrixMatcher(default_field_configs)
        perfect = {
            "valuation_date": "2025-01-15",
            "forward_rate": "1.2345",
            "settlement_date": "2025-03-15",
            "direction": "outgoing",
        }
        candidates = [perfect, perfect, perfect]
        result = matcher.score_one_to_many(sample_input_link, candidates)
        assert np.all(result.scores == result.scores[0])
        assert result.best_score == pytest.approx(1.0)
