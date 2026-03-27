import numpy as np
import pytest

from nerds_nlp.models.edge.matching.strategies import FuzzyMatchStrategy


class TestFuzzyMatchStrategy:
    """Tests for FuzzyMatchStrategy using difflib.SequenceMatcher."""

    # --- score_pair ---

    def test_identical_strings(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair("JPMorgan Chase", "JPMorgan Chase") == 1.0

    def test_case_insensitive(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair("JPMORGAN", "jpmorgan") == 1.0

    def test_whitespace_stripped(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair("  hello  ", "hello") == 1.0

    def test_partial_match_substring(self):
        s = FuzzyMatchStrategy()
        score = s.score_pair("JPMorgan Chase", "JPMorgan")
        assert 0.5 < score < 1.0

    def test_partial_match_abbreviation(self):
        s = FuzzyMatchStrategy()
        score = s.score_pair("ABC Corporation", "ABC Corp")
        assert 0.5 < score < 1.0

    def test_completely_different(self):
        s = FuzzyMatchStrategy()
        score = s.score_pair("Apple Inc", "Toyota Motor")
        assert score < 0.5

    def test_empty_vs_nonempty(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair("", "hello") == 0.0

    def test_nonempty_vs_empty(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair("hello", "") == 0.0

    def test_both_empty(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair("", "") == 1.0

    def test_none_input(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair(None, "hello") == 0.0

    def test_none_candidate(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair("hello", None) == 0.0

    def test_both_none(self):
        s = FuzzyMatchStrategy()
        assert s.score_pair(None, None) == 1.0

    # --- threshold ---

    def test_threshold_filters_low_scores(self):
        s = FuzzyMatchStrategy(threshold=0.8)
        # "Apple" vs "Zebra" should be very low
        assert s.score_pair("Apple", "Zebra") == 0.0

    def test_threshold_passes_high_scores(self):
        s = FuzzyMatchStrategy(threshold=0.5)
        score = s.score_pair("JPMorgan Chase", "JPMorgan")
        assert score > 0.5

    def test_threshold_exact_match_passes(self):
        s = FuzzyMatchStrategy(threshold=0.9)
        assert s.score_pair("hello", "hello") == 1.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            FuzzyMatchStrategy(threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            FuzzyMatchStrategy(threshold=-0.1)

    # --- score_many ---

    def test_score_many_basic(self):
        s = FuzzyMatchStrategy()
        candidates = ["JPMorgan Chase", "JPMorgan", "Goldman Sachs", "JP Morgan"]
        result = s.score_many("JPMorgan Chase", candidates)
        assert result.shape == (4,)
        assert result[0] == 1.0  # identical
        assert result[1] > 0.5  # substring
        assert result[2] < result[0]  # different company
        assert result[3] > 0.5  # close variant

    def test_score_many_empty_list(self):
        s = FuzzyMatchStrategy()
        result = s.score_many("hello", [])
        assert result.shape == (0,)

    def test_score_many_with_none_candidates(self):
        s = FuzzyMatchStrategy()
        result = s.score_many("hello", ["hello", None, "", "world"])
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] < 1.0

    def test_score_many_none_input(self):
        s = FuzzyMatchStrategy()
        result = s.score_many(None, ["hello", None, ""])
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    # --- scalar vs vector parity ---

    def test_scalar_vs_vector_parity(self):
        s = FuzzyMatchStrategy()
        candidates = [
            "JPMorgan Chase",
            "JPMorgan",
            "Goldman Sachs",
            None,
            "",
            "  jpmorgan chase  ",
        ]
        input_val = "JPMorgan Chase"
        scalar_scores = [s.score_pair(input_val, c) for c in candidates]
        vector_scores = s.score_many(input_val, candidates)
        np.testing.assert_array_almost_equal(vector_scores, scalar_scores)

    def test_scalar_vs_vector_parity_with_threshold(self):
        s = FuzzyMatchStrategy(threshold=0.6)
        candidates = ["JPMorgan Chase", "JPMorgan", "Goldman Sachs", "Apple"]
        input_val = "JPMorgan Chase"
        scalar_scores = [s.score_pair(input_val, c) for c in candidates]
        vector_scores = s.score_many(input_val, candidates)
        np.testing.assert_array_almost_equal(vector_scores, scalar_scores)

    # --- registry integration ---

    def test_registry_lookup(self):
        from nerds_nlp.models.edge.matching.strategies import get_strategy

        strategy = get_strategy("fuzzy_match", {"threshold": 0.5})
        assert isinstance(strategy, FuzzyMatchStrategy)
        assert strategy.threshold == 0.5

    def test_registry_lookup_default_params(self):
        from nerds_nlp.models.edge.matching.strategies import get_strategy

        strategy = get_strategy("fuzzy_match")
        assert isinstance(strategy, FuzzyMatchStrategy)
        assert strategy.threshold == 0.0
