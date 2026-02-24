import numpy as np
import pytest

from nerds_nlp.models.edge.matching.strategies import (
    DirectionInverseMatchStrategy,
    ExactMatchStrategy,
    NumericRangeMatchStrategy,
)


class TestExactMatchStrategy:
    def test_exact_match_identical(self):
        s = ExactMatchStrategy()
        assert s.score_pair("abc", "abc") == 1.0

    def test_exact_match_case_insensitive(self):
        s = ExactMatchStrategy()
        assert s.score_pair("ABC", "abc") == 1.0

    def test_exact_match_whitespace_stripped(self):
        s = ExactMatchStrategy()
        assert s.score_pair("  hello  ", "hello") == 1.0

    def test_exact_match_different(self):
        s = ExactMatchStrategy()
        assert s.score_pair("abc", "xyz") == 0.0

    def test_exact_match_none_input(self):
        s = ExactMatchStrategy()
        assert s.score_pair(None, "abc") == 0.0

    def test_exact_match_none_both(self):
        s = ExactMatchStrategy()
        assert s.score_pair(None, None) == 1.0

    def test_score_many_vectorized(self):
        s = ExactMatchStrategy()
        result = s.score_many("hello", ["hello", "HELLO", "world", "hello "])
        np.testing.assert_array_equal(result, [1.0, 1.0, 0.0, 1.0])

    def test_score_many_empty(self):
        s = ExactMatchStrategy()
        result = s.score_many("hello", [])
        assert result.shape == (0,)

    def test_scalar_vs_vector_parity(self):
        s = ExactMatchStrategy()
        candidates = ["2025-01-15", "2025-01-16", "2025-01-15", None, "  2025-01-15 "]
        input_val = "2025-01-15"
        scalar_scores = [s.score_pair(input_val, c) for c in candidates]
        vector_scores = s.score_many(input_val, candidates)
        np.testing.assert_array_almost_equal(vector_scores, scalar_scores)


class TestNumericRangeMatchStrategy:
    def test_exact_numeric_match(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        assert s.score_pair(1.0, 1.0) == 1.0

    def test_within_tolerance(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        score = s.score_pair(1.0, 1.005)
        assert 0.0 < score < 1.0
        assert abs(score - 0.5) < 1e-9

    def test_at_tolerance_boundary(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        assert s.score_pair(1.0, 1.01) == 0.0

    def test_beyond_tolerance(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        assert s.score_pair(1.0, 2.0) == 0.0

    def test_string_numeric_coercion(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        assert s.score_pair("1.2345", "1.2345") == 1.0

    def test_non_numeric_input_returns_zero(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        assert s.score_pair("abc", "1.0") == 0.0

    def test_non_numeric_candidate_returns_zero(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        assert s.score_pair("1.0", "abc") == 0.0

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError):
            NumericRangeMatchStrategy(tolerance=-0.01)

    def test_zero_tolerance_raises(self):
        with pytest.raises(ValueError):
            NumericRangeMatchStrategy(tolerance=0.0)

    def test_score_many_vectorized(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        result = s.score_many(1.0, [1.0, 1.005, 1.01, 2.0])
        expected = [1.0, 0.5, 0.0, 0.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_score_many_with_non_numeric(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        result = s.score_many("1.0", ["1.0", "abc", None, "1.005"])
        expected = [1.0, 0.0, 0.0, 0.5]
        np.testing.assert_array_almost_equal(result, expected)

    def test_score_many_non_numeric_input(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        result = s.score_many("abc", ["1.0", "2.0"])
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_scalar_vs_vector_parity(self):
        s = NumericRangeMatchStrategy(tolerance=0.01)
        candidates = ["1.2345", "1.2340", "2.0", "abc", None, "1.2350"]
        input_val = "1.2345"
        scalar_scores = [s.score_pair(input_val, c) for c in candidates]
        vector_scores = s.score_many(input_val, candidates)
        np.testing.assert_array_almost_equal(vector_scores, scalar_scores)


class TestDirectionInverseMatchStrategy:
    def test_incoming_outgoing(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair("incoming", "outgoing") == 1.0

    def test_outgoing_incoming(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair("outgoing", "incoming") == 1.0

    def test_buy_sell(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair("buy", "sell") == 1.0

    def test_sell_buy(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair("sell", "buy") == 1.0

    def test_same_direction(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair("incoming", "incoming") == 0.0

    def test_unknown_direction(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair("unknown", "outgoing") == 0.0

    def test_none_input(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair(None, "outgoing") == 0.0

    def test_case_insensitive(self):
        s = DirectionInverseMatchStrategy()
        assert s.score_pair("INCOMING", "OUTGOING") == 1.0

    def test_score_many_vectorized(self):
        s = DirectionInverseMatchStrategy()
        result = s.score_many("incoming", ["outgoing", "incoming", "outgoing", "unknown"])
        np.testing.assert_array_equal(result, [1.0, 0.0, 1.0, 0.0])

    def test_score_many_unknown_input(self):
        s = DirectionInverseMatchStrategy()
        result = s.score_many("unknown", ["outgoing", "incoming"])
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_scalar_vs_vector_parity(self):
        s = DirectionInverseMatchStrategy()
        candidates = ["outgoing", "incoming", "outgoing", "unknown", None, "OUTGOING"]
        input_val = "incoming"
        scalar_scores = [s.score_pair(input_val, c) for c in candidates]
        vector_scores = s.score_many(input_val, candidates)
        np.testing.assert_array_almost_equal(vector_scores, scalar_scores)
