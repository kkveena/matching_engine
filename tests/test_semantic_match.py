import numpy as np
import pytest

from nerds_nlp.models.edge.matching.strategies.semantic_match import (
    SemanticMatchStrategy,
)

# Skip all tests if sentence-transformers is not installed
pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed")


class TestSemanticMatchStrategy:
    """Tests for SemanticMatchStrategy using sentence-transformers.

    These tests are skipped when sentence-transformers is not installed.
    """

    @pytest.fixture(autouse=True)
    def _strategy(self):
        """Shared strategy instance to avoid reloading model per test."""
        self.s = SemanticMatchStrategy(model_name="all-MiniLM-L6-v2")

    # --- score_pair ---

    def test_identical_strings(self):
        assert self.s.score_pair("Amount Paid", "Amount Paid") == 1.0

    def test_semantic_similar(self):
        score = self.s.score_pair("Amount Paid", "Money Received")
        assert score > 0.3  # semantically related

    def test_semantic_synonym(self):
        score = self.s.score_pair("Legal Entity", "Counterparty")
        assert score > 0.1  # related financial concepts

    def test_completely_unrelated(self):
        score = self.s.score_pair("Amount Paid", "Blue Whale")
        related_score = self.s.score_pair("Amount Paid", "Money Received")
        assert score < related_score

    def test_empty_vs_nonempty(self):
        assert self.s.score_pair("", "hello") == 0.0

    def test_nonempty_vs_empty(self):
        assert self.s.score_pair("hello", "") == 0.0

    def test_both_empty(self):
        assert self.s.score_pair("", "") == 1.0

    def test_none_input(self):
        assert self.s.score_pair(None, "hello") == 0.0

    def test_none_candidate(self):
        assert self.s.score_pair("hello", None) == 0.0

    def test_both_none(self):
        assert self.s.score_pair(None, None) == 1.0

    # --- threshold ---

    def test_threshold_filters_low_scores(self):
        s = SemanticMatchStrategy(threshold=0.99)
        # Non-identical strings should fall below 0.99
        score = s.score_pair("Amount Paid", "Money Received")
        assert score == 0.0

    def test_threshold_passes_identical(self):
        s = SemanticMatchStrategy(threshold=0.99)
        assert s.score_pair("hello", "hello") == 1.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            SemanticMatchStrategy(threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            SemanticMatchStrategy(threshold=-0.1)

    # --- score_many ---

    def test_score_many_basic(self):
        candidates = ["Amount Paid", "Money Received", "Blue Whale", "Payment Amount"]
        result = self.s.score_many("Amount Paid", candidates)
        assert result.shape == (4,)
        assert result[0] == 1.0  # identical
        assert result[1] > result[2]  # semantic > unrelated

    def test_score_many_empty_list(self):
        result = self.s.score_many("hello", [])
        assert result.shape == (0,)

    def test_score_many_with_none_candidates(self):
        result = self.s.score_many("hello", ["hello", None, "", "world"])
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] > 0.0

    def test_score_many_none_input(self):
        result = self.s.score_many(None, ["hello", None, ""])
        np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

    # --- scalar vs vector parity ---

    def test_scalar_vs_vector_parity(self):
        candidates = ["Amount Paid", "Money Received", None, "", "Blue Whale"]
        input_val = "Amount Paid"
        scalar_scores = [self.s.score_pair(input_val, c) for c in candidates]
        vector_scores = self.s.score_many(input_val, candidates)
        np.testing.assert_array_almost_equal(vector_scores, scalar_scores, decimal=4)

    # --- score is in [0, 1] ---

    def test_scores_bounded(self):
        candidates = [
            "Amount Paid",
            "Money Received",
            "Legal Entity",
            "Counterparty",
            "Blue Whale",
            "Settlement Date",
            "12345",
        ]
        result = self.s.score_many("Amount Paid", candidates)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    # --- registry integration ---

    def test_registry_lookup(self):
        from nerds_nlp.models.edge.matching.strategies import get_strategy

        strategy = get_strategy("semantic_match", {"threshold": 0.5})
        assert isinstance(strategy, SemanticMatchStrategy)
        assert strategy.threshold == 0.5

    def test_registry_lookup_default_params(self):
        from nerds_nlp.models.edge.matching.strategies import get_strategy

        strategy = get_strategy("semantic_match")
        assert isinstance(strategy, SemanticMatchStrategy)
        assert strategy.threshold == 0.0


class TestSemanticMatchImportError:
    """Test that a clear error is raised when sentence-transformers is missing."""

    def test_import_error_message(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        s = SemanticMatchStrategy.__new__(SemanticMatchStrategy)
        s.model_name = "all-MiniLM-L6-v2"
        s.threshold = 0.0
        s._model = None

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install matching-engine"):
            s._get_model()
