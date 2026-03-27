from typing import Any, Optional

import numpy as np

from .base import BaseStrategy


class SemanticMatchStrategy(BaseStrategy):
    """Scores based on semantic similarity using sentence embeddings.

    Uses sentence-transformers to compute cosine similarity between
    input and candidate text. Captures meaning-based matches like:
        "Amount Paid" vs "Money Received" -> high similarity
        "Legal Entity" vs "Counterparty"  -> moderate similarity

    Requires optional dependency: pip install matching-engine[semantic]

    The model is loaded once on first use and cached for the lifetime
    of the strategy instance.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.0,
    ):
        """
        Args:
            model_name: Name of the sentence-transformers model to use.
                        Default 'all-MiniLM-L6-v2' is small (~80MB) and fast.
            threshold: Minimum similarity to return a non-zero score.
                       Below this, score is forced to 0.0.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.model_name = model_name
        self.threshold = threshold
        self._model: Optional[Any] = None

    def _get_model(self) -> Any:
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "SemanticMatchStrategy requires 'sentence-transformers'. "
                    "Install it with: pip install matching-engine[semantic]"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def score_pair(self, input_value: Any, candidate_value: Any) -> float:
        a = self._normalize(input_value)
        b = self._normalize(candidate_value)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        model = self._get_model()
        embeddings = model.encode([a, b], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        # Clamp to [0, 1] — cosine similarity of normalized embeddings
        # can occasionally be slightly negative for unrelated texts
        similarity = max(0.0, min(1.0, similarity))
        return similarity if similarity >= self.threshold else 0.0

    def score_many(self, input_value: Any, candidate_values: list[Any]) -> np.ndarray:
        n = len(candidate_values)
        a = self._normalize(input_value)

        if not a:
            scores = np.zeros(n, dtype=np.float64)
            for i, cv in enumerate(candidate_values):
                if not self._normalize(cv):
                    scores[i] = 1.0
            return scores

        model = self._get_model()

        # Separate out empty candidates (score = 0)
        norm_candidates = [self._normalize(cv) for cv in candidate_values]
        non_empty_indices = [i for i, c in enumerate(norm_candidates) if c]

        scores = np.zeros(n, dtype=np.float64)

        if not non_empty_indices:
            return scores

        # Batch encode: input + all non-empty candidates in one call
        texts = [a] + [norm_candidates[i] for i in non_empty_indices]
        embeddings = model.encode(texts, normalize_embeddings=True)

        input_emb = embeddings[0]
        candidate_embs = embeddings[1:]

        # Vectorized cosine similarity (embeddings are already normalized)
        similarities = candidate_embs @ input_emb

        # Clamp to [0, 1] and apply threshold
        np.clip(similarities, 0.0, 1.0, out=similarities)
        if self.threshold > 0.0:
            similarities[similarities < self.threshold] = 0.0

        for idx, orig_i in enumerate(non_empty_indices):
            scores[orig_i] = similarities[idx]

        return scores
