from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from .matrix_matcher import MatrixMatcher
from .schemas import (
    CandidateExplainability,
    FieldScoreDetail,
    MatchingConfig,
    MatchingContract,
    MatchingExplanation,
    MatchResult,
)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "config" / "matching_config.yaml"
)


class KKEdgeMatchingModel:
    """Top-level matching model.

    1. Loads YAML configuration
    2. Extracts input/candidate links from contract JSON
    3. Delegates scoring to MatrixMatcher
    4. Applies threshold logic
    5. Optionally builds explainability output
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config_dict: Optional[dict[str, Any]] = None,
        enable_matrix_matching: bool = True,
        return_explainability: bool = False,
        top_k_explainability: Optional[int] = None,
    ):
        self.enable_matrix_matching = enable_matrix_matching
        self.return_explainability = return_explainability
        self.top_k_explainability = top_k_explainability

        if config_dict is not None:
            raw = config_dict
        else:
            path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
            with open(path, "r") as f:
                raw = yaml.safe_load(f)

        matching_raw = raw.get("matching", raw)
        self.config = MatchingConfig.model_validate(matching_raw)
        self.matcher = MatrixMatcher(self.config.fields)

    def match(
        self, contract: dict[str, Any] | MatchingContract
    ) -> list[MatchResult]:
        """Run matching on a contract. Returns one MatchResult per input link."""
        if not self.enable_matrix_matching:
            return []

        if isinstance(contract, dict):
            contract = MatchingContract.model_validate(contract)

        results: list[MatchResult] = []

        # Build candidate link pool: list of (doc_id, link_index, link_dict)
        candidate_pool: list[tuple[str, int, dict[str, Any]]] = []
        for doc in contract.unmatched_documents:
            for li, link in enumerate(doc.links):
                link_dict = link.model_dump()
                candidate_pool.append((doc.id, li, link_dict))

        if not candidate_pool:
            for doc in contract.input_documents:
                for _ in doc.links:
                    results.append(MatchResult())
            return results

        candidate_doc_ids = [cp[0] for cp in candidate_pool]
        candidate_link_indices = [cp[1] for cp in candidate_pool]
        candidate_link_dicts = [cp[2] for cp in candidate_pool]

        for input_doc in contract.input_documents:
            for input_li, input_link in enumerate(input_doc.links):
                input_dict = input_link.model_dump()

                mr = self.matcher.score_one_to_many(input_dict, candidate_link_dicts)

                matched = mr.best_score >= self.config.threshold

                explanation = None
                if self.return_explainability:
                    explanation = self._build_explanation(
                        input_document_id=input_doc.id,
                        input_link_index=input_li,
                        mr=mr,
                        candidate_doc_ids=candidate_doc_ids,
                        candidate_link_indices=candidate_link_indices,
                    )

                results.append(
                    MatchResult(
                        best_candidate_document_id=(
                            candidate_doc_ids[mr.best_index] if matched else None
                        ),
                        best_candidate_link_index=(
                            candidate_link_indices[mr.best_index] if matched else None
                        ),
                        best_score=mr.best_score,
                        matched=matched,
                        explanation=explanation,
                    )
                )

        return results

    def _build_explanation(
        self,
        input_document_id: str,
        input_link_index: int,
        mr: Any,
        candidate_doc_ids: list[str],
        candidate_link_indices: list[int],
    ) -> MatchingExplanation:
        """Build the explainability object from MatrixMatchResult."""
        n_candidates = len(candidate_doc_ids)

        if self.top_k_explainability is not None:
            sorted_indices = list(reversed(np.argsort(mr.scores).tolist()))
            indices = sorted_indices[: self.top_k_explainability]
        else:
            indices = list(range(n_candidates))

        total_weight = sum(
            fc.weight for fc in self.config.fields if fc.name in mr.used_fields
        )

        candidates = []
        for idx in indices:
            field_details = []
            for f_idx, field_name in enumerate(mr.used_fields):
                fc = next(f for f in self.config.fields if f.name == field_name)
                field_details.append(
                    FieldScoreDetail(
                        field_name=field_name,
                        score=float(mr.per_field_scores[idx, f_idx]),
                        weight=fc.weight,
                        contribution=float(mr.weighted_contributions[idx, f_idx]),
                    )
                )
            candidates.append(
                CandidateExplainability(
                    candidate_document_id=candidate_doc_ids[idx],
                    candidate_link_index=candidate_link_indices[idx],
                    overall_score=float(mr.scores[idx]),
                    field_details=field_details,
                )
            )

        return MatchingExplanation(
            input_document_id=input_document_id,
            input_link_index=input_link_index,
            candidates=candidates,
        )
