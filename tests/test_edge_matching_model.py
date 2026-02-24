import pytest

from nerds_nlp.models.edge.matching.edge_matching_model import KKEdgeMatchingModel


class TestKKEdgeMatchingModel:
    def _make_contract(self) -> dict:
        return {
            "input_documents": [
                {
                    "id": "input-1",
                    "links": [
                        {
                            "valuation_date": "2025-01-15",
                            "forward_rate": "1.2345",
                            "settlement_date": "2025-03-15",
                            "direction": "incoming",
                        }
                    ],
                }
            ],
            "unmatched_documents": [
                {
                    "id": "out-1",
                    "links": [
                        {
                            "valuation_date": "2025-01-15",
                            "forward_rate": "1.2345",
                            "settlement_date": "2025-03-15",
                            "direction": "outgoing",
                        }
                    ],
                },
                {
                    "id": "out-2",
                    "links": [
                        {
                            "valuation_date": "2024-12-01",
                            "forward_rate": "2.0",
                            "settlement_date": "2024-12-15",
                            "direction": "incoming",
                        }
                    ],
                },
            ],
        }

    def test_perfect_match_above_threshold(self, sample_config_dict):
        model = KKEdgeMatchingModel(config_dict=sample_config_dict)
        results = model.match(self._make_contract())
        assert len(results) == 1
        assert results[0].matched is True
        assert results[0].best_candidate_document_id == "out-1"
        assert results[0].best_score == pytest.approx(1.0)

    def test_threshold_behavior_no_match(self, sample_config_dict):
        """Set threshold very high so nothing matches."""
        sample_config_dict["matching"]["threshold"] = 0.999
        model = KKEdgeMatchingModel(config_dict=sample_config_dict)
        contract = self._make_contract()
        # Modify out-1 to be a near-miss
        contract["unmatched_documents"][0]["links"][0]["forward_rate"] = "1.2350"
        results = model.match(contract)
        assert results[0].matched is False
        assert results[0].best_candidate_document_id is None
        assert results[0].best_candidate_link_index is None

    def test_threshold_behavior_just_above(self, sample_config_dict):
        """Perfect match should be at 1.0, above any threshold < 1.0."""
        sample_config_dict["matching"]["threshold"] = 0.99
        model = KKEdgeMatchingModel(config_dict=sample_config_dict)
        results = model.match(self._make_contract())
        assert results[0].matched is True
        assert results[0].best_score == pytest.approx(1.0)

    def test_explainability_structure(self, sample_config_dict):
        model = KKEdgeMatchingModel(
            config_dict=sample_config_dict, return_explainability=True
        )
        results = model.match(self._make_contract())
        assert results[0].explanation is not None
        explanation = results[0].explanation
        assert explanation.input_document_id == "input-1"
        assert explanation.input_link_index == 0
        assert len(explanation.candidates) == 2

    def test_explainability_score_consistency(self, sample_config_dict):
        """Verify sum(contributions)/total_weight == overall_score in explainability."""
        model = KKEdgeMatchingModel(
            config_dict=sample_config_dict, return_explainability=True
        )
        results = model.match(self._make_contract())
        explanation = results[0].explanation
        total_weight = sum(
            f["weight"] for f in sample_config_dict["matching"]["fields"]
        )
        for candidate in explanation.candidates:
            contrib_sum = sum(fd.contribution for fd in candidate.field_details)
            expected_score = contrib_sum / total_weight
            assert candidate.overall_score == pytest.approx(expected_score, abs=1e-10)

    def test_explainability_field_details(self, sample_config_dict):
        """Check that field details contain all configured fields."""
        model = KKEdgeMatchingModel(
            config_dict=sample_config_dict, return_explainability=True
        )
        results = model.match(self._make_contract())
        explanation = results[0].explanation
        for candidate in explanation.candidates:
            field_names = [fd.field_name for fd in candidate.field_details]
            assert "valuation_date" in field_names
            assert "forward_rate" in field_names
            assert "settlement_date" in field_names
            assert "direction" in field_names

    def test_top_k_explainability(self, sample_config_dict):
        model = KKEdgeMatchingModel(
            config_dict=sample_config_dict,
            return_explainability=True,
            top_k_explainability=1,
        )
        results = model.match(self._make_contract())
        assert len(results[0].explanation.candidates) == 1
        # Top-1 should be the best match
        assert results[0].explanation.candidates[0].candidate_document_id == "out-1"

    def test_disabled_matching(self, sample_config_dict):
        model = KKEdgeMatchingModel(
            config_dict=sample_config_dict, enable_matrix_matching=False
        )
        results = model.match(self._make_contract())
        assert results == []

    def test_no_candidates(self, sample_config_dict):
        model = KKEdgeMatchingModel(config_dict=sample_config_dict)
        contract = {
            "input_documents": [
                {
                    "id": "input-1",
                    "links": [
                        {"valuation_date": "2025-01-15", "direction": "incoming"}
                    ],
                }
            ],
            "unmatched_documents": [],
        }
        results = model.match(contract)
        assert len(results) == 1
        assert results[0].matched is False
        assert results[0].best_score == 0.0

    def test_multiple_input_documents(self, sample_config_dict):
        """Multiple input documents should each produce a result."""
        model = KKEdgeMatchingModel(config_dict=sample_config_dict)
        contract = {
            "input_documents": [
                {
                    "id": "input-1",
                    "links": [
                        {
                            "valuation_date": "2025-01-15",
                            "forward_rate": "1.2345",
                            "settlement_date": "2025-03-15",
                            "direction": "incoming",
                        }
                    ],
                },
                {
                    "id": "input-2",
                    "links": [
                        {
                            "valuation_date": "2024-12-01",
                            "forward_rate": "2.0",
                            "settlement_date": "2024-12-15",
                            "direction": "outgoing",
                        }
                    ],
                },
            ],
            "unmatched_documents": [
                {
                    "id": "out-1",
                    "links": [
                        {
                            "valuation_date": "2025-01-15",
                            "forward_rate": "1.2345",
                            "settlement_date": "2025-03-15",
                            "direction": "outgoing",
                        }
                    ],
                },
                {
                    "id": "out-2",
                    "links": [
                        {
                            "valuation_date": "2024-12-01",
                            "forward_rate": "2.0",
                            "settlement_date": "2024-12-15",
                            "direction": "incoming",
                        }
                    ],
                },
            ],
        }
        results = model.match(contract)
        assert len(results) == 2
        assert results[0].matched is True
        assert results[0].best_candidate_document_id == "out-1"
        assert results[1].matched is True
        assert results[1].best_candidate_document_id == "out-2"

    def test_multiple_links_per_document(self, sample_config_dict):
        """Multiple links in one document should each be scored independently."""
        model = KKEdgeMatchingModel(config_dict=sample_config_dict)
        contract = {
            "input_documents": [
                {
                    "id": "input-1",
                    "links": [
                        {
                            "valuation_date": "2025-01-15",
                            "forward_rate": "1.2345",
                            "settlement_date": "2025-03-15",
                            "direction": "incoming",
                        },
                        {
                            "valuation_date": "2024-06-01",
                            "forward_rate": "3.0",
                            "settlement_date": "2024-06-15",
                            "direction": "incoming",
                        },
                    ],
                }
            ],
            "unmatched_documents": [
                {
                    "id": "out-1",
                    "links": [
                        {
                            "valuation_date": "2025-01-15",
                            "forward_rate": "1.2345",
                            "settlement_date": "2025-03-15",
                            "direction": "outgoing",
                        }
                    ],
                }
            ],
        }
        results = model.match(contract)
        assert len(results) == 2  # One result per input link
        assert results[0].matched is True
        assert results[0].best_score == pytest.approx(1.0)
        # Second link doesn't match well
        assert results[1].best_score < 0.85

    def test_config_from_yaml_file(self):
        """Load configuration from the actual YAML file."""
        model = KKEdgeMatchingModel()
        assert model.config.threshold == 0.85
        assert len(model.config.fields) == 4

    def test_result_serialization(self, sample_config_dict):
        """Ensure results can be serialized to dict/JSON."""
        model = KKEdgeMatchingModel(
            config_dict=sample_config_dict, return_explainability=True
        )
        results = model.match(self._make_contract())
        result_dict = results[0].model_dump()
        assert isinstance(result_dict, dict)
        assert "best_candidate_document_id" in result_dict
        assert "explanation" in result_dict
        assert result_dict["explanation"] is not None
