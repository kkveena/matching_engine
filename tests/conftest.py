import pytest

from nerds_nlp.models.edge.matching.schemas import FieldConfig


@pytest.fixture
def default_field_configs() -> list[FieldConfig]:
    """Standard field configuration matching the YAML spec."""
    return [
        FieldConfig(name="valuation_date", weight=0.3, strategy="exact_match"),
        FieldConfig(
            name="forward_rate",
            weight=0.25,
            strategy="numeric_range",
            params={"tolerance": 0.01},
        ),
        FieldConfig(name="settlement_date", weight=0.25, strategy="exact_match"),
        FieldConfig(name="direction", weight=0.2, strategy="direction_inverse"),
    ]


@pytest.fixture
def sample_input_link() -> dict:
    return {
        "valuation_date": "2025-01-15",
        "forward_rate": "1.2345",
        "settlement_date": "2025-03-15",
        "direction": "incoming",
    }


@pytest.fixture
def sample_candidate_links() -> list[dict]:
    return [
        {  # Perfect match
            "valuation_date": "2025-01-15",
            "forward_rate": "1.2345",
            "settlement_date": "2025-03-15",
            "direction": "outgoing",
        },
        {  # Partial match (wrong date, close rate)
            "valuation_date": "2025-01-16",
            "forward_rate": "1.2340",
            "settlement_date": "2025-03-15",
            "direction": "outgoing",
        },
        {  # No match
            "valuation_date": "2024-12-01",
            "forward_rate": "2.0",
            "settlement_date": "2024-12-15",
            "direction": "incoming",
        },
    ]


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "matching": {
            "threshold": 0.85,
            "fields": [
                {"name": "valuation_date", "weight": 0.3, "strategy": "exact_match"},
                {
                    "name": "forward_rate",
                    "weight": 0.25,
                    "strategy": "numeric_range",
                    "params": {"tolerance": 0.01},
                },
                {
                    "name": "settlement_date",
                    "weight": 0.25,
                    "strategy": "exact_match",
                },
                {
                    "name": "direction",
                    "weight": 0.2,
                    "strategy": "direction_inverse",
                },
            ],
        }
    }
