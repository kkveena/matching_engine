# Matrix-Based Matching Engine

A matrix-wise document matching engine that compares 1 incoming document link against N outgoing document links, producing probability-like scores in [0,1] with optional per-candidate explainability.

## Features

- **Matrix scoring** — Vectorized 1-to-N matching using NumPy for performance
- **Strategy pattern** — Pluggable field comparison strategies (exact match, numeric range, direction inverse)
- **YAML-driven configuration** — Fields, weights, strategies, and thresholds configured via YAML
- **Normalized weighted scoring** — Weighted field scores normalized to [0,1]
- **Threshold-based acceptance** — Configurable threshold determines match/no-match
- **Explainability** — Optional per-field score breakdown for each candidate
- **Missing-field semantics** — Input missing a field excludes it from scoring; candidate missing a field scores 0

## Project Structure

```
matching_engine/
├── config/
│   └── matching_config.yaml            # Matching configuration
├── nerds_nlp/
│   └── models/
│       └── edge/
│           └── matching/
│               ├── schemas.py           # Pydantic data models
│               ├── matrix_matcher.py    # Core vectorized scoring engine
│               ├── edge_matching_model.py  # Main orchestration model
│               └── strategies/
│                   ├── base.py          # Abstract base strategy
│                   ├── exact_match.py   # String equality strategy
│                   ├── numeric_range.py # Triangular decay strategy
│                   ├── direction_inverse.py  # Direction inversion strategy
│                   └── registry.py      # Strategy factory/registry
├── tests/
│   ├── conftest.py                     # Shared test fixtures
│   ├── test_strategies.py              # 33 strategy tests
│   ├── test_matrix_matcher.py          # 11 matrix matcher tests
│   └── test_edge_matching_model.py     # 13 integration tests
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
pip install -e ".[dev]"
```

**Dependencies:** pydantic>=2.0, numpy>=1.26, pyyaml>=6.0 | **Dev:** pytest>=8.0, pytest-cov>=4.0

## Quick Start

```python
from nerds_nlp.models.edge.matching import KKEdgeMatchingModel

model = KKEdgeMatchingModel()  # Loads config/matching_config.yaml

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

results = model.match(contract)

for result in results:
    print(f"Matched: {result.matched}")
    print(f"Best candidate: {result.best_candidate_document_id}")
    print(f"Score: {result.best_score:.4f}")
```

## Configuration

The matching engine is configured via `config/matching_config.yaml`:

```yaml
matching:
  threshold: 0.85          # Minimum score to accept a match
  fields:
    - name: valuation_date
      weight: 0.3           # Relative weight for scoring
      strategy: exact_match  # Strategy to use for comparison
    - name: forward_rate
      weight: 0.25
      strategy: numeric_range
      params:                # Strategy-specific parameters
        tolerance: 0.01
    - name: settlement_date
      weight: 0.25
      strategy: exact_match
    - name: direction
      weight: 0.2
      strategy: direction_inverse
```

You can also pass configuration directly:

```python
model = KKEdgeMatchingModel(config_dict={
    "matching": {
        "threshold": 0.90,
        "fields": [
            {"name": "valuation_date", "weight": 0.5, "strategy": "exact_match"},
            {"name": "forward_rate", "weight": 0.5, "strategy": "numeric_range", "params": {"tolerance": 0.05}},
        ],
    }
})
```

## Architecture

The matching pipeline follows a 3-phase flow:

1. **Field Filtering** — Determines which configured fields are present on the input link. Missing input fields are excluded from the scoring denominator entirely.

2. **Matrix Scoring** — For each active field, the configured strategy's `score_many()` method computes scores for all N candidates in a single vectorized call. Candidates missing a field receive a score of 0 for that field.

3. **Weighted Aggregation** — Per-field scores are multiplied by their weights and summed, then divided by the total weight of active fields to produce a normalized score in [0,1].

### Key Classes

| Class | Purpose |
|-------|---------|
| `KKEdgeMatchingModel` | Top-level orchestrator: loads config, parses contracts, runs matching, builds explainability |
| `MatrixMatcher` | Core scoring engine: vectorized 1-to-N field comparison with weighted aggregation |
| `BaseStrategy` | Abstract base with `score_pair()` (scalar) and `score_many()` (vectorized) |
| `MatchingConfig` | Pydantic model for YAML configuration validation |
| `MatchResult` | Output model with best match, score, and optional explanation |

## Strategies

| Strategy | Config Name | Description | Parameters |
|----------|-------------|-------------|------------|
| `ExactMatchStrategy` | `exact_match` | Case-insensitive string equality. Returns 1.0 if equal, 0.0 otherwise. | None |
| `NumericRangeMatchStrategy` | `numeric_range` | Triangular decay: `max(0, 1 - |diff| / tolerance)`. Returns 1.0 for exact match, 0.0 beyond tolerance. | `tolerance` (float, required) |
| `DirectionInverseMatchStrategy` | `direction_inverse` | Returns 1.0 if candidate direction is the inverse of input (incoming/outgoing, buy/sell). | None |

### Custom Strategies

Register custom strategies at runtime:

```python
from nerds_nlp.models.edge.matching.strategies import BaseStrategy, register_strategy

class MyStrategy(BaseStrategy):
    def score_pair(self, input_value, candidate_value):
        # Your comparison logic returning a float in [0, 1]
        ...

register_strategy("my_strategy", MyStrategy)
```

## Explainability

Enable explainability to get per-field score breakdowns:

```python
model = KKEdgeMatchingModel(
    return_explainability=True,
    top_k_explainability=5,  # Only include top 5 candidates
)

results = model.match(contract)
explanation = results[0].explanation

for candidate in explanation.candidates:
    print(f"Candidate: {candidate.candidate_document_id} (score: {candidate.overall_score:.4f})")
    for detail in candidate.field_details:
        print(f"  {detail.field_name}: score={detail.score:.4f}, weight={detail.weight}, contribution={detail.contribution:.4f}")
```

The explainability guarantee: `sum(contributions) / total_weight == overall_score` for every candidate.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nerds_nlp --cov-report=term-missing

# Run specific test files
pytest tests/test_strategies.py -v
pytest tests/test_matrix_matcher.py -v
pytest tests/test_edge_matching_model.py -v
```

57 tests covering strategy correctness, scalar/vector parity, missing-field semantics, weighted contribution consistency, threshold behavior, explainability correctness, and result serialization.
