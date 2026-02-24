from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Input/Output Contract ---


class LinkData(BaseModel):
    """A single link within a document. Fields are dynamic based on config."""

    model_config = ConfigDict(extra="allow")

    direction: Optional[str] = None


class DocumentLinks(BaseModel):
    id: str
    links: list[LinkData] = Field(default_factory=list)


class MatchingContract(BaseModel):
    """Top-level contract passed into the matching model."""

    input_documents: list[DocumentLinks]
    unmatched_documents: list[DocumentLinks]


# --- Explainability ---


class FieldScoreDetail(BaseModel):
    field_name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    contribution: float = Field(ge=0.0)


class CandidateExplainability(BaseModel):
    candidate_document_id: str
    candidate_link_index: int
    overall_score: float = Field(ge=0.0, le=1.0)
    field_details: list[FieldScoreDetail]


class MatchingExplanation(BaseModel):
    input_document_id: str
    input_link_index: int
    candidates: list[CandidateExplainability]


# --- Config models (parsed from YAML) ---


class FieldConfig(BaseModel):
    name: str
    weight: float = Field(gt=0.0)
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)


class MatchingConfig(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0, default=0.85)
    fields: list[FieldConfig]


# --- Result model ---


class MatchResult(BaseModel):
    best_candidate_document_id: Optional[str] = None
    best_candidate_link_index: Optional[int] = None
    best_score: float = 0.0
    matched: bool = False
    explanation: Optional[MatchingExplanation] = None
