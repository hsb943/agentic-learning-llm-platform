"""Pydantic models used by the learning workflow."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Goals(BaseModel):
    """Structure for defining learning goals."""

    goals: str = Field(..., description="Learning goals")


class LearningCheckpoint(BaseModel):
    """Structure for a single checkpoint."""

    description: str = Field(..., description="Main checkpoint description")
    criteria: List[str] = Field(..., description="List of success criteria")
    verification: str = Field(..., description="How to verify this checkpoint")


class Checkpoints(BaseModel):
    """Main checkpoints container with index tracking."""

    checkpoints: List[LearningCheckpoint] = Field(
        ...,
        description="List of checkpoints covering foundation, application, and mastery levels",
    )


class SearchQuery(BaseModel):
    """Structure for search query collection."""

    search_queries: List[str] = Field(
        default_factory=list,
        description="Search queries for retrieval.",
    )


class LearningVerification(BaseModel):
    """Structure for verification results."""

    understanding_level: float = Field(..., ge=0, le=1)
    feedback: str
    suggestions: List[str]
    context_alignment: bool


class FeynmanTeaching(BaseModel):
    """Structure for Feynman teaching method."""

    simplified_explanation: str
    key_concepts: List[str]
    analogies: List[str]


class QuestionOutput(BaseModel):
    """Structure for question generation output."""

    question: str


class InContext(BaseModel):
    """Structure for context verification."""

    is_in_context: str = Field(..., description="Yes or No")
