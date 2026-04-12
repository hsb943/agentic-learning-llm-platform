"""Request and response models for the backend API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .schemas import LearningCheckpoint


class StartSessionRequest(BaseModel):
    topic: str
    goals: list[str]
    note: str
    thread_id: str | None = Field(
        default=None,
        description="Optional caller-provided workflow identifier.",
    )


class ConfirmCheckpointsRequest(BaseModel):
    checkpoints: list[LearningCheckpoint]


class SubmitAnswerRequest(BaseModel):
    answer: str


class SessionStateResponse(BaseModel):
    thread_id: str
    status: str
    next_nodes: list[str]
    values: dict[str, Any]


class QuestionResponse(BaseModel):
    thread_id: str
    current_checkpoint: int
    question: str
    status: str


class HealthResponse(BaseModel):
    ok: bool
