"""Workflow state definitions."""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from .schemas import (
    Checkpoints,
    FeynmanTeaching,
    Goals,
    LearningVerification,
    SearchQuery,
)


class LearningState(TypedDict, total=False):
    topic: str
    goals: List[Goals]
    context: str
    context_chunks: Annotated[list[str], operator.add]
    context_key: str
    search_queries: SearchQuery | None
    checkpoints: Checkpoints
    verifications: LearningVerification
    teachings: FeynmanTeaching
    current_checkpoint: int
    current_question: str
    current_answer: str
