"""Session-oriented helpers for driving the learning graph over HTTP."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from pydantic import BaseModel

from .api_models import (
    ConfirmCheckpointsRequest,
    QuestionResponse,
    SessionStateResponse,
    StartSessionRequest,
)
from .config import RuntimeDependencies, build_runtime
from .graph import build_learning_graph
from .schemas import Checkpoints


def _serialize(value):
    """Convert graph state values into JSON-friendly Python data."""

    if isinstance(value, BaseModel):
        return {key: _serialize(item) for key, item in value.model_dump().items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@dataclass(slots=True)
class LearningAPIManager:
    """Own the graph and expose session-friendly operations."""

    runtime: RuntimeDependencies = field(default_factory=build_runtime)

    def __post_init__(self) -> None:
        self.graph = build_learning_graph(self.runtime)
        self.sessions: set[str] = set()

    @staticmethod
    def _thread(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _drain_stream(self, thread_id: str, payload=None) -> dict | None:
        thread = self._thread(thread_id)
        last_event = None
        for event in self.graph.stream(payload, thread, stream_mode="values"):
            last_event = event
        return last_event

    def _snapshot_to_response(self, thread_id: str) -> SessionStateResponse:
        snapshot = self.graph.get_state(self._thread(thread_id))
        values = _serialize(dict(snapshot.values))
        next_nodes = [str(node) for node in snapshot.next]

        if not next_nodes:
            status = "completed"
        elif "user_answer" in next_nodes:
            status = "waiting_for_answer"
        elif "generate_checkpoints" in next_nodes:
            status = "awaiting_checkpoint_confirmation"
        else:
            status = "running"

        return SessionStateResponse(
            thread_id=thread_id,
            status=status,
            next_nodes=next_nodes,
            values=values,
        )

    def start_session(self, request: StartSessionRequest) -> SessionStateResponse:
        thread_id = request.thread_id or str(uuid4())
        payload = {
            "topic": request.topic,
            "goals": request.goals,
            "context": request.note,
            "current_checkpoint": 0,
        }
        self._drain_stream(thread_id, payload)
        self.sessions.add(thread_id)
        return self._snapshot_to_response(thread_id)

    def ensure_session(self, thread_id: str) -> None:
        if thread_id not in self.sessions:
            raise KeyError(thread_id)

    def get_session_state(self, thread_id: str) -> SessionStateResponse:
        self.ensure_session(thread_id)
        return self._snapshot_to_response(thread_id)

    def confirm_checkpoints(
        self, thread_id: str, request: ConfirmCheckpointsRequest
    ) -> SessionStateResponse:
        self.ensure_session(thread_id)
        checkpoints = Checkpoints(checkpoints=request.checkpoints)
        self.graph.update_state(
            self._thread(thread_id),
            {"checkpoints": checkpoints},
            as_node="generate_checkpoints",
        )
        self._drain_stream(thread_id, None)
        return self._snapshot_to_response(thread_id)

    def get_current_question(self, thread_id: str) -> QuestionResponse:
        self.ensure_session(thread_id)
        snapshot = self.graph.get_state(self._thread(thread_id))
        current_question = snapshot.values.get("current_question", "")
        if not current_question:
            self._drain_stream(thread_id, None)
            snapshot = self.graph.get_state(self._thread(thread_id))
            current_question = snapshot.values.get("current_question", "")

        if not current_question:
            raise ValueError("No question is available for the current session.")

        return QuestionResponse(
            thread_id=thread_id,
            current_checkpoint=int(snapshot.values.get("current_checkpoint", 0)),
            question=str(current_question),
            status="waiting_for_answer",
        )

    def submit_answer(self, thread_id: str, answer: str) -> SessionStateResponse:
        self.ensure_session(thread_id)
        self.graph.update_state(
            self._thread(thread_id),
            {"current_answer": answer},
            as_node="user_answer",
        )
        self._drain_stream(thread_id, None)
        return self._snapshot_to_response(thread_id)
