"""FastAPI app exposing the extracted learning workflow."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from .api_models import (
    ConfirmCheckpointsRequest,
    HealthResponse,
    SessionStateResponse,
    StartSessionRequest,
    SubmitAnswerRequest,
)
from .manager import LearningAPIManager


@lru_cache(maxsize=1)
def get_manager() -> LearningAPIManager:
    """Build the workflow manager lazily on first use."""

    return LearningAPIManager()


def create_app() -> FastAPI:
    """Create the backend API app."""

    app = FastAPI(title="Learning Backend API", version="0.1.0")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(ok=True)

    @app.post("/sessions/start", response_model=SessionStateResponse)
    def start_session(request: StartSessionRequest) -> SessionStateResponse:
        return get_manager().start_session(request)

    @app.get("/sessions/{thread_id}", response_model=SessionStateResponse)
    def get_session(thread_id: str) -> SessionStateResponse:
        try:
            return get_manager().get_session_state(thread_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found") from None

    @app.post(
        "/sessions/{thread_id}/checkpoints/confirm",
        response_model=SessionStateResponse,
    )
    def confirm_checkpoints(
        thread_id: str, request: ConfirmCheckpointsRequest
    ) -> SessionStateResponse:
        try:
            return get_manager().confirm_checkpoints(thread_id, request)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found") from None

    @app.get("/sessions/{thread_id}/question")
    def get_current_question(thread_id: str):
        try:
            return get_manager().get_current_question(thread_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{thread_id}/answer", response_model=SessionStateResponse)
    def submit_answer(
        thread_id: str, request: SubmitAnswerRequest
    ) -> SessionStateResponse:
        try:
            return get_manager().submit_answer(thread_id, request.answer)
        except KeyError:
            raise HTTPException(status_code=404, detail="Session not found") from None

    return app


app = create_app()
