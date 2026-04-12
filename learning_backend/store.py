"""In-memory stores used by the backend."""

from __future__ import annotations

import uuid

from langgraph.store.memory import InMemoryStore


class ContextStore:
    """Store for managing context chunks and embeddings in memory."""

    def __init__(self) -> None:
        self.store = InMemoryStore()

    def save_context(
        self,
        context_chunks: list[str],
        embeddings: list[list[float]],
        key: str | None = None,
    ) -> str:
        namespace = ("context",)
        context_key = key or str(uuid.uuid4())
        value = {"chunks": context_chunks, "embeddings": embeddings}
        self.store.put(namespace, context_key, value)
        return context_key

    def get_context(self, context_key: str) -> dict:
        namespace = ("context",)
        memory = self.store.get(namespace, context_key)
        return memory.value
