"""Small example showing how to use the extracted backend package."""

from __future__ import annotations

from .config import build_runtime
from .graph import build_learning_graph
from .helpers import print_checkpoints


SAMPLE_NOTE = """
Transformer neural networks rely on self-attention to process tokens in parallel.
Important topics include positional encoding, encoder-decoder structure, tokenization,
training at scale, and fine-tuning for downstream tasks.
""".strip()


def run_example() -> None:
    """Run a minimal checkpoint-generation example."""

    runtime = build_runtime()
    graph = build_learning_graph(runtime)
    initial_input = {
        "topic": "Transformer Neural Network",
        "goals": [
            "I am a machine learning engineer and want to master transformer neural networks."
        ],
        "context": SAMPLE_NOTE,
        "current_checkpoint": 0,
    }
    thread = {"configurable": {"thread_id": "example-thread"}}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        checkpoints = event.get("checkpoints")
        if checkpoints:
            print_checkpoints(checkpoints)


if __name__ == "__main__":
    run_example()
