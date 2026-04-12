"""Small formatting and printing helpers."""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import HumanMessage

from .schemas import Checkpoints, LearningCheckpoint


def extract_content_from_chunks(chunks: Iterable[object]) -> str:
    """Extract and combine content from chunk objects with a ``splits`` attribute."""

    content: list[str] = []
    for chunk in chunks:
        splits = getattr(chunk, "splits", None)
        if splits:
            content.append(" ".join(splits))
    return "\n".join(content)


def format_checkpoints_as_message(checkpoints: Checkpoints) -> str:
    """Convert checkpoints into a readable prompt string."""

    lines = ["Here are the learning checkpoints:", ""]
    for index, checkpoint in enumerate(checkpoints.checkpoints, start=1):
        lines.append(f"Checkpoint {index}:")
        lines.append(f"Description: {checkpoint.description}")
        lines.append("Success Criteria:")
        lines.extend(f"- {criterion}" for criterion in checkpoint.criteria)
        lines.append("")
    return "\n".join(lines).strip()


def generate_checkpoint_message(checks: list[LearningCheckpoint]) -> HumanMessage:
    """Generate a prompt for checkpoints that need additional context."""

    formatted_checks: list[str] = []
    for check in checks:
        checkpoint_text = (
            f"Description: {check.description}\n"
            "Success Criteria:\n"
            f"{chr(10).join(f'- {criterion}' for criterion in check.criteria)}\n"
            f"Verification Method: {check.verification}"
        )
        formatted_checks.append(checkpoint_text)

    all_checks = "\n---\n".join(formatted_checks)
    return HumanMessage(
        content=(
            "The following learning checkpoints need additional context:\n\n"
            f"{all_checks}\n\n"
            "Please generate search queries to find relevant information."
        )
    )


def print_checkpoints(checkpoints: Checkpoints) -> None:
    """Pretty-print checkpoints for local debugging."""

    for index, checkpoint in enumerate(checkpoints.checkpoints, start=1):
        print(f"Checkpoint {index}:")
        print("Description:", checkpoint.description)
        print("Criteria:", checkpoint.criteria)
        print("Verification:", checkpoint.verification)
        print()
