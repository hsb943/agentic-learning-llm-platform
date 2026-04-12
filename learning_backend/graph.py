"""Graph construction and routing for the learning backend."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import RuntimeDependencies
from .services import LearningWorkflow
from .state import LearningState


def route_context(state: LearningState) -> str:
    """Process supplied context directly or generate new search queries."""

    if state.get("context"):
        return "chunk_context"
    return "generate_query"


def route_verification(state: LearningState) -> str:
    """Choose the next node after answer verification."""

    current_checkpoint = state["current_checkpoint"]
    if state["verifications"].understanding_level < 0.7:
        return "teach_concept"
    if current_checkpoint + 1 < len(state["checkpoints"].checkpoints):
        return "next_checkpoint"
    return END


def route_teaching(state: LearningState) -> str:
    """Advance after teaching or finish the workflow."""

    current_checkpoint = state["current_checkpoint"]
    if current_checkpoint + 1 < len(state["checkpoints"].checkpoints):
        return "next_checkpoint"
    return END


def route_search(state: LearningState) -> str:
    """Generate a question when context is sufficient, otherwise search again."""

    if state["search_queries"] is None:
        return "generate_question"
    return "search_web"


def build_learning_graph(runtime: RuntimeDependencies):
    """Build the LangGraph workflow from the extracted notebook logic."""

    workflow = LearningWorkflow(runtime)
    searcher = StateGraph(LearningState)
    memory = MemorySaver()

    searcher.add_node("generate_query", workflow.generate_query)
    searcher.add_node("search_web", workflow.search_web)
    searcher.add_node("chunk_context", workflow.chunk_context)
    searcher.add_node("context_validation", workflow.context_validation)
    searcher.add_node("generate_checkpoints", workflow.generate_checkpoints)
    searcher.add_node("generate_question", workflow.generate_question)
    searcher.add_node("next_checkpoint", workflow.next_checkpoint)
    searcher.add_node("user_answer", workflow.user_answer)
    searcher.add_node("verify_answer", workflow.verify_answer)
    searcher.add_node("teach_concept", workflow.teach_concept)

    searcher.add_edge(START, "generate_checkpoints")
    searcher.add_conditional_edges(
        "generate_checkpoints",
        route_context,
        ["chunk_context", "generate_query"],
    )
    searcher.add_edge("generate_query", "search_web")
    searcher.add_edge("search_web", "generate_question")
    searcher.add_edge("chunk_context", "context_validation")
    searcher.add_conditional_edges(
        "context_validation",
        route_search,
        ["search_web", "generate_question"],
    )

    searcher.add_edge("generate_question", "user_answer")
    searcher.add_edge("user_answer", "verify_answer")
    searcher.add_conditional_edges(
        "verify_answer",
        route_verification,
        {
            "next_checkpoint": "next_checkpoint",
            "teach_concept": "teach_concept",
            END: END,
        },
    )
    searcher.add_conditional_edges(
        "teach_concept",
        route_teaching,
        {"next_checkpoint": "next_checkpoint", END: END},
    )
    searcher.add_edge("next_checkpoint", "generate_question")

    graph = searcher.compile(
        interrupt_after=["generate_checkpoints"],
        interrupt_before=["user_answer"],
        checkpointer=memory,
    )
    return graph
