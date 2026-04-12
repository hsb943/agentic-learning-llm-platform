"""Workflow node implementations extracted from the notebook."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.utils.math import cosine_similarity
from langchain_core.messages import HumanMessage, SystemMessage
from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import OpenAIEncoder

from .config import RuntimeDependencies
from .helpers import (
    extract_content_from_chunks,
    format_checkpoints_as_message,
    generate_checkpoint_message,
)
from .prompts import (
    ANSWER_VERIFIER,
    CHECKPOINT_BASED_QUERY_GENERATOR,
    FEYNMAN_TEACHER,
    LEARNING_CHECKPOINTS_GENERATOR,
    QUESTION_GENERATOR,
    VALIDATE_CONTEXT,
)
from .schemas import (
    Checkpoints,
    FeynmanTeaching,
    InContext,
    LearningVerification,
    QuestionOutput,
    SearchQuery,
)
from .state import LearningState


@dataclass(slots=True)
class LearningWorkflow:
    """Encapsulates the notebook's workflow logic as reusable methods."""

    runtime: RuntimeDependencies

    def generate_query(self, state: LearningState) -> dict:
        structured_llm = self.runtime.llm.with_structured_output(SearchQuery)
        checkpoints_message = HumanMessage(
            content=format_checkpoints_as_message(state["checkpoints"])
        )
        messages = [CHECKPOINT_BASED_QUERY_GENERATOR, checkpoints_message]
        search_queries = structured_llm.invoke(messages)
        return {"search_queries": search_queries}

    def search_web(self, state: LearningState) -> dict:
        search_queries = state["search_queries"].search_queries
        all_search_docs: list[dict] = []
        for query in search_queries:
            search_docs = self.runtime.tavily_search.invoke(query)
            all_search_docs.extend(search_docs)

        formatted_search_docs = [
            f'Context: {doc["content"]}\n Source: {doc["url"]}\n'
            for doc in all_search_docs
        ]

        chunk_embeddings = self.runtime.embeddings.embed_documents(formatted_search_docs)
        context_key = self.runtime.context_store.save_context(
            formatted_search_docs,
            chunk_embeddings,
            key=state.get("context_key"),
        )
        return {"context_chunks": formatted_search_docs, "context_key": context_key}

    def generate_checkpoints(self, state: LearningState) -> dict:
        structured_llm = self.runtime.llm.with_structured_output(Checkpoints)
        messages = [
            LEARNING_CHECKPOINTS_GENERATOR,
            SystemMessage(content=f"Topic: {state['topic']}"),
            SystemMessage(
                content=f"Goals: {', '.join(str(goal) for goal in state['goals'])}"
            ),
        ]
        checkpoints = structured_llm.invoke(messages)
        return {"checkpoints": checkpoints}

    def chunk_context(self, state: LearningState) -> dict:
        encoder = OpenAIEncoder(name="text-embedding-3-large")
        chunker = StatisticalChunker(
            encoder=encoder,
            min_split_tokens=128,
            max_split_tokens=512,
        )

        chunks = chunker([state["context"]])
        content = [extract_content_from_chunks(chunk) for chunk in chunks]

        chunk_embeddings = self.runtime.embeddings.embed_documents(content)
        context_key = self.runtime.context_store.save_context(
            content,
            chunk_embeddings,
            key=state.get("context_key"),
        )
        return {"context_chunks": content, "context_key": context_key}

    def context_validation(self, state: LearningState) -> dict:
        context = self.runtime.context_store.get_context(state["context_key"])
        chunks = context["chunks"]
        chunk_embeddings = context["embeddings"]

        checks = []
        structured_llm = self.runtime.llm.with_structured_output(InContext)

        for checkpoint in state["checkpoints"].checkpoints:
            query = self.runtime.embeddings.embed_query(checkpoint.verification)
            similarities = cosine_similarity([query], chunk_embeddings)[0]
            top_indices = sorted(
                range(len(similarities)),
                key=lambda index: similarities[index],
                reverse=True,
            )[:3]
            relevant_chunks = [chunks[index] for index in top_indices]

            messages = [
                VALIDATE_CONTEXT,
                HumanMessage(
                    content=(
                        "Criteria:\n"
                        f"{chr(10).join(f'- {criterion}' for criterion in checkpoint.criteria)}\n\n"
                        "Context:\n"
                        f"{chr(10).join(relevant_chunks)}"
                    )
                ),
            ]

            response = structured_llm.invoke(messages)
            if response.is_in_context.lower() == "no":
                checks.append(checkpoint)

        if checks:
            structured_llm = self.runtime.llm.with_structured_output(SearchQuery)
            checkpoints_message = generate_checkpoint_message(checks)
            messages = [CHECKPOINT_BASED_QUERY_GENERATOR, checkpoints_message]
            search_queries = structured_llm.invoke(messages)
            return {"search_queries": search_queries}

        return {"search_queries": None}

    def generate_question(self, state: LearningState) -> dict:
        structured_llm = self.runtime.llm.with_structured_output(QuestionOutput)
        current_checkpoint = state["current_checkpoint"]
        checkpoint_info = state["checkpoints"].checkpoints[current_checkpoint]
        messages = [
            QUESTION_GENERATOR,
            HumanMessage(
                content=(
                    f"Checkpoint Description: {checkpoint_info.description}\n"
                    "Success Criteria:\n"
                    f"{chr(10).join(f'- {criterion}' for criterion in checkpoint_info.criteria)}\n"
                    f"Verification Method: {checkpoint_info.verification}\n\n"
                    "Generate an appropriate verification question."
                )
            ),
        ]
        question_output = structured_llm.invoke(messages)
        return {"current_question": question_output.question}

    def verify_answer(self, state: LearningState) -> dict:
        structured_llm = self.runtime.llm.with_structured_output(LearningVerification)
        current_checkpoint = state["current_checkpoint"]
        checkpoint_info = state["checkpoints"].checkpoints[current_checkpoint]

        context = self.runtime.context_store.get_context(state["context_key"])
        chunks = context["chunks"]
        chunk_embeddings = context["embeddings"]

        query = self.runtime.embeddings.embed_query(checkpoint_info.verification)
        similarities = cosine_similarity([query], chunk_embeddings)[0]
        top_indices = sorted(
            range(len(similarities)),
            key=lambda index: similarities[index],
            reverse=True,
        )[:3]
        relevant_chunks = [chunks[index] for index in top_indices]

        messages = [
            ANSWER_VERIFIER,
            HumanMessage(
                content=(
                    f"Question: {state['current_question']}\n"
                    f"Answer: {state['current_answer']}\n\n"
                    f"Checkpoint Description: {checkpoint_info.description}\n"
                    "Success Criteria:\n"
                    f"{chr(10).join(f'- {criterion}' for criterion in checkpoint_info.criteria)}\n"
                    f"Verification Method: {checkpoint_info.verification}\n\n"
                    "Context:\n"
                    f"{chr(10).join(relevant_chunks)}\n\n"
                    "Assess the answer."
                )
            ),
        ]
        verification = structured_llm.invoke(messages)
        return {"verifications": verification}

    def teach_concept(self, state: LearningState) -> dict:
        structured_llm = self.runtime.llm.with_structured_output(FeynmanTeaching)
        current_checkpoint = state["current_checkpoint"]
        checkpoint_info = state["checkpoints"].checkpoints[current_checkpoint]
        messages = [
            FEYNMAN_TEACHER,
            HumanMessage(
                content=(
                    f"Criteria: {checkpoint_info.criteria}\n"
                    f"Verification: {state['verifications']}\n\n"
                    "Context:\n"
                    f"{state['context_chunks']}\n\n"
                    "Create a Feynman teaching explanation."
                )
            ),
        ]
        teaching = structured_llm.invoke(messages)
        return {"teachings": teaching}

    @staticmethod
    def user_answer(_: LearningState) -> None:
        """Placeholder node; the backend should inject the real user answer."""

    @staticmethod
    def next_checkpoint(state: LearningState) -> dict:
        """Advance to the next checkpoint in the learning sequence."""

        current_checkpoint = state["current_checkpoint"] + 1
        return {"current_checkpoint": current_checkpoint}
