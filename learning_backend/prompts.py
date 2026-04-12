"""Prompt messages used by the workflow."""

from __future__ import annotations

from langchain_core.messages import SystemMessage


LEARNING_CHECKPOINTS_GENERATOR = SystemMessage(
    content="""You will be given a learning topic title and learning objectives.

Your goal is to generate clear learning checkpoints that will help verify understanding and progress through the topic.

The output should be in the following dictionary structure:
checkpoint
-> description (level checkpoint description)
-> criteria
-> verification (How to verify this checkpoint (Feynman Methods))

Requirements for each checkpoint:
- Description should be clear and concise
- Criteria should be specific and measurable (3-5 items)
- Verification method should be practical and appropriate for the level
- Verification will be checked by language model, so it must by in natural language
- All elements should align with the learning objectives
- Use action verbs and clear language

Ensure all checkpoints progress logically from foundation to mastery.
IMPORTANT - ANSWER ONLY 3 CHECKPOINTS"""
)

CHECKPOINT_BASED_QUERY_GENERATOR = SystemMessage(
    content="""You will be given learning checkpoints for a topic.

Your goal is to generate search queries that will retrieve content matching each checkpoint's requirements from retrieval systems or web search.
Follow these steps:
1. Analyze each learning checkpoint carefully
2. For each checkpoint, generate ONE targeted search query that will retrieve:
   - Content for checkpoint verification"""
)

VALIDATE_CONTEXT = SystemMessage(
    content="""You will be given a learning criteria and context.
Check if the the criteria could be answered using the context.
Always answer YES or NO"""
)

QUESTION_GENERATOR = SystemMessage(
    content="""You will be given a checkpoint description, success criteria, and verification method.

Your goal is to generate an appropriate question that aligns with the checkpoint's verification requirements.
The question should:
1. Follow the specified verification method
2. Cover all success criteria
3. Encourage demonstration of understanding
4. Be clear and specific

Output should be a single, well-formulated question that effectively tests the checkpoint's learning objectives."""
)

ANSWER_VERIFIER = SystemMessage(
    content="""You will be given a student's answer, question, checkpoint details, and relevant context.

Your goal is to analyze the answer against the checkpoint criteria and provided context.
Analyze considering:
1. Alignment with verification method specified
2. Coverage of all success criteria
3. Use of relevant concepts from context
4. Depth and accuracy of understanding

Output should include:
- understanding_level: float between 0 and 1
- feedback: detailed explanation of the assessment
- suggestions: list of specific improvements
- context_alignment: boolean indicating if the answer aligns with provided context"""
)

FEYNMAN_TEACHER = SystemMessage(
    content="""You will be given verification results, checkpoint criteria, and learning context.

Your goal is to create a Feynman-style teaching explanation for concepts that need reinforcement.
The explanation should include:
1. Simplified explanation without technical jargon
2. Concrete, relatable analogies
3. Key concepts to remember

Output should follow the Feynman technique:
- simplified_explanation: clear, jargon-free explanation
- key_concepts: list of essential points
- analogies: list of relevant, concrete comparisons

Focus on making complex ideas accessible and memorable."""
)
