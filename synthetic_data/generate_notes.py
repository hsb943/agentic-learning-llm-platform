import json
import time
import os
import logging
from openai import OpenAI

# =========================
# CONFIG
# =========================

MODEL_NAME = "gpt-4o"
TOPIC = "Feed-Forward Networks in Transformers"
OUTPUT_FILE = "notes_feed_forward_networks_in_transformers.md"

GLOBAL_THROTTLE = 1.5
MAX_RETRIES = 3
MIN_WORD_COUNT = 1500
MAX_OUTPUT_TOKENS = 3000  # per call

# =========================
# LOGGING SETUP
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# =========================
# INIT CLIENT
# =========================

logger.info("Initializing OpenAI client...")
client = OpenAI()
logger.info("Client initialized successfully.")

# =========================
# PROMPT
# =========================

PROMPT = f"""
Write a complete, structured technical reference document on:

{TOPIC}

These notes will be used as ground-truth reasoning context for an AI system that:
- Generates learning checkpoints
- Generates analytical questions
- Verifies reasoning step-by-step
- Performs structured teaching interventions

Requirements:

1. Length:
   - Target 1800–2200 words.
   - Do not exceed 2500 words.

2. Style:
   - Mechanistically precise
   - Mathematically explicit where applicable
   - Self-contained
   - Structured and consistent
   - Optimized for reasoning evaluation (not narrative storytelling)

3. Include where relevant:
   - Explicit variable definitions
   - Matrix shape notation (e.g., X ∈ ℝ^(n × d_model))
   - Dimensional reasoning
   - Formal asymptotic complexity (e.g., O(n² d_k))
   - Failure modes and edge cases
   - Trade-offs and comparisons
   - At least 6 logically ordered derivation or reasoning steps for core mechanisms

4. Do NOT include:
   - Markdown formatting symbols (#, ##, etc.)
   - Code blocks
   - Framework-specific API details
   - Historical commentary
   - Motivational introductions

Use the following structure exactly:

Title

Assumptions and Notation
Core Concepts and Mathematical Foundations
Mechanism and Formal Derivation
Computational and Complexity Analysis
Expressivity and Theoretical Implications
Failure Modes and Edge Cases
Key Analytical Insights

Write the full document completely.
Do not summarize.
Do not stop early.
"""

# =========================
# GENERATION FUNCTION
# =========================

def generate_notes():
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Generation attempt {attempt}/{MAX_RETRIES} started...")

        try:
            full_text = ""
            messages = [{"role": "user", "content": PROMPT}]

            while True:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )

                chunk = response.choices[0].message.content or ""
                full_text += chunk.strip() + "\n\n"

                word_count = len(full_text.split())
                logger.info(f"Current word count: {word_count}")

                if word_count >= MIN_WORD_COUNT:
                    logger.info("Minimum word count satisfied.")
                    break

                logger.info("Continuing generation to reach required length...")

                messages = [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": full_text},
                    {
                        "role": "user",
                        "content": "Continue the document from where it stopped. Do not repeat previous content."
                    },
                ]

                time.sleep(GLOBAL_THROTTLE)

            return full_text.strip()

        except Exception as e:
            logger.exception(f"Error during generation attempt {attempt}: {e}")
            time.sleep(GLOBAL_THROTTLE)

    logger.error("All retries exhausted. Generation failed.")
    return None


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    logger.info("Starting note generation pipeline...")

    notes = generate_notes()

    if notes is None:
        logger.error("No valid output generated. Exiting.")
        exit(1)

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(notes)

        logger.info(f"Notes successfully saved to {OUTPUT_FILE}")

    except Exception as e:
        logger.exception(f"Failed to write output file: {e}")
        exit(1)

    logger.info("Pipeline completed successfully.")