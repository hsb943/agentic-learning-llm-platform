import json
import re
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ---------- helpers ----------
def extract_json(text: str):
    if not text or not text.strip():
        raise ValueError("Model returned empty response")

    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found. Raw output was:\n{text}")

    return json.loads(match.group())

# ---------- init ----------
vertexai.init(
    project="titanium-kiln-485804-q3",
    location="us-central1"
)

model = GenerativeModel(
    "publishers/google/models/gemini-2.5-flash"
)

PROMPT = """
You are a medical education curriculum designer creating topics for an AI-powered
checkpoint-based learning system.

CRITICAL CONSTRAINTS (must follow strictly):

1. Generate EXACTLY 30 topics.
   - Count internally before responding.
   - Do NOT stop early.
   - Do NOT generate more or fewer.

2. Each topic must be a BOUNDED KNOWLEDGE UNIT.
   A valid topic must:
   - Be fully coverable in 2–3 pages of structured notes
   - Have clear diagnostic definitions, criteria, and reasoning
   - Be suitable for generating checkpoints, exam questions,
     answer verification, and targeted teaching

3. Topic focus:
   - Diagnostic frameworks
   - Interpretation of investigations
   - Disease-level diagnostic reasoning
   - Differential diagnosis within a narrow scope

4. STRICTLY AVOID generating:
   - Very broad domains (e.g. "Diagnosing common infectious diseases")
   - Multi-system grab bags (e.g. "Pediatric emergencies")
   - Vague symptom clusters without diagnostic closure
   - Management-heavy or treatment-protocol topics
   - Non-clinical or pre-clinical basic science topics

5. Topic granularity rules:
   - One topic = one coherent diagnostic unit
   - A medical student should be able to study it in ONE focused sitting
   - If a topic would naturally split into multiple diseases,
     it is TOO BROAD and must be avoided

6. Style rules:
   - Use precise, clinical language
   - Prefer "diagnostic approach", "differential diagnosis",
     "interpretation", "classification"
   - Do NOT include subtopics or explanations
   - Each topic should stand alone

OUTPUT FORMAT (MANDATORY):
Return ONLY valid JSON in this exact structure.
No markdown. No comments. No extra text.

{
  "topics": [
    "Topic 1",
    "Topic 2",
    ...
    "Topic 30"
  ]
}

"""

response = model.generate_content(
    PROMPT,
    generation_config=GenerationConfig(
        temperature=0,
        max_output_tokens=4096
    )
)

# ---------- parse ----------
topics = extract_json(response.text)

# ---------- save ----------
with open("topics.json", "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=2)

print("Generated topics:")
for i, t in enumerate(topics["topics"], 1):
    print(f"{i}. {t}")

