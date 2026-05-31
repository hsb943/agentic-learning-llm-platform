import json
import time
import random
import hashlib
import uuid
import threading
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict
from json import JSONDecoder, JSONDecodeError
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# =========================
# CONFIG
# =========================

MODEL_NAME = "gpt-4o-mini"
NUM_WORKERS = 4
MAX_EMPTY_PARSE_RETRIES = 8
ENFORCE_QUERY_DIVERSITY = True

ALLOWED_NODES = {
    "generate_checkpoints",
    "generate_query",
    "generate_question",
    "verify_answer",
    "teach_concept",
}

TASK_SPECS = [
    {
        "node": "generate_checkpoints",
        "description": "Generate foundation-to-mastery learning checkpoints from the context.",
        "target": 16,
        "temperature": 0.7,
        "input_contract": """input must include:
{"topic":"string","goals":["string","string"]}""",
        "schema_contract": """output must be:
{"checkpoints":[
  {"description":"string","criteria":["string","string"],"verification":"string"}
]}""",
    },
    {
        "node": "generate_query",
        "description": "Generate retrieval queries from checkpoint-style requirements.",
        "target": 16,
        "temperature": 0.7,
        "input_contract": """input must include:
{"topic":"string","checkpoint_descriptions":["string","string"]}""",
        "schema_contract": """output must be:
{"search_queries":["string","string","string"]}""",
    },
    {
        "node": "generate_question",
        "description": "Generate one verification question for a checkpoint.",
        "target": 20,
        "temperature": 0.7,
        "input_contract": """input must include:
{"checkpoint_description":"string","criteria":["string","string"],"verification_method":"string"}""",
        "schema_contract": """output must be:
{"question":"string"}""",
    },
    {
        "node": "verify_answer",
        "description": "Evaluate student answers against checkpoint criteria and context.",
        "target": 48,
        "temperature": 0.4,
        "input_contract": """input must include:
{"question":"string","student_answer":"string","context":"string","criteria":["string","string"]}""",
        "schema_contract": """output must be:
{"understanding_level":0.0_to_1.0,"feedback":"string","suggestions":["string"],"context_alignment":true_or_false}""",
    },
    {
        "node": "teach_concept",
        "description": "Generate Feynman-style teaching interventions for weak understanding.",
        "target": 20,
        "temperature": 0.7,
        "input_contract": """input must include:
{"concept":"string","verification_feedback":"string","context":"string"}""",
        "schema_contract": """output must be:
{"simplified_explanation":"string","key_concepts":["string"],"analogies":["string"]}""",
    },
]

# =========================
# INIT CLIENT
# =========================

client = OpenAI()

# =========================
# THREAD SAFETY
# =========================

write_lock = threading.Lock()
hash_lock = threading.Lock()

# =========================
# UTILITIES
# =========================


def now_utc():
    return datetime.now(timezone.utc)


def clean_response_text(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 3:
            raw = parts[1]
            raw_lines = raw.splitlines()
            if raw_lines and raw_lines[0].strip().lower() in {"json", "jsonl"}:
                raw = "\n".join(raw_lines[1:])
    return raw.strip()


def parse_jsonl(raw: str) -> List[Dict]:
    required_keys = ("node", "instruction", "input", "output")
    raw = raw.strip()
    if not raw:
        return []

    # 1) Try full JSON response first (single object or array).
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if all(k in parsed for k in required_keys):
                return [parsed]
        elif isinstance(parsed, list):
            samples = []
            for obj in parsed:
                if isinstance(obj, dict) and all(k in obj for k in required_keys):
                    samples.append(obj)
            if samples:
                return samples
    except Exception:
        pass

    # 2) Fall back to strict JSONL line-by-line parsing.
    samples = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if all(k in obj for k in required_keys):
                samples.append(obj)
        except Exception:
            continue
    if samples:
        return samples

    # 3) Fall back to concatenated JSON objects with no newlines.
    decoder = JSONDecoder()
    idx = 0
    n = len(raw)
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end_idx = decoder.raw_decode(raw, idx)
        except JSONDecodeError:
            break
        if isinstance(obj, dict) and all(k in obj for k in required_keys):
            samples.append(obj)
        idx = end_idx

    return samples


def hash_sample(sample):
    normalized = json.dumps(sample, sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()


def normalize_topic_name(topic: str) -> str:
    topic = topic.strip()
    topic = topic.replace(" ", "_")
    topic = re.sub(r"[^a-zA-Z0-9_]", "", topic)
    return topic


# =========================
# VALIDATION
# =========================


def is_non_empty_string(value):
    return isinstance(value, str) and value.strip() != ""


def has_required_keys(data, required_keys):
    return all(k in data and is_non_empty_string(data.get(k)) for k in required_keys)


def has_non_empty_string_list(data, key):
    values = data.get(key)
    return isinstance(values, list) and values and all(is_non_empty_string(v) for v in values)


def normalize_query_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_jaccard(a, b):
    set_a = set(normalize_query_text(a).split())
    set_b = set(normalize_query_text(b).split())
    if not set_a or not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def validate_query_diversity(queries, max_similarity=0.8):
    normalized = [normalize_query_text(q) for q in queries if is_non_empty_string(q)]
    if len(normalized) < 2:
        return False
    if len(set(normalized)) != len(normalized):
        return False
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            if token_jaccard(queries[i], queries[j]) >= max_similarity:
                return False
    return True


def validate_node_input(node, input_data):
    if not isinstance(input_data, dict):
        return False

    if node == "generate_checkpoints":
        return has_required_keys(input_data, ("topic",)) and has_non_empty_string_list(input_data, "goals")

    if node == "generate_query":
        return has_required_keys(input_data, ("topic",)) and has_non_empty_string_list(input_data, "checkpoint_descriptions")

    if node == "generate_question":
        return (
            has_required_keys(input_data, ("checkpoint_description", "verification_method"))
            and has_non_empty_string_list(input_data, "criteria")
        )

    if node == "verify_answer":
        return (
            has_required_keys(input_data, ("question", "student_answer", "context"))
            and has_non_empty_string_list(input_data, "criteria")
        )

    if node == "teach_concept":
        return has_required_keys(input_data, ("concept", "verification_feedback", "context"))

    return False


def validate_node_output(node, output):
    if not isinstance(output, dict):
        return False

    if node == "generate_checkpoints":
        checkpoints = output.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            return False
        for cp in checkpoints:
            if not isinstance(cp, dict):
                return False
            if not is_non_empty_string(cp.get("description")):
                return False
            criteria = cp.get("criteria")
            if not isinstance(criteria, list) or not criteria:
                return False
            if not all(is_non_empty_string(c) for c in criteria):
                return False
            if not is_non_empty_string(cp.get("verification")):
                return False
        return True

    if node == "generate_query":
        queries = output.get("search_queries")
        if not isinstance(queries, list) or len(queries) < 2:
            return False
        if not all(is_non_empty_string(q) for q in queries):
            return False
        if ENFORCE_QUERY_DIVERSITY and not validate_query_diversity(queries):
            return False
        return True

    if node == "generate_question":
        question = output.get("question")
        return is_non_empty_string(question) and question.strip().endswith("?")

    if node == "verify_answer":
        if not isinstance(output.get("understanding_level"), (float, int)):
            return False
        if not (0 <= float(output["understanding_level"]) <= 1):
            return False
        if not isinstance(output.get("context_alignment"), bool):
            return False
        if not is_non_empty_string(output.get("feedback")):
            return False
        suggestions = output.get("suggestions")
        if not isinstance(suggestions, list) or not suggestions:
            return False
        if not all(is_non_empty_string(s) for s in suggestions):
            return False
        return True

    if node == "teach_concept":
        if not is_non_empty_string(output.get("simplified_explanation")):
            return False
        key_concepts = output.get("key_concepts")
        analogies = output.get("analogies")
        if not isinstance(key_concepts, list) or len(key_concepts) < 2:
            return False
        if not isinstance(analogies, list) or len(analogies) < 1:
            return False
        if not all(is_non_empty_string(x) for x in key_concepts):
            return False
        if not all(is_non_empty_string(x) for x in analogies):
            return False
        return True

    return False


def validate_sample(sample):
    if not all(k in sample for k in ("node", "instruction", "input", "output")):
        return False
    if sample["node"] not in ALLOWED_NODES:
        return False
    if not is_non_empty_string(sample["instruction"]):
        return False
    if not isinstance(sample["input"], dict):
        return False
    if not isinstance(sample["output"], dict):
        return False
    if not validate_node_input(sample["node"], sample["input"]):
        return False
    if not validate_node_output(sample["node"], sample["output"]):
        return False
    return True


# =========================
# PROMPT FACTORY
# =========================


def build_prompt(notes, task_spec, count):
    random_seed = random.randint(0, 1_000_000)
    node = task_spec["node"]
    task_description = task_spec["description"]
    input_contract = task_spec["input_contract"]
    schema_contract = task_spec["schema_contract"]

    return f"""
You are generating HIGH-QUALITY instruction tuning data for a LangGraph tutoring agent.

Random seed: {random_seed}

CONTEXT:
<<<
{notes}
>>>

Generate EXACTLY {count} objects.
Return ONLY JSONL.
Each JSON object must be ONE LINE.
No markdown.
No code fences.
No array wrapper.
No prose before or after JSONL.

Every object MUST follow this exact top-level format:
{{
  "node": "{node}",
  "instruction": "clear task instruction",
  "input": {{ ... }},
  "output": {{ ... }}
}}

Schema contract for "input":
{input_contract}

Schema contract for "output":
{schema_contract}

Task:
{task_description}

Additional requirements:
- Keep all data grounded in context.
- Use diverse values and phrasing.
- Keep each object logically valid.
- The input object MUST satisfy the input contract exactly.
- The output object MUST satisfy the output contract exactly.
- For node "generate_question", output.question MUST be an actual question ending with "?".
- For node "verify_answer", include actionable suggestions list (not empty).
"""


# =========================
# SAFE GENERATION
# =========================


def safe_generate(prompt, temperature=0.7, max_retries=6, worker_id=None):
    delay = 2

    for attempt in range(max_retries):
        try:
            print(f"[Worker-{worker_id}] Sending request (attempt {attempt+1})")

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=2800,
            )

            print(f"[Worker-{worker_id}] Response received")
            return response

        except Exception as e:
            print(f"[Worker-{worker_id}] Error: {e}")

            if attempt == max_retries - 1:
                raise

            sleep_time = delay + random.uniform(0, 1)
            print(f"[Worker-{worker_id}] Retrying in {sleep_time:.2f}s")
            time.sleep(sleep_time)
            delay *= 2


# =========================
# WORKER FUNCTION
# =========================


def worker_task(
    worker_id,
    prompt_builder,
    target_count,
    existing_hashes,
    temperature,
    topic,
    node_name,
    output_file,
    max_empty_parse_retries=MAX_EMPTY_PARSE_RETRIES,
):
    collected = 0
    empty_parse_retries = 0

    while collected < target_count:
        prompt = prompt_builder()
        response = safe_generate(prompt, temperature=temperature, worker_id=worker_id)

        raw = clean_response_text(response.choices[0].message.content)
        samples = parse_jsonl(raw)

        print(f"[Worker-{worker_id}] Parsed {len(samples)} samples for node={node_name}")

        if not samples:
            empty_parse_retries += 1
            print(f"[Worker-{worker_id}] No valid samples parsed")
            preview = raw.replace("\n", "\\n")[:400]
            print(f"[Worker-{worker_id}] Raw preview: {preview}")
            if empty_parse_retries >= max_empty_parse_retries:
                raise RuntimeError(
                    f"[Worker-{worker_id}] Max empty-parse retries reached "
                    f"({max_empty_parse_retries}). Last response was not parseable."
                )
            continue

        empty_parse_retries = 0

        for s in samples:
            if s.get("node") != node_name:
                print(f"[Worker-{worker_id}] Wrong node in sample, expected={node_name}")
                continue

            if not validate_sample(s):
                print(f"[Worker-{worker_id}] Invalid sample skipped")
                continue

            h = hash_sample(s)

            with hash_lock:
                if h in existing_hashes:
                    print(f"[Worker-{worker_id}] Duplicate skipped")
                    continue
                existing_hashes.add(h)

            s["meta"] = {
                "created_at": now_utc().isoformat(),
                "topic": topic,
                "node": node_name,
            }

            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

            collected += 1
            print(f"[Worker-{worker_id}] Saved sample ({collected}/{target_count})")

            if collected >= target_count:
                break

    print(f"[Worker-{worker_id}] Completed assigned work for node={node_name}")
    return collected


# =========================
# MAIN PIPELINE
# =========================


def generate_for_topic(topic, notes):
    safe_topic = normalize_topic_name(topic)

    run_prefix = now_utc().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:5]
    run_folder_name = f"{run_prefix}-{safe_topic}"

    base_dir = Path(__file__).parent
    run_dir = base_dir / "data" / "raw" / run_folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRun directory: {run_dir}")

    output_file = run_dir / f"samples-{safe_topic}.jsonl"
    existing_hashes = set()

    total_generated = 0

    for task in TASK_SPECS:
        node_name = task["node"]
        target = task["target"]
        temp = task["temperature"]

        print(f"\nStarting task block: node={node_name}, target={target}")

        # Distribute target exactly across workers (no off-by-one loss).
        base_target = target // NUM_WORKERS
        remainder = target % NUM_WORKERS
        worker_targets = [
            base_target + (1 if i < remainder else 0) for i in range(NUM_WORKERS)
        ]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i in range(NUM_WORKERS):
                worker_target = worker_targets[i]
                if worker_target <= 0:
                    continue
                prompt_builder = lambda t=task, c=worker_target: build_prompt(notes, t, c)
                futures.append(
                    executor.submit(
                        worker_task,
                        i + 1,
                        prompt_builder,
                        worker_target,
                        existing_hashes,
                        temp,
                        topic,
                        node_name,
                        output_file,
                    )
                )

            for future in as_completed(futures):
                total_generated += future.result()

    print(f"\nCompleted topic: {topic} ({total_generated} new samples)")


# =========================
# ENTRY POINT
# =========================


if __name__ == "__main__":
    notes_dir = Path(__file__).parent / "notes"

    topics = [
        {
            "topic": "Transformer Training Dynamics and Gradient Flow",
            "notes_file": "notes_transformer_training_dynamics_and_gradient_flow.md",
        },
    ]

    for t in topics:
        print(f"\nStarting topic: {t['topic']}")

        notes_path = notes_dir / t["notes_file"]

        with open(notes_path, "r", encoding="utf-8") as f:
            notes = f.read()

        generate_for_topic(t["topic"], notes)

    print("\nALL TOPICS COMPLETED")
