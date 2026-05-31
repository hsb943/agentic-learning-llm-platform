import argparse
import json
import re
from collections import Counter


ALLOWED_NODES = {
    "generate_checkpoints",
    "generate_query",
    "generate_question",
    "verify_answer",
    "teach_concept",
}


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


def validate_node_output(node, output, enforce_query_diversity=True):
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
        if enforce_query_diversity and not validate_query_diversity(queries):
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


def validate_sample(sample, enforce_query_diversity=True):
    if not all(k in sample for k in ("node", "instruction", "input", "output")):
        return False, "missing_keys"
    node = sample.get("node")
    if node not in ALLOWED_NODES:
        return False, "invalid_node"
    if not is_non_empty_string(sample.get("instruction")):
        return False, "bad_instruction"
    if not isinstance(sample.get("input"), dict):
        return False, "bad_input_type"
    if not isinstance(sample.get("output"), dict):
        return False, "bad_output_type"
    if not validate_node_input(node, sample["input"]):
        return False, f"bad_input_contract:{node}"
    if not validate_node_output(node, sample["output"], enforce_query_diversity=enforce_query_diversity):
        return False, f"bad_output_contract:{node}"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Validate agent-schema dataset JSONL.")
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--no-query-diversity",
        action="store_true",
        help="Disable query diversity enforcement for generate_query node.",
    )
    args = parser.parse_args()

    enforce_query_diversity = not args.no_query_diversity
    total = 0
    valid = 0
    node_counts = Counter()
    reason_counts = Counter()

    with open(args.input, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                sample = json.loads(line)
            except Exception:
                reason_counts["invalid_json"] += 1
                continue

            ok, reason = validate_sample(sample, enforce_query_diversity=enforce_query_diversity)
            if ok:
                valid += 1
                node_counts[sample["node"]] += 1
            else:
                reason_counts[reason] += 1

    print(f"Input: {args.input}")
    print(f"Total rows: {total}")
    print(f"Valid rows: {valid}")
    print(f"Invalid rows: {total - valid}")
    print(f"Valid rate: {valid / total:.4f}" if total else "Valid rate: 0.0000")
    print("Node counts (valid rows):")
    for node in sorted(ALLOWED_NODES):
        print(f"  {node}: {node_counts[node]}")
    if reason_counts:
        print("Invalid reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
