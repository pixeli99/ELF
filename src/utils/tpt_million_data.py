"""Canonical, deterministic utilities for thinking-response dataset v1."""

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict

CLASSIFIER_VERSION = "tpt_million_v1_rules_20260811"
GENERAL_TASKS = {"general_text_v1"}
AUX_TASKS = {"math_primary", "code_primary", "mcq", "tool_or_structured", "multimodal_dependent"}
ALL_TASKS = GENERAL_TASKS | AUX_TASKS | {"ambiguous_mixed", "no_reasoning", "malformed"}
CONTROL_RE = re.compile(
    r"<\|[^>]+\|>|</?think>|</?tool(?:_call|_response)?>|</s>|<pad>|<unk>|<extra_id_\d+>",
    re.I,
)


def normalize_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n").split())


def stable_hash(*values):
    payload = json.dumps([normalize_text(str(v)) for v in values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_record_hash(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def stable_source_identity(dataset_name, dataset_revision, config, split, raw_record_sha256,
                           relative_source_file=None, raw_row_index=None,
                           byte_window_start=None, record_offset_within_window=None):
    position = {"relative_source_file": relative_source_file, "raw_row_index": raw_row_index,
                "byte_window_start": byte_window_start,
                "record_offset_within_window": record_offset_within_window}
    if raw_row_index is None and (relative_source_file is None or byte_window_start is None or record_offset_within_window is None):
        raise ValueError("source identity requires raw_row_index or file/window/record offset")
    payload = {"dataset_name": dataset_name, "dataset_revision": dataset_revision, "config": config,
               "split": split, "position": position, "raw_record_sha256": raw_record_sha256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _messages_prompt(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages_missing_or_empty")
    parts = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError("invalid_message")
        parts.append(f"{message['role']}: {message['content']}")
    if not any(m.get("role") == "user" for m in messages):
        raise ValueError("no_user_message")
    return "\n\n".join(parts)


def parse_source(kind, row):
    if kind == "nemotron":
        if all(isinstance(row.get(k), str) for k in ("prompt", "thinking", "final_response")):
            return dict(prompt=row["prompt"], thinking=row["thinking"], response=row["final_response"],
                        source_task=row.get("source") or row.get("category") or "",
                        teacher_model=row.get("generator") or "", license="cc-by-4.0",
                        provenance_type="documented_forward")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
            raise ValueError("messages_not_exact_user_assistant")
        prompt = _messages_prompt(messages[:1])
        assistant = messages[1].get("content")
        if row.get("thinking") is not True or not isinstance(assistant, str):
            raise ValueError("not_documented_thinking")
        opens, closes = assistant.count("<think>"), assistant.count("</think>")
        if opens != 1 or closes != 1 or assistant.index("<think>") > assistant.index("</think>"):
            raise ValueError("invalid_think_boundary")
        thinking, response = assistant.split("<think>", 1)[1].split("</think>", 1)
        return dict(prompt=prompt, thinking=thinking, response=response, source_task=row.get("source") or row.get("category") or "",
                    teacher_model=row.get("generator") or "", license="cc-by-4.0", provenance_type="documented_forward")
    if kind == "dolphin":
        return dict(prompt=_messages_prompt(row.get("messages")), thinking=row.get("thinking", row.get("reasoning")),
                    response=row.get("response", row.get("answer")), source_task=row.get("source_task", ""),
                    teacher_model=row.get("model", ""), license=row.get("license", "unknown"), provenance_type="forward_like_public")
    if kind == "gta":
        return dict(prompt=row.get("prompt", row.get("question")), thinking=row.get("thinking", row.get("model_reasoning")),
                    response=row.get("response", row.get("model_answer")), source_task=row.get("task") or row.get("question_source") or "",
                    teacher_model=row.get("teacher_model", row.get("model_name", "")),
                    license=row.get("license", row.get("question_license", "unknown")), provenance_type="forward_like_public")
    raise ValueError(f"unknown source kind: {kind}")


def validate_text(name, value, reject_controls=True):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_missing_or_empty")
    if "\x00" in value or (reject_controls and CONTROL_RE.search(value)):
        raise ValueError(f"{name}_contains_control_token")
    return value


def token_ids(tokenizer, text, require_exactly_one_terminal_eos=True):
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    eos = int(tokenizer.eos_token_id)
    if not ids or ids[-1] != eos:
        raise ValueError("tokenization_must_append_terminal_eos")
    if require_exactly_one_terminal_eos and ids.count(eos) != 1:
        raise ValueError("tokenization_must_append_exactly_one_terminal_eos")
    return ids


def classify_task_with_reason(prompt, thinking="", response="", source_task=""):
    """Classify the primary objective rather than incidental syntax."""
    text = normalize_text(f"{source_task} {prompt}").lower()
    signals = []
    if re.search(r"\b(image|photo|figure|diagram|chart)\b", text) and re.search(r"\b(shown|above|attached|pictured|visual)\b", text):
        signals.append(("multimodal_dependent", "requires unavailable visual input"))
    if re.search(r"\b(call (?:the )?(?:tool|function)|use (?:the )?(?:tool|api)|function[_ ]call|return (?:valid )?json schema)\b", text):
        signals.append(("tool_or_structured", "primary objective is tool/API structured output"))
    if re.search(r"\b(multiple[- ]choice|choose the (?:correct|best) (?:answer|option)|select one option|mmlu)\b", text) or len(re.findall(r"(?:^|\n)\s*[A-D][\).:]\s+", prompt)) >= 3:
        signals.append(("mcq", "explicit multiple-choice objective"))
    if re.search(r"\b(write|implement|debug|fix|refactor|compile)\b.{0,40}\b(code|program|function|class|script|query)\b", text) or re.search(r"\b(code|program|function|class|script|sql query)\b.{0,40}\b(write|implement|debug|fix|refactor)\b", text):
        signals.append(("code_primary", "primary objective is producing or repairing code"))
    if re.search(r"\b(calculate|compute|solve|prove|derive|factor|integrate|differentiate)\b", text) and re.search(r"\b(equation|theorem|integral|derivative|probability|geometry|algebra|mathematics|polynomial|matrix)\b", text):
        signals.append(("math_primary", "primary objective is mathematical derivation or calculation"))
    kinds = {item[0] for item in signals}
    if len(kinds) > 1:
        return "ambiguous_mixed", "multiple primary-objective signals: " + ",".join(sorted(kinds))
    if signals:
        return signals[0]
    if thinking and len(normalize_text(thinking).split()) < 2:
        return "no_reasoning", "thinking contains fewer than two normalized words"
    return "general_text_v1", "natural-language question, explanation, writing, summary, or conceptual task"


def classify_task(prompt, source_task=""):
    return classify_task_with_reason(prompt, source_task=source_task)[0]


def canonicalize(kind, row, tokenizer, source_meta, reuse_pair_counts=False):
    parsed = parse_source(kind, row)
    prompt = validate_text("prompt", parsed["prompt"], reject_controls=False)
    thinking = validate_text("thinking", parsed["thinking"])
    response = validate_text("response", parsed["response"])
    # Prompt length is auxiliary. Preserve and count embedded literal special tokens;
    # the exactly-one-EOS invariant applies only to thinking/response pair fields.
    prompt_n = len(token_ids(tokenizer, prompt, require_exactly_one_terminal_eos=False))
    if reuse_pair_counts:
        thinking_n = int(row["thinking_tokens"] if "thinking_tokens" in row else row["thinking_t5_tokens"])
        response_n = int(row["response_tokens"] if "response_tokens" in row else row["response_t5_tokens"])
        stored_pair = int(row.get("total_tokens", row.get("pair_tokens", row.get("pair_t5_tokens"))))
        if thinking_n <= 0 or response_n <= 0 or thinking_n + response_n != stored_pair:
            raise ValueError("reused_token_counts_invalid")
    else:
        thinking_n, response_n = len(token_ids(tokenizer, thinking)), len(token_ids(tokenizer, response))
    pair_n = thinking_n + response_n
    task, classification_reason = classify_task_with_reason(
        prompt, thinking=thinking, response=response, source_task=parsed["source_task"]
    )
    prompt_hash, pair_hash = stable_hash(prompt), stable_hash(thinking, response)
    source_row_id = str(row.get("source_id", row.get("source_row_id", row.get("raw_row_index", row.get("row_index", row.get("sample_id", ""))))))
    sample_id = str(row.get("source_id") or f"{source_meta['short_name']}:{source_row_id}")
    pair_eligible = pair_n <= 1024
    result = {
        "sample_id": sample_id, "source": source_meta["dataset"], "source_dataset": source_meta["dataset"],
        "source_revision": source_meta["revision"], "source_split": source_meta.get("split", "train"),
        "source_config": str(row.get("config", source_meta.get("config", ""))),
        "source_file": str(row.get("relative_source_file", "")),
        "source_row_index": row.get("raw_row_index", row.get("row_index", source_row_id)),
        "source_row_id": source_row_id, "raw_sha256": str(row.get("raw_record_sha256", source_record_hash(row))),
        "prompt": prompt, "thinking": thinking, "response": response,
        "source_task": str(parsed["source_task"]), "task_class": task, "task_type": task,
        "classification_reason": classification_reason, "classifier_version": CLASSIFIER_VERSION, "language": "en",
        "license": str(parsed["license"]), "teacher_model": str(parsed["teacher_model"]),
        "provenance_type": parsed["provenance_type"], "prompt_tokens": prompt_n, "prompt_t5_tokens": prompt_n,
        "thinking_tokens": thinking_n, "thinking_t5_tokens": thinking_n,
        "response_tokens": response_n, "response_t5_tokens": response_n,
        "pair_tokens": pair_n, "pair_t5_tokens": pair_n, "full_tokens": prompt_n + pair_n,
        "full_t5_tokens": prompt_n + pair_n, "canonical_prompt_hash": prompt_hash,
        "pair_hash": pair_hash, "canonical_pair_hash": pair_hash, "source_cluster_id": prompt_hash,
        "source_record_hash": source_record_hash(row), "eligible": pair_eligible,
        "reject_reason": None if pair_eligible else "pair_over_1024", "cluster_id": prompt_hash,
    }
    return result


def view_name(task):
    if task in GENERAL_TASKS: return "general_text_v1"
    if task in AUX_TASKS: return "auxiliary_specialized"
    return "other"


def assign_cluster_splits(rows, train_n, validation_n, test_n, seed=42):
    by_cluster = defaultdict(list)
    for row in rows: by_cluster[row["cluster_id"]].append(row)
    ordered = sorted(by_cluster, key=lambda key: stable_hash(seed, key))
    selected, counts = {"train": [], "validation": [], "test": []}, {"train": train_n, "validation": validation_n, "test": test_n}
    seen_pairs = set()
    candidates = []
    for cluster in ordered:
        group = sorted(by_cluster[cluster], key=lambda r: (r["canonical_pair_hash"], r["sample_id"]))
        unique = [r for r in group if not (r["canonical_pair_hash"] in seen_pairs or seen_pairs.add(r["canonical_pair_hash"]))]
        if unique: candidates.append(unique[0])
    for split in ("validation", "test", "train"):
        selected[split] = candidates[:counts[split]]; candidates = candidates[counts[split]:]
    if any(len(selected[s]) != counts[s] for s in counts):
        raise ValueError("insufficient unique prompt clusters for requested split sizes")
    return selected


def assert_split_disjoint(splits):
    for field in ("canonical_prompt_hash", "canonical_pair_hash", "cluster_id"):
        sets = {name: {r[field] for r in rows} for name, rows in splits.items()}
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if sets[a] & sets[b]: raise AssertionError(f"{field} overlap: {a}/{b}")
