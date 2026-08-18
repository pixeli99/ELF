"""Shared T5 thinking-token semantics for generation, Stage-A, Stage-B, and eval."""

import math


def thinking_token_ids(tokenizer, text, add_special_tokens=True, truncation=None):
    kwargs = {"add_special_tokens": add_special_tokens}
    if truncation is not None:
        kwargs["truncation"] = bool(truncation)
    return tokenizer(text, **kwargs)["input_ids"]


def thinking_plan_metadata(tokenizer, text, add_special_tokens=True, group_size=4):
    token_ids = thinking_token_ids(tokenizer, text, add_special_tokens=add_special_tokens)
    return {
        "token_ids": token_ids,
        "final_encoded_length": len(token_ids),
        "slot_count": math.ceil(len(token_ids) / group_size),
        "add_special_tokens": bool(add_special_tokens),
        "eos_included": getattr(tokenizer, "eos_token_id", None) in token_ids,
    }
