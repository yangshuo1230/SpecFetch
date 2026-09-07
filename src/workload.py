from __future__ import annotations

import json
from pathlib import Path

import torch


def load_prompt_texts(path: str | Path) -> list[str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = value.get("prompts") if isinstance(value, dict) else value
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("prompts must contain a string array")
    return prompts


def fixed_contexts(tokenizer, prompts: list[str], batch: int, tokens: int) -> torch.Tensor:
    """Build deterministic equal-length contexts without padding artifacts."""
    if batch <= 0 or tokens <= 0:
        raise ValueError("batch and tokens must be positive")
    rows = []
    for prompt in prompts[:batch]:
        encoded = tokenizer(prompt, add_special_tokens=False).input_ids
        if not encoded:
            raise ValueError("an empty prompt cannot form a benchmark context")
        repeats = (tokens + len(encoded) - 1) // len(encoded)
        rows.append((encoded * repeats)[:tokens])
    if len(rows) != batch:
        raise ValueError("not enough prompts for requested batch size")
    return torch.tensor(rows, dtype=torch.long)
