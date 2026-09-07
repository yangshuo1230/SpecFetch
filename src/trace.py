from __future__ import annotations

from dataclasses import dataclass

import torch

Mass = dict[int, float]


@dataclass
class ModelTrace:
    """CPU-side observations for one model and one generated sequence."""

    kv_mass: dict[int, list[Mass]]
    hidden_states: dict[int, torch.Tensor]
    router_probabilities: dict[int, torch.Tensor]


@dataclass
class PairedTrace:
    """Aligned target/draft observations for one prompt."""

    target: ModelTrace
    draft: ModelTrace
    prompt_tokens: int
    evaluated_tokens: int


def aggregate_mass(values: list[float], block_size: int) -> Mass:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks: Mass = {}
    for index, value in enumerate(values):
        block = index // block_size
        blocks[block] = blocks.get(block, 0.0) + float(value)
    return blocks


def normalize_mass(values: Mass) -> Mass:
    total = sum(values.values())
    return {key: value / total for key, value in values.items()} if total > 0 else {}


def attention_block_mass(
    attentions: tuple[torch.Tensor | None, ...],
    block_size: int,
    token_slice: slice,
    *,
    key_limit: int | None = None,
    key_lag: int = 0,
) -> dict[int, list[Mass]]:
    """Aggregate attention over heads and KV blocks visible at prefetch time.

    ``key_limit`` keeps only a fixed prefix, as used for a draft rollout. ``key_lag``
    derives the visible prefix from each query, as used to score an h-token-ahead
    target access (``key_lag=h``). They are mutually exclusive.
    """
    if key_limit is not None and key_lag:
        raise ValueError("key_limit and key_lag are mutually exclusive")
    if key_lag < 0:
        raise ValueError("key_lag must be non-negative")
    result: dict[int, list[Mass]] = {}
    for layer, attention in enumerate(attentions):
        if attention is None:
            continue
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError(f"expected (1, heads, queries, keys), got {attention.shape}")
        per_query = attention.detach().float().sum(dim=1)[0].cpu()
        rows = []
        for query in range(*token_slice.indices(per_query.shape[0])):
            limit = key_limit if key_limit is not None else query + 1 - key_lag
            limit = max(0, min(limit, query + 1))
            mass = aggregate_mass(per_query[query, :limit].tolist(), block_size)
            rows.append(normalize_mass(mass))
        result[layer] = rows
    return result


def router_probabilities(logits: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Convert a Qwen3-MoE gate output into ``(tokens, experts)`` probabilities."""
    logits = logits.detach().float().reshape(-1, logits.shape[-1])
    if logits.shape[0] != sequence_length:
        raise ValueError(
            f"router emitted {logits.shape[0]} rows for a {sequence_length}-token sequence"
        )
    return torch.softmax(logits, dim=-1).cpu()


def map_layer(layer: int, source_layers: int, target_layers: int) -> int:
    """Map a layer to the nearest relative-depth layer in another model."""
    if source_layers <= 0 or target_layers <= 0:
        raise ValueError("layer counts must be positive")
    if source_layers == 1 or target_layers == 1:
        return 0
    return round(layer * (target_layers - 1) / (source_layers - 1))
