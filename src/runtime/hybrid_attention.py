from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class HybridStopController:
    """Online stopping rule for draft-ranked old KV chunks."""

    predicted_mass_threshold: float = 0.95
    marginal_mass_threshold: float = 0.01
    marginal_patience: int = 2
    minimum_chunks: int = 2
    retrieved_chunks: int = 0
    predicted_mass: float = 0.0
    low_marginal_streak: int = 0

    def observe(self, predicted_mass: float, target_marginal_mass: float) -> bool:
        if not 0 <= predicted_mass <= 1 or not 0 <= target_marginal_mass <= 1:
            raise ValueError("mass values must be in [0, 1]")
        self.retrieved_chunks += 1
        self.predicted_mass = min(1.0, self.predicted_mass + predicted_mass)
        if target_marginal_mass <= self.marginal_mass_threshold:
            self.low_marginal_streak += 1
        else:
            self.low_marginal_streak = 0
        return (
            self.retrieved_chunks >= self.minimum_chunks
            and self.predicted_mass >= self.predicted_mass_threshold
            and self.low_marginal_streak >= self.marginal_patience
        )


def marginal_partition_mass(previous_lse: torch.Tensor, chunk_lse: torch.Tensor) -> torch.Tensor:
    """Fraction of the updated softmax partition contributed by a new chunk."""
    combined = torch.logaddexp(previous_lse, chunk_lse)
    return torch.exp(chunk_lse - combined)


def gqa_logits(query: torch.Tensor, key: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    if query.ndim != 2 or key.ndim != 3:
        raise ValueError("query must be (heads, dim), key must be (tokens, kv_heads, dim)")
    heads, dimension = query.shape
    kv_heads = key.shape[1]
    if key.shape[-1] != dimension or heads % kv_heads:
        raise ValueError("incompatible query and KV head shapes")
    groups = heads // kv_heads
    grouped_query = query.float().reshape(kv_heads, groups, dimension)
    logits = torch.bmm(grouped_query, key.float().permute(1, 2, 0))
    return logits.reshape(heads, len(key)) * (scale or dimension**-0.5)


def chunk_logsumexp(
    query: torch.Tensor, key: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Per-head log partition for one GQA KV chunk and one query token."""
    return torch.logsumexp(gqa_logits(query, key, scale), dim=-1)


def mean_target_marginal(previous_lse: torch.Tensor, chunk_lse: torch.Tensor) -> float:
    if torch.isneginf(previous_lse).all():
        return 1.0
    return float(marginal_partition_mass(previous_lse, chunk_lse).mean())


def update_partition(previous_lse: torch.Tensor, chunk_lse: torch.Tensor) -> torch.Tensor:
    if previous_lse.shape != chunk_lse.shape:
        raise ValueError("partition tensors must have the same shape")
    return torch.logaddexp(previous_lse, chunk_lse)


def sequence_target_marginals(
    previous_lse: torch.Tensor, chunk_lses: list[torch.Tensor]
) -> tuple[torch.Tensor, list[float]]:
    """Update an ordered partition while synchronizing all marginal scalars once."""
    if not chunk_lses:
        return previous_lse, []
    marginals = []
    partition = previous_lse
    for chunk_lse in chunk_lses:
        if partition.shape != chunk_lse.shape:
            raise ValueError("partition tensors must have the same shape")
        combined = torch.logaddexp(partition, chunk_lse)
        marginals.append(torch.exp(chunk_lse - combined).mean())
        partition = combined
    return partition, torch.stack(marginals).tolist()


def empty_partition(heads: int, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.full((heads,), -math.inf, dtype=torch.float32, device=device)


def attention_output(
    query: torch.Tensor,
    chunks: list[tuple[torch.Tensor, torch.Tensor]],
    scale: float | None = None,
) -> torch.Tensor:
    """Reference GQA attention over an explicitly selected set of KV chunks."""
    if not chunks:
        raise ValueError("at least one KV chunk is required")
    heads, dimension = query.shape
    keys = torch.cat([key for key, _ in chunks])
    values = torch.cat([value for _, value in chunks])
    if keys.shape != values.shape:
        raise ValueError("incompatible query, key, and value shapes")
    kv_heads = keys.shape[1]
    if heads % kv_heads:
        raise ValueError("incompatible query, key, and value shapes")
    groups = heads // kv_heads
    logits = gqa_logits(query, keys, scale).reshape(kv_heads, groups, len(keys))
    weights = torch.softmax(logits, dim=-1).to(values.dtype)
    return torch.bmm(weights, values.permute(1, 0, 2)).reshape(heads, dimension)


def attention_output_from_logits(
    logits: list[torch.Tensor],
    values: list[torch.Tensor],
) -> torch.Tensor:
    """Finish GQA from per-chunk logits already evaluated for marginal stopping."""
    if not logits or len(logits) != len(values):
        raise ValueError("aligned non-empty logits and values are required")
    heads = logits[0].shape[0]
    dimension = values[0].shape[-1]
    kv_heads = values[0].shape[1]
    if heads % kv_heads:
        raise ValueError("incompatible logits and value head shapes")
    for chunk_logits, chunk_values in zip(logits, values):
        if (
            chunk_logits.ndim != 2
            or chunk_values.ndim != 3
            or chunk_logits.shape[0] != heads
            or chunk_logits.shape[1] != len(chunk_values)
            or chunk_values.shape[1:] != (kv_heads, dimension)
        ):
            raise ValueError("incompatible per-chunk logits and value shapes")
    combined_logits = torch.cat(logits, dim=-1)
    combined_values = torch.cat(values)
    groups = heads // kv_heads
    weights = torch.softmax(
        combined_logits.reshape(kv_heads, groups, len(combined_values)),
        dim=-1,
    ).to(combined_values.dtype)
    return torch.bmm(weights, combined_values.permute(1, 0, 2)).reshape(heads, dimension)
