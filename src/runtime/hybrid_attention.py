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


def chunk_logsumexp(
    query: torch.Tensor, key: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Per-head log partition for one GQA KV chunk and one query token."""
    if query.ndim != 2 or key.ndim != 3:
        raise ValueError("query must be (heads, dim), key must be (tokens, kv_heads, dim)")
    heads, dimension = query.shape
    if key.shape[-1] != dimension or heads % key.shape[1]:
        raise ValueError("incompatible query and KV head shapes")
    repeated_key = key.repeat_interleave(heads // key.shape[1], dim=1)
    logits = torch.einsum("hd,thd->ht", query.float(), repeated_key.float())
    return torch.logsumexp(logits * (scale or dimension**-0.5), dim=-1)


def mean_target_marginal(previous_lse: torch.Tensor, chunk_lse: torch.Tensor) -> float:
    if torch.isneginf(previous_lse).all():
        return 1.0
    return float(marginal_partition_mass(previous_lse, chunk_lse).mean())


def update_partition(previous_lse: torch.Tensor, chunk_lse: torch.Tensor) -> torch.Tensor:
    if previous_lse.shape != chunk_lse.shape:
        raise ValueError("partition tensors must have the same shape")
    return torch.logaddexp(previous_lse, chunk_lse)


def empty_partition(heads: int, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.full((heads,), -math.inf, dtype=torch.float32, device=device)
