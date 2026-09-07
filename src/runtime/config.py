from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Small, explicit configuration shared by the runtime components."""

    sink_tokens: int = 4
    recent_tokens: int = 256
    kv_chunk_tokens: int = 64
    predicted_mass_threshold: float = 0.95
    marginal_mass_threshold: float = 0.01
    marginal_patience: int = 2
    minimum_old_chunks: int = 2
    expert_cache_slots: int = 64
    kv_cache_slots: int = 512

    def __post_init__(self) -> None:
        positive = {
            "sink_tokens": self.sink_tokens,
            "recent_tokens": self.recent_tokens,
            "kv_chunk_tokens": self.kv_chunk_tokens,
            "marginal_patience": self.marginal_patience,
            "minimum_old_chunks": self.minimum_old_chunks,
            "expert_cache_slots": self.expert_cache_slots,
            "kv_cache_slots": self.kv_cache_slots,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("predicted_mass_threshold", self.predicted_mass_threshold),
            ("marginal_mass_threshold", self.marginal_mass_threshold),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
