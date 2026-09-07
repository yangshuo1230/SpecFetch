from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from src.runtime.memory_queue import ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import OffloadRuntime


@dataclass(frozen=True)
class ExpertWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    @property
    def size_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size() for tensor in (self.gate, self.up, self.down)
        )


class SafetensorExpertSource:
    """Lazy CPU source for original Qwen3-MoE expert checkpoint tensors."""

    def __init__(self, model_path: str | Path, pin_memory: bool = True) -> None:
        self.model_path = Path(model_path)
        index_path = self.model_path / "model.safetensors.index.json"
        self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        self.pin_memory = pin_memory
        self._cache: dict[tuple[int, int], ExpertWeights] = {}
        self._lock = threading.Lock()

    @staticmethod
    def names(layer: int, expert: int) -> dict[str, str]:
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        return {
            "gate": f"{prefix}.gate_proj.weight",
            "up": f"{prefix}.up_proj.weight",
            "down": f"{prefix}.down_proj.weight",
        }

    def _load_tensor(self, name: str) -> torch.Tensor:
        shard = self.model_path / self.weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name).contiguous()
        if self.pin_memory and torch.cuda.is_available():
            tensor = tensor.pin_memory()
        return tensor

    def get(self, layer: int, expert: int) -> ExpertWeights:
        identity = (layer, expert)
        with self._lock:
            cached = self._cache.get(identity)
            if cached is not None:
                return cached
            names = self.names(layer, expert)
            shards = {self.weight_map[name] for name in names.values()}
            if len(shards) == 1:
                with safe_open(
                    self.model_path / shards.pop(), framework="pt", device="cpu"
                ) as handle:
                    tensors = {
                        name: handle.get_tensor(key).contiguous() for name, key in names.items()
                    }
                if self.pin_memory and torch.cuda.is_available():
                    tensors = {name: value.pin_memory() for name, value in tensors.items()}
                weights = ExpertWeights(**tensors)
            else:
                weights = ExpertWeights(
                    **{name: self._load_tensor(key) for name, key in names.items()}
                )
            self._cache[identity] = weights
            return weights

    def preload(self, layers: range, experts: range) -> None:
        """Populate pinned CPU storage shard-by-shard before request timing."""
        identities = [(layer, expert) for layer in layers for expert in experts]
        by_shard: dict[str, list[tuple[int, int]]] = {}
        for identity in identities:
            names = self.names(*identity)
            shard = self.weight_map[names["gate"]]
            by_shard.setdefault(shard, []).append(identity)
        with self._lock:
            for shard, items in by_shard.items():
                with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
                    for identity in items:
                        if identity in self._cache:
                            continue
                        names = self.names(*identity)
                        tensors = {
                            name: handle.get_tensor(key).contiguous() for name, key in names.items()
                        }
                        if self.pin_memory and torch.cuda.is_available():
                            tensors = {name: value.pin_memory() for name, value in tensors.items()}
                        self._cache[identity] = ExpertWeights(**tensors)


def expert_key(layer: int, expert: int) -> ResourceKey:
    return ResourceKey(ResourceKind.EXPERT, layer=layer, object_id=expert)


class ExpertRegistry:
    """Register expert CPU weights exactly once with the residency manager."""

    def __init__(self, source, residency: ResidencyManager) -> None:
        self.source = source
        self.residency = residency
        self._registered: set[ResourceKey] = set()
        self._lock = threading.Lock()

    def ensure(self, layer: int, expert: int) -> ResourceKey:
        key = expert_key(layer, expert)
        with self._lock:
            if key not in self._registered:
                weights = self.source.get(layer, expert)
                self.residency.register_cpu(key, weights, weights.size_bytes)
                self._registered.add(key)
        return key

    def preload(self, layers: range, experts: range) -> None:
        """Materialize CPU expert storage before serving begins."""
        preload = getattr(self.source, "preload", None)
        if preload is not None:
            preload(layers, experts)
        for layer in layers:
            for expert in experts:
                self.ensure(layer, expert)


def enqueue_expert_predictions(
    probabilities: torch.Tensor,
    *,
    layer: int,
    request_ids: list[str],
    top_k: int,
    deadline: int,
    miss_cost_ms: float,
    registry: ExpertRegistry,
    runtime: OffloadRuntime,
) -> list[tuple[ResourceKey, str]]:
    """Merge per-request predicted routes into shared expert queue entries."""
    if probabilities.ndim != 2 or len(probabilities) != len(request_ids):
        raise ValueError("probabilities must align with request IDs")
    queued = []
    for row, request_id in zip(probabilities, request_ids):
        values, experts = row.topk(min(top_k, row.numel()))
        for probability, expert in zip(values.tolist(), experts.tolist()):
            key = registry.ensure(layer, expert)
            runtime.prefetch(
                key,
                consumer=request_id,
                probability=float(probability),
                deadline=deadline,
                miss_cost_ms=miss_cost_ms,
            )
            queued.append((key, request_id))
    return queued


class OffloadedExpertExecutor:
    """Exact Top-K Qwen3 MoE calculation backed by demand-loaded experts."""

    def __init__(
        self,
        runtime: OffloadRuntime,
        registry: ExpertRegistry,
        top_k: int = 8,
        miss_cost_ms: float = 0.5,
    ) -> None:
        self.runtime = runtime
        self.registry = registry
        self.top_k = top_k
        self.miss_cost_ms = miss_cost_ms

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        layer: int,
        request_ids: list[str],
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or router_logits.ndim != 2:
            raise ValueError("hidden states and router logits must be two-dimensional")
        if len(hidden_states) != len(request_ids) or len(hidden_states) != len(router_logits):
            raise ValueError("tokens, router rows, and request IDs must align")
        routing = torch.softmax(router_logits.float(), dim=-1)
        routing, selected = routing.topk(min(self.top_k, routing.shape[-1]), dim=-1)
        routing = (routing / routing.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
        result = torch.zeros_like(hidden_states)
        for expert in selected.unique().tolist():
            token_indices, route_indices = torch.where(selected == expert)
            key = self.registry.ensure(layer, expert)
            weights: ExpertWeights = self.runtime.demand(
                key,
                consumer="demand:" + ",".join(request_ids[index] for index in token_indices),
                miss_cost_ms=self.miss_cost_ms,
            )
            inputs = hidden_states[token_indices]
            activated = F.silu(F.linear(inputs, weights.gate)) * F.linear(inputs, weights.up)
            outputs = F.linear(activated, weights.down)
            outputs *= routing[token_indices, route_indices, None]
            result.index_add_(0, token_indices, outputs.to(result.dtype))
            self.runtime.release(key)
        return result
