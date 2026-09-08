from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open

from src.runtime.memory_queue import ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import OffloadRuntime, PrefetchRequest


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
        by_shard: dict[str, list[tuple[tuple[int, int], str, str]]] = {}
        for identity in identities:
            for field, name in self.names(*identity).items():
                by_shard.setdefault(self.weight_map[name], []).append((identity, field, name))
        with self._lock:
            partial: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
            for shard, items in by_shard.items():
                with safe_open(self.model_path / shard, framework="pt", device="cpu") as handle:
                    for identity, field, name in items:
                        if identity in self._cache:
                            continue
                        tensor = handle.get_tensor(name).contiguous()
                        if self.pin_memory and torch.cuda.is_available():
                            tensor = tensor.pin_memory()
                        partial.setdefault(identity, {})[field] = tensor
            for identity, tensors in partial.items():
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
    requests = []
    for row, request_id in zip(probabilities, request_ids):
        values, experts = row.topk(min(top_k, row.numel()))
        for probability, expert in zip(values.tolist(), experts.tolist()):
            key = registry.ensure(layer, expert)
            requests.append(
                PrefetchRequest(key, request_id, float(probability), deadline, miss_cost_ms)
            )
            queued.append((key, request_id))
    runtime.prefetch_many(requests)
    return queued


class OffloadedExpertExecutor:
    """Exact Top-K Qwen3 MoE calculation backed by demand-loaded experts."""

    def __init__(
        self,
        runtime: OffloadRuntime,
        registry: ExpertRegistry,
        top_k: int = 8,
        miss_cost_ms: float = 0.5,
        vectorized_token_limit: int = 8,
        fused_moe: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        self.runtime = runtime
        self.registry = registry
        self.top_k = top_k
        self.miss_cost_ms = miss_cost_ms
        self.vectorized_token_limit = vectorized_token_limit
        self.fused_moe = fused_moe

    def _load(
        self, selected: torch.Tensor, layer: int, request_ids: list[str]
    ) -> dict[int, tuple[ResourceKey, ExpertWeights]]:
        loaded = {}
        for expert in selected.unique().tolist():
            token_indices, _ = torch.where(selected == expert)
            key = self.registry.ensure(layer, expert)
            weights = self.runtime.demand(
                key,
                consumer="demand:" + ",".join(request_ids[index] for index in token_indices),
                miss_cost_ms=self.miss_cost_ms,
            )
            loaded[expert] = (key, weights)
        return loaded

    def _vectorized(
        self,
        hidden_states: torch.Tensor,
        selected: torch.Tensor,
        routing: torch.Tensor,
        loaded: dict[int, tuple[ResourceKey, ExpertWeights]],
    ) -> torch.Tensor:
        routes = selected.reshape(-1).tolist()
        inputs = (
            hidden_states[:, None]
            .expand(-1, selected.shape[1], -1)
            .reshape(len(routes), hidden_states.shape[-1])
        )
        gate = torch.stack([loaded[expert][1].gate for expert in routes])
        up = torch.stack([loaded[expert][1].up for expert in routes])
        down = torch.stack([loaded[expert][1].down for expert in routes])
        gate_values = torch.bmm(gate, inputs.unsqueeze(-1)).squeeze(-1)
        up_values = torch.bmm(up, inputs.unsqueeze(-1)).squeeze(-1)
        activated = F.silu(gate_values) * up_values
        outputs = torch.bmm(down, activated.unsqueeze(-1)).squeeze(-1)
        outputs = outputs.reshape(*selected.shape, hidden_states.shape[-1])
        return (outputs * routing[..., None]).sum(dim=1).to(hidden_states.dtype)

    def _fused(
        self,
        hidden_states: torch.Tensor,
        selected: torch.Tensor,
        routing: torch.Tensor,
        loaded: dict[int, tuple[ResourceKey, ExpertWeights]],
    ) -> torch.Tensor:
        backend = self.runtime.worker.backend
        w1, w2 = backend.packed_expert_weights()
        physical_ids = torch.empty_like(selected)
        for expert, (key, _) in loaded.items():
            physical_ids.masked_fill_(selected == expert, backend.expert_slot(key))
        return self.fused_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=routing,
            topk_ids=physical_ids,
            inplace=False,
            activation="silu",
        )

    def _grouped_streaming(
        self,
        hidden_states: torch.Tensor,
        selected: torch.Tensor,
        routing: torch.Tensor,
        layer: int,
        request_ids: list[str],
    ) -> torch.Tensor:
        """Bound residency during prefill by computing one expert at a time."""
        result = torch.zeros_like(hidden_states)
        for expert in selected.unique().tolist():
            token_indices, route_indices = torch.where(selected == expert)
            key = self.registry.ensure(layer, expert)
            weights = self.runtime.demand(
                key,
                consumer="demand:"
                + ",".join(request_ids[index] for index in token_indices),
                miss_cost_ms=self.miss_cost_ms,
            )
            inputs = hidden_states[token_indices]
            activated = F.silu(F.linear(inputs, weights.gate)) * F.linear(inputs, weights.up)
            outputs = F.linear(activated, weights.down)
            outputs *= routing[token_indices, route_indices, None]
            result.index_add_(0, token_indices, outputs.to(result.dtype))
            self.runtime.release(key)
        return result

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
        if len(hidden_states) <= self.vectorized_token_limit:
            loaded = self._load(selected, layer, request_ids)
            backend = self.runtime.worker.backend
            packed = hasattr(backend, "packed_expert_weights") and hasattr(
                backend, "expert_slot"
            )
            if self.fused_moe is not None and packed and hidden_states.is_cuda:
                result = self._fused(hidden_states, selected, routing, loaded)
            else:
                result = self._vectorized(hidden_states, selected, routing, loaded)
            for key, _ in loaded.values():
                self.runtime.release(key)
        else:
            result = self._grouped_streaming(
                hidden_states,
                selected,
                routing,
                layer,
                request_ids,
            )
        return result


def optional_vllm_fused_moe(mode: str, backend) -> Callable[..., torch.Tensor] | None:
    """Resolve the optional production kernel without coupling the core to vLLM."""
    if mode not in {"auto", "torch", "vllm"}:
        raise ValueError("MoE backend must be auto, torch, or vllm")
    if mode == "torch":
        return None
    if not hasattr(backend, "packed_expert_weights"):
        if mode == "vllm":
            raise RuntimeError("vLLM fused MoE requires a packed expert backend")
        return None
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    except Exception:  # noqa: BLE001 - auto mode is an optional acceleration path
        if mode == "vllm":
            raise
        return None
    return fused_experts
