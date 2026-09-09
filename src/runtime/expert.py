from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from src.runtime.memory_queue import ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import DemandRequest, OffloadRuntime, PrefetchRequest


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


def pack_expert_weights(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    *,
    pin_memory: bool,
) -> ExpertWeights:
    """Store gate/up as adjacent views matching the fused GPU slot layout."""
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("expert gate and up weights must have matching layouts")
    gate_up = torch.cat((gate, up), dim=0)
    down = down.contiguous()
    if pin_memory and torch.cuda.is_available():
        gate_up = gate_up.pin_memory()
        down = down.pin_memory()
    width = gate.shape[0]
    return ExpertWeights(gate_up[:width], gate_up[width:], down)


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
            return handle.get_tensor(name).contiguous()

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
                weights = pack_expert_weights(
                    tensors["gate"],
                    tensors["up"],
                    tensors["down"],
                    pin_memory=self.pin_memory,
                )
            else:
                tensors = {name: self._load_tensor(key) for name, key in names.items()}
                weights = pack_expert_weights(
                    tensors["gate"],
                    tensors["up"],
                    tensors["down"],
                    pin_memory=self.pin_memory,
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
                        partial.setdefault(identity, {})[field] = tensor
            while partial:
                identity, tensors = partial.popitem()
                self._cache[identity] = pack_expert_weights(
                    tensors["gate"],
                    tensors["up"],
                    tensors["down"],
                    pin_memory=self.pin_memory,
                )


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
        # Registrations only grow, and release/eviction never removes the CPU
        # record. Preloaded serving therefore takes this read-only fast path
        # instead of acquiring one registry lock per predicted/actual route.
        if key in self._registered:
            return key
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
    requests, queued = expert_prediction_requests(
        probabilities,
        layer=layer,
        request_ids=request_ids,
        top_k=top_k,
        deadline=deadline,
        miss_cost_ms=miss_cost_ms,
        registry=registry,
    )
    runtime.prefetch_many(requests)
    return queued


def expert_prediction_requests(
    probabilities: torch.Tensor,
    *,
    layer: int,
    request_ids: list[str],
    top_k: int,
    deadline: int,
    miss_cost_ms: float,
    registry: ExpertRegistry,
) -> tuple[list[PrefetchRequest], list[tuple[ResourceKey, str]]]:
    """构造专家预测请求，供调用方跨层合并提交。"""
    if probabilities.ndim != 2 or len(probabilities) != len(request_ids):
        raise ValueError("probabilities must align with request IDs")
    queued = []
    requests = []
    keys: dict[int, ResourceKey] = {}
    values, experts = probabilities.topk(min(top_k, probabilities.shape[1]), dim=1)
    for row_values, row_experts, request_id in zip(values.tolist(), experts.tolist(), request_ids):
        for probability, expert in zip(row_values, row_experts):
            key = keys.get(expert)
            if key is None:
                key = registry.ensure(layer, expert)
                keys[expert] = key
            requests.append(
                PrefetchRequest(key, request_id, float(probability), deadline, miss_cost_ms)
            )
            queued.append((key, request_id))
    return requests, queued


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
        fused_topk: Callable[..., tuple[torch.Tensor, ...]] | None = None,
    ) -> None:
        self.runtime = runtime
        self.registry = registry
        self.top_k = top_k
        self.miss_cost_ms = miss_cost_ms
        self.vectorized_token_limit = vectorized_token_limit
        self.fused_moe = fused_moe
        self.fused_topk = fused_topk

    def _load(
        self,
        selected_cpu: torch.Tensor,
        layer: int,
        request_ids: list[str],
        expert_ids: list[int] | None = None,
    ) -> dict[int, tuple[ResourceKey, ExpertWeights]]:
        if selected_cpu.device.type != "cpu":
            raise ValueError("expert demand planning requires CPU route IDs")
        dependencies = {}
        for expert in expert_ids if expert_ids is not None else selected_cpu.unique().tolist():
            token_indices = torch.where(selected_cpu == expert)[0].tolist()
            key = self.registry.ensure(layer, expert)
            dependencies[expert] = (
                key,
                DemandRequest(
                    key,
                    "demand:" + ",".join(request_ids[index] for index in token_indices),
                    self.miss_cost_ms,
                ),
            )
        values = self.runtime.demand_many([request for _, request in dependencies.values()])
        return {expert: (key, values[key]) for expert, (key, _) in dependencies.items()}

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
        layer: int,
        global_num_experts: int,
        *,
        persistent_map: bool = True,
    ) -> torch.Tensor:
        backend = self.runtime.worker.backend
        w1, w2 = backend.packed_expert_weights()
        if any(key.layer != layer for key, _ in loaded.values()):
            raise ValueError("loaded expert belongs to a different layer")
        if persistent_map:
            expert_map = backend.expert_map(layer, global_num_experts)
        else:
            # A capacity-split call must hide resident experts belonging to a
            # different split or their contributions would be accumulated twice.
            expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32, device="cpu")
            logical_ids = list(loaded)
            if any(not 0 <= expert < global_num_experts for expert in logical_ids):
                raise ValueError("logical expert ID exceeds global expert count")
            if logical_ids:
                physical_ids = [backend.expert_slot(loaded[expert][0]) for expert in logical_ids]
                expert_map[torch.tensor(logical_ids)] = torch.tensor(
                    physical_ids, dtype=torch.int32
                )
            expert_map = expert_map.to(selected.device)
        return self.fused_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=routing,
            topk_ids=selected,
            inplace=False,
            activation="silu",
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

    def _grouped_batched(
        self,
        hidden_states: torch.Tensor,
        selected: torch.Tensor,
        routing: torch.Tensor,
        layer: int,
        request_ids: list[str],
        expert_ids: list[int],
        global_num_experts: int,
        selected_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Batch H2D within the slot bound, then compute each routed expert."""
        result = torch.zeros_like(hidden_states)
        capacity = self.runtime.residency.capacities[ResourceKind.EXPERT]
        for offset in range(0, len(expert_ids), capacity):
            loaded = self._load(
                selected_cpu,
                layer,
                request_ids,
                expert_ids=expert_ids[offset : offset + capacity],
            )
            backend = self.runtime.worker.backend
            packed = (
                hasattr(backend, "packed_expert_weights")
                and hasattr(backend, "expert_map")
                and hasattr(backend, "expert_slot")
            )
            if self.fused_moe is not None and packed and hidden_states.is_cuda:
                result += self._fused(
                    hidden_states,
                    selected,
                    routing,
                    loaded,
                    layer,
                    global_num_experts,
                    persistent_map=False,
                )
            else:
                for expert, (_, weights) in loaded.items():
                    token_indices, route_indices = torch.where(selected == expert)
                    inputs = hidden_states[token_indices]
                    activated = F.silu(F.linear(inputs, weights.gate)) * F.linear(
                        inputs, weights.up
                    )
                    outputs = F.linear(activated, weights.down)
                    outputs *= routing[token_indices, route_indices, None]
                    result.index_add_(0, token_indices, outputs.to(result.dtype))
            self.runtime.release_many([key for key, _ in loaded.values()])
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
        top_k = min(self.top_k, router_logits.shape[-1])
        if self.fused_topk is not None:
            routing, selected, *_ = self.fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=True,
            )
        else:
            routing = torch.softmax(router_logits.float(), dim=-1)
            routing, selected = routing.topk(top_k, dim=-1)
            routing = (routing / routing.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
        # Demand planning needs route identities on the host. Snapshot the tiny
        # Top-K matrix once per layer, then perform unique/consumer discovery on
        # CPU instead of synchronizing once for unique() and again for every
        # expert's torch.where indices.
        selected_cpu = selected.detach().to(device="cpu")
        unique_experts = selected_cpu.unique().tolist()
        capacity = self.runtime.residency.capacities[ResourceKind.EXPERT]
        backend = self.runtime.worker.backend
        fused_packed = (
            self.fused_moe is not None
            and hidden_states.is_cuda
            and hasattr(backend, "packed_expert_weights")
            and hasattr(backend, "expert_map")
            and hasattr(backend, "expert_slot")
        )
        if len(unique_experts) <= capacity and (
            len(hidden_states) <= self.vectorized_token_limit or fused_packed
        ):
            loaded = self._load(selected_cpu, layer, request_ids, expert_ids=unique_experts)
            if fused_packed:
                result = self._fused(
                    hidden_states,
                    selected,
                    routing,
                    loaded,
                    layer,
                    router_logits.shape[-1],
                )
            else:
                result = self._vectorized(hidden_states, selected, routing, loaded)
            self.runtime.release_many([key for key, _ in loaded.values()])
        else:
            result = self._grouped_batched(
                hidden_states,
                selected,
                routing,
                layer,
                request_ids,
                unique_experts,
                router_logits.shape[-1],
                selected_cpu,
            )
        return result


def optional_vllm_fused_moe(mode: str, backend) -> Callable[..., torch.Tensor] | None:
    """Resolve the optional production kernel without coupling the core to vLLM."""
    if mode not in {"auto", "torch", "vllm"}:
        raise ValueError("MoE backend must be auto, torch, or vllm")
    if mode == "torch":
        return None
    if not (
        hasattr(backend, "packed_expert_weights")
        and hasattr(backend, "expert_map")
        and hasattr(backend, "expert_slot")
    ):
        if mode == "vllm":
            raise RuntimeError("vLLM fused MoE requires a packed expert backend")
        return None
    try:
        from vllm import envs
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    except Exception:
        if mode == "vllm":
            raise
        return None
    # The direct functional adapter bypasses FusedMoE layer setup, including its
    # platform checks. Force the portable Triton path: AC-MoE produced incorrect
    # results for physical slot IDs on PPU, while DeepGemm shape warmup is unbounded.
    envs.VLLM_MOE_USE_ACEXT = False
    envs.VLLM_USE_DEEP_GEMM = False
    return fused_experts


def optional_vllm_fused_topk(mode: str, backend) -> Callable[..., tuple[torch.Tensor, ...]] | None:
    """Resolve vLLM's fused router softmax/Top-K kernel with the MoE backend."""
    if mode not in {"auto", "torch", "vllm"}:
        raise ValueError("MoE backend must be auto, torch, or vllm")
    if mode == "torch":
        return None
    if not (
        hasattr(backend, "packed_expert_weights")
        and hasattr(backend, "expert_map")
        and hasattr(backend, "expert_slot")
    ):
        if mode == "vllm":
            raise RuntimeError("vLLM fused MoE requires a packed expert backend")
        return None
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
    except Exception:
        if mode == "vllm":
            raise
        return None
    return fused_topk


def optional_vllm_rms_norm(mode: str, backend) -> Callable[..., torch.Tensor] | None:
    """Resolve vLLM's fused RMSNorm alongside its packed MoE backend."""
    if mode not in {"auto", "torch", "vllm"}:
        raise ValueError("MoE backend must be auto, torch, or vllm")
    if mode == "torch":
        return None
    if not (
        hasattr(backend, "packed_expert_weights")
        and hasattr(backend, "expert_map")
        and hasattr(backend, "expert_slot")
    ):
        if mode == "vllm":
            raise RuntimeError("vLLM fused MoE requires a packed expert backend")
        return None
    try:
        from vllm.model_executor.layers.layernorm import rms_norm
    except Exception:
        if mode == "vllm":
            raise
        return None
    return rms_norm
