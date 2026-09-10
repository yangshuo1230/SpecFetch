from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

from src.runtime.config import RuntimeConfig
from src.runtime.expert import (
    ExpertRegistry,
    ExpertWeights,
    OffloadedExpertExecutor,
    expert_prediction_requests,
    optional_flash_kv_attention,
    optional_vllm_fused_add_rms_norm,
    optional_vllm_fused_moe,
    optional_vllm_fused_topk,
    optional_vllm_rms_norm,
    optional_vllm_rotary_embedding,
    pack_expert_weights,
)
from src.runtime.kv_cache import RequestLayerKV, SparseAttentionResult
from src.runtime.memory_queue import ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import OffloadRuntime, PrefetchRequest


class ModuleExpertSource:
    """CPU expert source extracted from a Transformers Qwen3-MoE model."""

    def __init__(self, model, pin_memory: bool = False) -> None:
        self.layers: list[list[nn.Module]] = []
        for layer in model.model.layers:
            experts = list(layer.mlp.experts)
            self.layers.append(experts)
            layer.mlp.experts = nn.ModuleList()
        self.pin_memory = pin_memory
        self._cache: dict[tuple[int, int], ExpertWeights] = {}

    def get(self, layer: int, expert: int) -> ExpertWeights:
        identity = (layer, expert)
        if identity not in self._cache:
            module = self.layers[layer][expert]
            gate, up, down = [
                parameter.weight.detach().contiguous()
                for parameter in (module.gate_proj, module.up_proj, module.down_proj)
            ]
            self._cache[identity] = (
                pack_expert_weights(gate, up, down, pin_memory=True)
                if self.pin_memory
                else ExpertWeights(gate, up, down)
            )
        return self._cache[identity]


@dataclass
class StepPredictions:
    kv: dict[tuple[str, int], dict[int, float]] = field(default_factory=dict)
    experts: dict[int, torch.Tensor] = field(default_factory=dict)
    expert_scores: dict[int, torch.Tensor] = field(default_factory=dict)


class SpeculativeConsumers:
    """Prediction leases indexed by layer and absolute-token consumer."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[int, str], list[ResourceKey]] = {}
        self._size = 0

    def __iter__(self):
        for (_, consumer), keys in self._buckets.items():
            for key in keys:
                yield key, consumer

    def __len__(self) -> int:
        return self._size

    def __eq__(self, other) -> bool:
        if isinstance(other, SpeculativeConsumers):
            return self._buckets == other._buckets
        return list(self) == other

    def add(self, key: ResourceKey, consumer: str) -> None:
        self._buckets.setdefault((key.layer, consumer), []).append(key)
        self._size += 1

    def contains(self, key: ResourceKey, consumer: str) -> bool:
        return key in self._buckets.get((key.layer, consumer), ())

    def contains_bucket(self, layer: int, consumer: str) -> bool:
        return (layer, consumer) in self._buckets

    def extend(self, consumers: Iterable[tuple[ResourceKey, str]]) -> None:
        for key, consumer in consumers:
            self.add(key, consumer)

    def pop_all(self) -> list[tuple[ResourceKey, str]]:
        consumers = list(self)
        self._buckets.clear()
        self._size = 0
        return consumers

    def pop_layer_consumers(
        self, layer: int, consumers: Iterable[str]
    ) -> list[tuple[ResourceKey, str]]:
        removed = []
        for consumer in consumers:
            keys = self._buckets.pop((layer, consumer), [])
            removed.extend((key, consumer) for key in keys)
            self._size -= len(keys)
        return removed

    def pop_request_owners(self, owners: set[str]) -> list[tuple[ResourceKey, str]]:
        removed = []
        for layer, consumer in list(self._buckets):
            if consumer.rsplit("@", 1)[0] not in owners:
                continue
            keys = self._buckets.pop((layer, consumer))
            removed.extend((key, consumer) for key in keys)
            self._size -= len(keys)
        return removed


@dataclass
class BatchState:
    request_ids: list[str]
    lengths: list[int]
    kv: dict[tuple[str, int], RequestLayerKV]
    step: int = 0
    speculative_consumers: SpeculativeConsumers = field(default_factory=SpeculativeConsumers)
    resident_groups: list[ResidentKVGroup] = field(default_factory=list)


@dataclass
class ResidentKVGroup:
    """Batch-contiguous resident KV allocated by one uniform-length prefill."""

    request_ids: list[str]
    length: int
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def initialize_layer(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("batched KV must have shape (batch, tokens, kv_heads, head_dim)")
        if key.shape[:2] != (len(self.request_ids), self.length):
            raise ValueError("batched KV does not match its resident group")
        if self.length > capacity:
            raise ValueError(
                f"prefix has {self.length} tokens but resident KV capacity is {capacity}"
            )
        buffers = tuple(
            torch.empty(
                (len(self.request_ids), capacity, *tensor.shape[2:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for tensor in (key, value)
        )
        for buffer, tensor in zip(buffers, (key, value)):
            buffer[:, : self.length].copy_(tensor)
        self.layers[layer] = buffers
        return buffers

    def append_layer(
        self,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> int:
        buffers = self.layers[layer]
        end = self.length + key.shape[1]
        if end > buffers[0].shape[1]:
            raise RuntimeError(
                f"resident KV capacity {buffers[0].shape[1]} exceeded by token {end}"
            )
        expected = (len(self.request_ids), key.shape[1], *buffers[0].shape[2:])
        if key.shape != expected or value.shape != expected:
            raise ValueError("decode KV does not match its resident group")
        for buffer, tensor in zip(buffers, (key, value)):
            buffer[: len(self.request_ids), self.length : end].copy_(tensor)
        return end

    def attention(self, layer: int, query: torch.Tensor, tokens: int) -> torch.Tensor:
        key, value = self.layers[layer]
        key = key[: len(self.request_ids), :tokens].transpose(1, 2)
        value = value[: len(self.request_ids), :tokens].transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            enable_gqa=query.shape[1] != key.shape[1],
        )

    def append_attention(
        self,
        layer: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scaling: float,
        flash_attention=None,
    ) -> tuple[torch.Tensor, int]:
        """Append decode KV and attend, using one fused kernel when available."""
        if flash_attention is None:
            tokens = self.append_layer(layer, key, value)
            return self.attention(layer, query, tokens), tokens
        key_cache, value_cache = self.layers[layer]
        tokens = self.length + key.shape[1]
        if tokens > key_cache.shape[1]:
            raise RuntimeError(
                f"resident KV capacity {key_cache.shape[1]} exceeded by token {tokens}"
            )
        expected = (len(self.request_ids), key.shape[1], *key_cache.shape[2:])
        if key.shape != expected or value.shape != expected:
            raise ValueError("decode KV does not match its resident group")
        output = flash_attention(
            query.transpose(1, 2),
            key_cache,
            value_cache,
            k=key,
            v=value,
            cache_seqlens=self.length,
            softmax_scale=scaling,
            causal=True,
        )
        return output.transpose(1, 2), tokens

    def compact(
        self,
        removing: set[str],
        caches: dict[tuple[str, int], RequestLayerKV],
    ) -> None:
        retained = [
            (source, request_id)
            for source, request_id in enumerate(self.request_ids)
            if request_id not in removing
        ]
        if len(retained) == len(self.request_ids):
            return
        compacted = {}
        for layer, buffers in self.layers.items():
            replacements = tuple(
                torch.empty(
                    (len(retained), *buffer.shape[1:]),
                    dtype=buffer.dtype,
                    device=buffer.device,
                )
                for buffer in buffers
            )
            for target, (source, _) in enumerate(retained):
                for replacement, buffer in zip(replacements, buffers):
                    replacement[target, : self.length].copy_(buffer[source, : self.length])
            compacted[layer] = replacements
        self.layers = compacted
        self.request_ids = [request_id for _, request_id in retained]
        for row, request_id in enumerate(self.request_ids):
            for layer, buffers in self.layers.items():
                caches[(request_id, layer)].rebind_resident((buffers[0][row], buffers[1][row]))


@dataclass
class EngineOutput:
    logits: torch.Tensor
    state: BatchState
    sparse_attention: dict[tuple[str, int], SparseAttentionResult]


def _repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    return states.repeat_interleave(groups, dim=1)


def _dense_causal_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float
):
    groups = query.shape[1] // key.shape[1]
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)
    if query.is_cuda:
        return (
            torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True, scale=scale
            )
            .transpose(1, 2)
            .contiguous()
        )
    logits = torch.matmul(query.float(), key.float().transpose(2, 3)) * scale
    length = query.shape[-2]
    mask = torch.ones((length, length), dtype=torch.bool, device=query.device).triu(1)
    logits.masked_fill_(mask, -torch.inf)
    weights = torch.softmax(logits, dim=-1).to(value.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


class Qwen3SparseOffloadEngine:
    """Readable Qwen3-MoE prefill/decode adapter for the offload runtime.

    All non-expert modules remain ordinary Transformers modules. Routed experts are
    removed from the model tree and resolved through ``OffloadedExpertExecutor``.
    """

    def __init__(
        self,
        model,
        expert_source,
        runtime: OffloadRuntime,
        residency: ResidencyManager,
        config: RuntimeConfig,
        *,
        moe_backend: str = "torch",
    ) -> None:
        self.model = model.eval()
        self.runtime = runtime
        self.residency = residency
        self.config = config
        self.expert_registry = ExpertRegistry(expert_source, residency)
        self._qkv_weights = self._pack_qkv_weights()
        fused_moe = optional_vllm_fused_moe(moe_backend, runtime.worker.backend)
        fused_topk = optional_vllm_fused_topk(moe_backend, runtime.worker.backend)
        self._rms_norm = optional_vllm_rms_norm(moe_backend, runtime.worker.backend)
        self._fused_add_rms_norm = optional_vllm_fused_add_rms_norm(
            moe_backend, runtime.worker.backend
        )
        self._rotary_embedding = optional_vllm_rotary_embedding(
            moe_backend, runtime.worker.backend, model
        )
        self._flash_attention = optional_flash_kv_attention(moe_backend, runtime.worker.backend)
        if config.kv_storage == "resident":
            self.attention_backend = (
                "flash_kvcache" if self._flash_attention is not None else "sdpa"
            )
        else:
            self.attention_backend = "sparse_hybrid"
        self.moe_backend = "vllm" if fused_moe is not None else "torch"
        self.experts = OffloadedExpertExecutor(
            runtime,
            self.expert_registry,
            top_k=model.config.num_experts_per_tok,
            fused_moe=fused_moe,
            fused_topk=fused_topk,
        )

    def _pack_qkv_weights(self) -> list[torch.Tensor]:
        """Fuse bias-free Q/K/V projections without retaining duplicate parameters."""
        packed = []
        for layer in self.model.model.layers:
            projections = (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)
            if any(projection.bias is not None for projection in projections):
                raise ValueError("packed QKV projection requires bias-free attention")
            packed.append(torch.cat([projection.weight.detach() for projection in projections]))
            for projection in projections:
                projection.weight = nn.Parameter(
                    projection.weight.detach().new_empty(0),
                    requires_grad=False,
                )
        return packed

    def _norm(self, module, hidden: torch.Tensor) -> torch.Tensor:
        if self._rms_norm is None:
            return module(hidden)
        return self._rms_norm(hidden, module.weight, module.variance_epsilon)

    def _add_norm(
        self,
        module,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._norm(module, hidden), hidden
        if self._fused_add_rms_norm is not None:
            return self._fused_add_rms_norm(
                hidden,
                residual,
                module.weight,
                module.variance_epsilon,
            )
        combined = hidden + residual
        return self._norm(module, combined), combined

    @classmethod
    def from_transformers_model(
        cls,
        model,
        runtime: OffloadRuntime,
        residency: ResidencyManager,
        config: RuntimeConfig,
        *,
        pin_experts: bool = False,
        moe_backend: str = "torch",
    ) -> Qwen3SparseOffloadEngine:
        source = ModuleExpertSource(model, pin_memory=pin_experts)
        return cls(model, source, runtime, residency, config, moe_backend=moe_backend)

    @property
    def device(self) -> torch.device:
        return self.model.model.embed_tokens.weight.device

    @staticmethod
    def resident_kv_payload_bytes(model, config: RuntimeConfig, batch_size: int) -> int:
        """Return the exact persistent buffer payload allocated by resident KV."""
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if config.kv_storage != "resident":
            return 0
        capacity = config.resident_kv_capacity_tokens
        assert capacity is not None
        total = 0
        for layer in model.model.layers:
            attention = layer.self_attn
            total += (
                batch_size
                * capacity
                * (
                    attention.k_proj.out_features * attention.k_proj.weight.element_size()
                    + attention.v_proj.out_features * attention.v_proj.weight.element_size()
                )
            )
        return total

    def resident_kv_allocation_bytes(self, batch_size: int) -> int:
        return self.resident_kv_payload_bytes(self.model, self.config, batch_size)

    @staticmethod
    def expert_slot_bytes(model) -> int:
        """Return one packed gate/up/down expert slot payload."""
        return (
            3
            * model.config.hidden_size
            * model.config.moe_intermediate_size
            * model.model.embed_tokens.weight.element_size()
        )

    def expert_slot_allocation_bytes(self) -> int:
        """Return the exact packed payload reserved by fixed expert slots."""
        return self.config.expert_cache_slots * self.expert_slot_bytes(self.model)

    def _project(
        self,
        layer,
        layer_index: int,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings,
    ):
        attention = layer.self_attn
        projected = torch.nn.functional.linear(hidden, self._qkv_weights[layer_index])
        query_states, key_states, value = projected.split(
            (
                attention.q_proj.out_features,
                attention.k_proj.out_features,
                attention.v_proj.out_features,
            ),
            dim=-1,
        )
        shape = (*hidden.shape[:-1], -1, attention.head_dim)
        query = self._norm(attention.q_norm, query_states.view(shape))
        key = self._norm(attention.k_norm, key_states.view(shape))
        value = value.view(shape).transpose(1, 2)
        if self._rotary_embedding is not None:
            query, key = self._rotary_embedding(positions, query, key)
            return query.transpose(1, 2), key.transpose(1, 2), value
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        assert position_embeddings is not None
        return apply_rotary_pos_emb(query, key, *position_embeddings) + (value,)

    def _moe(self, layer, hidden: torch.Tensor, layer_index: int, request_ids: list[str]):
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits = layer.mlp.gate(flat)
        token_requests = [request for request in request_ids for _ in range(shape[1])]
        output = self.experts(
            flat,
            logits,
            layer=layer_index,
            request_ids=token_requests,
        )
        return output.reshape(shape)

    @torch.inference_mode()
    def warmup_moe(self, token_counts: list[int]) -> bool:
        """在请求计时前编译实际形状的融合 MoE，并恢复冷专家驻留。"""
        if self.moe_backend != "vllm":
            return False
        counts = list(dict.fromkeys(token_counts))
        if not counts or any(count <= 0 for count in counts):
            raise ValueError("MoE warmup token 数必须为正")
        top_k = min(self.model.config.num_experts_per_tok, self.model.config.num_experts)
        dtype = self.model.model.embed_tokens.weight.dtype
        for count in counts:
            hidden = torch.zeros(
                (count, self.model.config.hidden_size), dtype=dtype, device=self.device
            )
            router_logits = torch.full(
                (count, self.model.config.num_experts),
                -100.0,
                dtype=dtype,
                device=self.device,
            )
            router_logits[:, :top_k] = torch.arange(top_k, dtype=dtype, device=self.device)
            self.experts(
                hidden,
                router_logits,
                layer=0,
                request_ids=["__moe_warmup__"] * count,
            )
        torch.cuda.synchronize(self.device)
        for key in list(self.residency.resident_keys(ResourceKind.EXPERT)):
            self.residency.evict(key)
        return True

    def initialize_expert_maps(self) -> bool:
        """Materialize every persistent logical-to-slot map before request timing."""
        if self.moe_backend != "vllm":
            return False
        expert_map = getattr(self.runtime.worker.backend, "expert_map", None)
        if not callable(expert_map):
            return False
        for layer in range(len(self.model.model.layers)):
            expert_map(layer, self.model.config.num_experts)
        return True

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        request_ids: list[str],
        *,
        full_logits: bool = False,
    ) -> EngineOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] != len(request_ids):
            raise ValueError("input IDs must be a uniform batch aligned with request IDs")
        input_ids = input_ids.to(self.device)
        batch, tokens = input_ids.shape
        hidden = self.model.model.embed_tokens(input_ids)
        residual = None
        positions = torch.arange(tokens, device=self.device).expand(batch, -1)
        position_embeddings = (
            None
            if self._rotary_embedding is not None
            else self.model.model.rotary_emb(hidden, positions)
        )
        caches = {}
        resident_group = (
            ResidentKVGroup(list(request_ids), tokens)
            if self.config.kv_storage == "resident"
            else None
        )
        for layer_index, layer in enumerate(self.model.model.layers):
            normalized, residual = self._add_norm(layer.input_layernorm, hidden, residual)
            query, key, value = self._project(
                layer,
                layer_index,
                normalized,
                positions,
                position_embeddings,
            )
            attended = _dense_causal_attention(query, key, value, layer.self_attn.scaling).reshape(
                batch, tokens, -1
            )
            attention_output = layer.self_attn.o_proj(attended)
            resident_buffers = None
            if resident_group is not None:
                capacity = self.config.resident_kv_capacity_tokens
                assert capacity is not None
                resident_buffers = resident_group.initialize_layer(
                    layer_index,
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    capacity,
                )
            for request_index, request_id in enumerate(request_ids):
                cache = RequestLayerKV(
                    request_id,
                    layer_index,
                    self.config,
                    self.residency,
                    self.runtime,
                )
                if resident_buffers is None:
                    cache.initialize(
                        key[request_index].transpose(0, 1),
                        value[request_index].transpose(0, 1),
                    )
                else:
                    cache.initialize_resident(
                        (resident_buffers[0][request_index], resident_buffers[1][request_index]),
                        tokens,
                    )
                caches[(request_id, layer_index)] = cache
            normalized, residual = self._add_norm(
                layer.post_attention_layernorm,
                attention_output,
                residual,
            )
            hidden = self._moe(
                layer,
                normalized,
                layer_index,
                request_ids,
            )
        hidden, _ = self._add_norm(self.model.model.norm, hidden, residual)
        # Serving consumes only the next-token row. At the 4K release workload,
        # projecting every prefix row to a 152K vocabulary would materialize
        # roughly 4.6 GiB of logits for batch four despite being immediately
        # discarded. Full logits remain opt-in for numerical reference tests.
        logits = self.model.lm_head(hidden if full_logits else hidden[:, -1:])
        resident_groups = [resident_group] if resident_group is not None else []
        return EngineOutput(
            logits,
            BatchState(request_ids, [tokens] * batch, caches, resident_groups=resident_groups),
            {},
        )

    def _reset_prediction_window(self, state: BatchState) -> None:
        self.runtime.cancel_many(state.speculative_consumers.pop_all())

    def _retire_prediction_layer(self, state: BatchState, layer_index: int) -> None:
        """Cancel current-token candidates after their layer's last possible use."""
        current_consumers = {f"{request_id}@{state.step + 1}" for request_id in state.request_ids}
        retiring = state.speculative_consumers.pop_layer_consumers(layer_index, current_consumers)
        self.runtime.cancel_many(retiring)

    def _enqueue_prediction_items(
        self,
        state: BatchState,
        items: Iterable[tuple[int, StepPredictions, int]],
    ) -> None:
        layers = len(self.model.model.layers)
        prefetch_requests: list[PrefetchRequest] = []
        for horizon, prediction, layer_index in items:
            deadline = (state.step + horizon - 1) * layers + layer_index
            expert_probabilities = prediction.experts.get(layer_index)
            expert_scores = prediction.expert_scores.get(layer_index)
            consumers = [f"{request_id}@{state.step + horizon}" for request_id in state.request_ids]
            new_indices = [
                index
                for index, consumer in enumerate(consumers)
                if not state.speculative_consumers.contains_bucket(layer_index, consumer)
            ]
            if (expert_probabilities is not None or expert_scores is not None) and new_indices:
                route_values = expert_scores if expert_scores is not None else expert_probabilities
                selected_probabilities = (
                    route_values
                    if len(new_indices) == len(consumers)
                    else route_values[new_indices]
                )
                expert_requests, _ = expert_prediction_requests(
                    selected_probabilities,
                    layer=layer_index,
                    request_ids=[consumers[index] for index in new_indices],
                    top_k=self.model.config.num_experts_per_tok,
                    deadline=deadline,
                    miss_cost_ms=0.5,
                    registry=self.expert_registry,
                    logits=expert_scores is not None,
                    collect_queued=False,
                )
                prefetch_requests.extend(expert_requests)
            if self.config.kv_storage != "resident":
                for index in new_indices:
                    request_id = state.request_ids[index]
                    consumer = consumers[index]
                    cache = state.kv[(request_id, layer_index)]
                    scores = prediction.kv.get((request_id, layer_index), {})
                    kv_requests, _ = cache.prefetch_requests(
                        scores,
                        deadline,
                        miss_cost_ms=0.05,
                        consumer=consumer,
                    )
                    prefetch_requests.extend(kv_requests)
        candidate_count = len(prefetch_requests)
        prefetch_requests = self._apply_prefetch_budget(prefetch_requests)
        admitted_count = len(prefetch_requests)
        unseen_requests = []
        seen = set()
        for request in prefetch_requests:
            identity = (request.key, request.consumer)
            if identity in seen or state.speculative_consumers.contains(*identity):
                continue
            seen.add(identity)
            unseen_requests.append(request)
        prefetch_requests = unseen_requests
        state.speculative_consumers.extend(
            (request.key, request.consumer) for request in prefetch_requests
        )
        self.runtime.prefetch_many(
            prefetch_requests,
            candidate_count=candidate_count - (admitted_count - len(prefetch_requests)),
        )

    def _enqueue_prediction_layers(
        self,
        state: BatchState,
        predictions: list[StepPredictions],
        layer_indices: Iterable[int],
    ) -> None:
        layer_indices = tuple(layer_indices)
        self._enqueue_prediction_items(
            state,
            [
                (horizon, prediction, layer_index)
                for horizon, prediction in enumerate(predictions, 1)
                for layer_index in layer_indices
            ],
        )

    def _retain_predictions(self, state: BatchState, predictions: list[StepPredictions]) -> None:
        if self.config.kv_storage == "resident":
            return
        layers = len(self.model.model.layers)
        for layer_index in range(layers):
            for request_id in state.request_ids:
                cache = state.kv[(request_id, layer_index)]
                keep_predictions = [
                    item.kv.get((request_id, layer_index), {}) for item in predictions[:2]
                ]
                cache.retain_predicted(keep_predictions)

    def enqueue_predictions(self, state: BatchState, predictions: list[StepPredictions]) -> None:
        """Replace the prediction window and admit all layers (compatibility API)."""
        self._reset_prediction_window(state)
        self._enqueue_prediction_layers(state, predictions, range(len(self.model.model.layers)))
        self._retain_predictions(state, predictions)

    def _apply_prefetch_budget(self, requests: list[PrefetchRequest]) -> list[PrefetchRequest]:
        """Admit the highest-value unique resources without splitting shared consumers."""
        budgets = {
            ResourceKind.EXPERT: self.config.speculative_expert_budget,
            ResourceKind.KV: self.config.speculative_kv_budget,
        }
        if all(budget is None for budget in budgets.values()):
            return requests
        grouped: dict[ResourceKey, list[PrefetchRequest]] = {}
        for request in requests:
            grouped.setdefault(request.key, []).append(request)
        selected: set[ResourceKey] = set()
        current_step = self.runtime.queue.current_step
        for kind, budget in budgets.items():
            keys = [key for key in grouped if key.kind == kind]
            if budget is None or len(keys) <= budget:
                selected.update(keys)
                continue

            def rank(key: ResourceKey):
                consumers = grouped[key]
                deadline = min(request.deadline for request in consumers)
                probability = sum(request.probability for request in consumers)
                miss_cost_ms = max(request.miss_cost_ms for request in consumers)
                size_bytes = self.residency.record(key).size_bytes
                urgency = 1.0 / max(1, deadline - current_step)
                priority = miss_cost_ms * probability * urgency / max(size_bytes / 2**20, 1e-6)
                identity = (key.layer, key.object_id, key.request_id)
                return (-priority, deadline, identity)

            selected.update(sorted(keys, key=rank)[:budget])
        return [request for request in requests if request.key in selected]

    def add_state(self, state: BatchState, admitted: BatchState) -> None:
        """Append separately-prefilled requests to an active decode batch."""
        overlap = set(state.request_ids) & set(admitted.request_ids)
        if overlap:
            raise ValueError(f"requests are already active: {sorted(overlap)}")
        state.request_ids.extend(admitted.request_ids)
        state.lengths.extend(admitted.lengths)
        state.kv.update(admitted.kv)
        state.speculative_consumers.extend(admitted.speculative_consumers)
        state.resident_groups.extend(admitted.resident_groups)

    @torch.inference_mode()
    def remove_requests(self, state: BatchState, request_ids: list[str]) -> None:
        """Cancel predictions and free KV storage for completed requests."""
        removing = set(request_ids)
        if not removing <= set(state.request_ids):
            raise KeyError("cannot remove a request that is not active")
        self.runtime.cancel_many(state.speculative_consumers.pop_request_owners(removing))
        for request_id in removing:
            for layer_index in range(len(self.model.model.layers)):
                state.kv.pop((request_id, layer_index)).close()
        for group in state.resident_groups:
            group.compact(removing, state.kv)
        state.resident_groups = [group for group in state.resident_groups if group.request_ids]
        retained = [
            (request_id, length)
            for request_id, length in zip(state.request_ids, state.lengths)
            if request_id not in removing
        ]
        state.request_ids = [item[0] for item in retained]
        state.lengths = [item[1] for item in retained]

    @torch.inference_mode()
    def decode(
        self,
        token_ids: torch.Tensor,
        state: BatchState,
        predictions: StepPredictions | list[StepPredictions],
        *,
        prefetch: bool = True,
        reuse_prediction_window: bool = False,
        shadow_attention: bool = False,
        shadow_thresholds: tuple[float, ...] = (),
    ) -> EngineOutput:
        if token_ids.ndim != 1 or len(token_ids) != len(state.request_ids):
            raise ValueError("decode requires one token per active request")
        predictions = [predictions] if isinstance(predictions, StepPredictions) else predictions
        if not predictions:
            predictions = [StepPredictions()]
        reuse_prediction_window = reuse_prediction_window and (
            self.config.speculative_expert_budget is None
            and self.config.speculative_kv_budget is None
            and self.config.speculative_layer_lookahead is None
        )
        if prefetch and not reuse_prediction_window:
            self._reset_prediction_window(state)
            self._retain_predictions(state, predictions)
        current_predictions = predictions[0]
        hidden = self.model.model.embed_tokens(token_ids[:, None].to(self.device))
        residual = None
        positions = torch.tensor(state.lengths, device=self.device)[:, None]
        position_embeddings = (
            None
            if self._rotary_embedding is not None
            else self.model.model.rotary_emb(hidden, positions)
        )
        traces = {}
        layers = len(self.model.model.layers)
        scheduled_deadlines: set[int] = set()
        for layer_index, layer in enumerate(self.model.model.layers):
            current_deadline = state.step * layers + layer_index
            self.runtime.queue.set_step(current_deadline)
            if prefetch:
                if self.config.speculative_layer_lookahead is None:
                    if not scheduled_deadlines:
                        self._enqueue_prediction_layers(state, predictions, range(layers))
                        scheduled_deadlines.update(
                            (state.step + horizon - 1) * layers + predicted_layer
                            for horizon in range(1, len(predictions) + 1)
                            for predicted_layer in range(layers)
                        )
                else:
                    items = []
                    for deadline in range(
                        current_deadline,
                        current_deadline + self.config.speculative_layer_lookahead,
                    ):
                        if deadline in scheduled_deadlines:
                            continue
                        horizon = deadline // layers - state.step + 1
                        if not 1 <= horizon <= len(predictions):
                            continue
                        items.append((horizon, predictions[horizon - 1], deadline % layers))
                        scheduled_deadlines.add(deadline)
                    self._enqueue_prediction_items(state, items)
            normalized, residual = self._add_norm(layer.input_layernorm, hidden, residual)
            query, key, value = self._project(
                layer,
                layer_index,
                normalized,
                positions,
                position_embeddings,
            )
            if state.resident_groups:
                grouped_ids = [
                    request_id
                    for group in state.resident_groups
                    for request_id in group.request_ids
                ]
                if grouped_ids != state.request_ids:
                    raise RuntimeError("resident KV groups are not aligned with the decode batch")
                attended_groups = []
                offset = 0
                for group in state.resident_groups:
                    end = offset + len(group.request_ids)
                    group_output, _ = group.append_attention(
                        layer_index,
                        query[offset:end],
                        key[offset:end].transpose(1, 2),
                        value[offset:end].transpose(1, 2),
                        scaling=layer.self_attn.scaling,
                        flash_attention=self._flash_attention,
                    )
                    attended_groups.append(group_output.transpose(1, 2))
                    for row, request_id in enumerate(group.request_ids):
                        cache = state.kv[(request_id, layer_index)]
                        cache.commit_resident_append()
                        traces[(request_id, layer_index)] = SparseAttentionResult(
                            group_output[row, :, 0], [], 1.0, []
                        )
                    offset = end
                attended = (
                    attended_groups[0]
                    if len(attended_groups) == 1
                    else torch.cat(attended_groups, dim=0)
                )
                attended_tensor = attended.reshape(len(state.request_ids), 1, -1)
                attention_output = layer.self_attn.o_proj(attended_tensor)
                normalized, residual = self._add_norm(
                    layer.post_attention_layernorm,
                    attention_output,
                    residual,
                )
                hidden = self._moe(
                    layer,
                    normalized,
                    layer_index,
                    state.request_ids,
                )
                if prefetch:
                    self._retire_prediction_layer(state, layer_index)
                continue
            layer_scores = {}
            for request_index, request_id in enumerate(state.request_ids):
                cache = state.kv[(request_id, layer_index)]
                cache.append(
                    key[request_index].transpose(0, 1),
                    value[request_index].transpose(0, 1),
                )
                scores = current_predictions.kv.get((request_id, layer_index))
                if scores is None:
                    count = len(cache.old)
                    scores = {index: 1 / count for index in cache.old} if count else {}
                layer_scores[request_id] = scores
            guaranteed_requests = []
            for request_id in state.request_ids:
                cache = state.kv[(request_id, layer_index)]
                guaranteed_requests.extend(
                    cache.guaranteed_demand_requests(layer_scores[request_id], miss_cost_ms=0.05)
                )
            guaranteed_payloads = (
                self.runtime.demand_many(guaranteed_requests)
                if len(guaranteed_requests) <= self.residency.capacities[ResourceKind.KV]
                else None
            )
            attended = []
            for request_index, request_id in enumerate(state.request_ids):
                cache = state.kv[(request_id, layer_index)]
                result = cache.sparse_attention(
                    query[request_index, :, 0],
                    layer_scores[request_id],
                    miss_cost_ms=0.05,
                    guaranteed_payloads=guaranteed_payloads,
                    shadow=shadow_attention,
                    shadow_thresholds=shadow_thresholds,
                )
                traces[(request_id, layer_index)] = result
                attended.append(result.output)
            attended_tensor = torch.stack(attended)[:, None].reshape(len(attended), 1, -1)
            attention_output = layer.self_attn.o_proj(attended_tensor)
            normalized, residual = self._add_norm(
                layer.post_attention_layernorm,
                attention_output,
                residual,
            )
            hidden = self._moe(
                layer,
                normalized,
                layer_index,
                state.request_ids,
            )
            if prefetch:
                self._retire_prediction_layer(state, layer_index)
        hidden, _ = self._add_norm(self.model.model.norm, hidden, residual)
        logits = self.model.lm_head(hidden)
        for group in state.resident_groups:
            group.length += 1
        state.lengths = [length + 1 for length in state.lengths]
        state.step += 1
        return EngineOutput(logits, state, traces)


def clone_for_reference(model):
    """Tiny-model test helper kept here to document adapter ownership semantics."""
    return copy.deepcopy(model)
