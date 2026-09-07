from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

from src.runtime.config import RuntimeConfig
from src.runtime.expert import (
    ExpertRegistry,
    ExpertWeights,
    OffloadedExpertExecutor,
    enqueue_expert_predictions,
)
from src.runtime.kv_cache import RequestLayerKV, SparseAttentionResult
from src.runtime.memory_queue import ResourceKey
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import OffloadRuntime


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
            values = [
                parameter.weight.detach().contiguous()
                for parameter in (module.gate_proj, module.up_proj, module.down_proj)
            ]
            if self.pin_memory and torch.cuda.is_available():
                values = [value.pin_memory() for value in values]
            self._cache[identity] = ExpertWeights(*values)
        return self._cache[identity]


@dataclass
class StepPredictions:
    kv: dict[tuple[str, int], dict[int, float]] = field(default_factory=dict)
    experts: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class BatchState:
    request_ids: list[str]
    lengths: list[int]
    kv: dict[tuple[str, int], RequestLayerKV]
    step: int = 0
    speculative_consumers: list[tuple[ResourceKey, str]] = field(default_factory=list)


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
    ) -> None:
        self.model = model.eval()
        self.runtime = runtime
        self.residency = residency
        self.config = config
        self.expert_registry = ExpertRegistry(expert_source, residency)
        self.experts = OffloadedExpertExecutor(
            runtime,
            self.expert_registry,
            top_k=model.config.num_experts_per_tok,
        )

    @classmethod
    def from_transformers_model(
        cls,
        model,
        runtime: OffloadRuntime,
        residency: ResidencyManager,
        config: RuntimeConfig,
        *,
        pin_experts: bool = False,
    ) -> Qwen3SparseOffloadEngine:
        source = ModuleExpertSource(model, pin_memory=pin_experts)
        return cls(model, source, runtime, residency, config)

    @property
    def device(self) -> torch.device:
        return self.model.model.embed_tokens.weight.device

    def _project(self, layer, hidden: torch.Tensor, position_embeddings):
        attention = layer.self_attn
        shape = (*hidden.shape[:-1], -1, attention.head_dim)
        query = attention.q_norm(attention.q_proj(hidden).view(shape)).transpose(1, 2)
        key = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
        value = attention.v_proj(hidden).view(shape).transpose(1, 2)
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
    def prefill(self, input_ids: torch.Tensor, request_ids: list[str]) -> EngineOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] != len(request_ids):
            raise ValueError("input IDs must be a uniform batch aligned with request IDs")
        input_ids = input_ids.to(self.device)
        batch, tokens = input_ids.shape
        hidden = self.model.model.embed_tokens(input_ids)
        positions = torch.arange(tokens, device=self.device).expand(batch, -1)
        position_embeddings = self.model.model.rotary_emb(hidden, positions)
        caches = {}
        for layer_index, layer in enumerate(self.model.model.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            query, key, value = self._project(layer, normalized, position_embeddings)
            attended = _dense_causal_attention(query, key, value, layer.self_attn.scaling).reshape(
                batch, tokens, -1
            )
            hidden = residual + layer.self_attn.o_proj(attended)
            for request_index, request_id in enumerate(request_ids):
                cache = RequestLayerKV(
                    request_id,
                    layer_index,
                    self.config,
                    self.residency,
                    self.runtime,
                )
                cache.initialize(
                    key[request_index].transpose(0, 1),
                    value[request_index].transpose(0, 1),
                )
                caches[(request_id, layer_index)] = cache
            residual = hidden
            hidden = residual + self._moe(
                layer,
                layer.post_attention_layernorm(hidden),
                layer_index,
                request_ids,
            )
        hidden = self.model.model.norm(hidden)
        logits = self.model.lm_head(hidden)
        return EngineOutput(logits, BatchState(request_ids, [tokens] * batch, caches), {})

    def enqueue_predictions(self, state: BatchState, predictions: list[StepPredictions]) -> None:
        for key, consumer in state.speculative_consumers:
            self.runtime.cancel(key, consumer)
        state.speculative_consumers.clear()
        layers = len(self.model.model.layers)
        for horizon, prediction in enumerate(predictions, 1):
            for layer_index in range(layers):
                deadline = (state.step + horizon - 1) * layers + layer_index
                expert_probabilities = prediction.experts.get(layer_index)
                consumers = [
                    f"{request_id}@{state.step + horizon}" for request_id in state.request_ids
                ]
                if expert_probabilities is not None:
                    state.speculative_consumers.extend(
                        enqueue_expert_predictions(
                            expert_probabilities,
                            layer=layer_index,
                            request_ids=consumers,
                            top_k=self.model.config.num_experts_per_tok,
                            deadline=deadline,
                            miss_cost_ms=0.5,
                            registry=self.expert_registry,
                            runtime=self.runtime,
                        )
                    )
                for request_id, consumer in zip(state.request_ids, consumers):
                    cache = state.kv[(request_id, layer_index)]
                    scores = prediction.kv.get((request_id, layer_index), {})
                    state.speculative_consumers.extend(
                        cache.enqueue(
                            scores,
                            deadline,
                            miss_cost_ms=0.05,
                            consumer=consumer,
                        )
                    )
        for layer_index in range(layers):
            for request_id in state.request_ids:
                cache = state.kv[(request_id, layer_index)]
                keep_predictions = [
                    item.kv.get((request_id, layer_index), {}) for item in predictions[:2]
                ]
                cache.retain_predicted(keep_predictions)

    @torch.inference_mode()
    def decode(
        self,
        token_ids: torch.Tensor,
        state: BatchState,
        predictions: StepPredictions | list[StepPredictions],
        *,
        prefetch: bool = True,
    ) -> EngineOutput:
        if token_ids.ndim != 1 or len(token_ids) != len(state.request_ids):
            raise ValueError("decode requires one token per active request")
        predictions = [predictions] if isinstance(predictions, StepPredictions) else predictions
        if not predictions:
            predictions = [StepPredictions()]
        if prefetch:
            self.enqueue_predictions(state, predictions)
        current_predictions = predictions[0]
        hidden = self.model.model.embed_tokens(token_ids[:, None].to(self.device))
        positions = torch.tensor(state.lengths, device=self.device)[:, None]
        position_embeddings = self.model.model.rotary_emb(hidden, positions)
        traces = {}
        layers = len(self.model.model.layers)
        for layer_index, layer in enumerate(self.model.model.layers):
            self.runtime.queue.set_step(state.step * layers + layer_index)
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            query, key, value = self._project(layer, normalized, position_embeddings)
            attended = []
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
                result = cache.sparse_attention(
                    query[request_index, :, 0], scores, miss_cost_ms=0.05
                )
                traces[(request_id, layer_index)] = result
                attended.append(result.output)
            attended_tensor = torch.stack(attended)[:, None].reshape(len(attended), 1, -1)
            hidden = residual + layer.self_attn.o_proj(attended_tensor)
            residual = hidden
            hidden = residual + self._moe(
                layer,
                layer.post_attention_layernorm(hidden),
                layer_index,
                state.request_ids,
            )
        hidden = self.model.model.norm(hidden)
        logits = self.model.lm_head(hidden)
        state.lengths = [length + 1 for length in state.lengths]
        state.step += 1
        return EngineOutput(logits, state, traces)


def clone_for_reference(model):
    """Tiny-model test helper kept here to document adapter ownership semantics."""
    return copy.deepcopy(model)
