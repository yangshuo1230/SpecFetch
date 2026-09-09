from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.metrics import predict_probe
from src.runtime.qwen3_engine import BatchState, StepPredictions
from src.trace import map_layer


def to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    """Transfer compact source dtype first, then promote for CPU signal math."""
    return tensor.detach().cpu().float()


def aggregate_old_chunk_mass(
    attention: torch.Tensor, ranges: dict[int, tuple[int, int]]
) -> dict[int, float]:
    """Aggregate one draft attention row over target old-KV token ranges."""
    if attention.ndim != 2:
        raise ValueError("attention must have shape (heads, keys)")
    token_mass = attention.detach().float().sum(dim=0)
    masses = {
        chunk: float(token_mass[start : min(end, len(token_mass))].sum())
        for chunk, (start, end) in ranges.items()
        if start < len(token_mass)
    }
    total = sum(masses.values())
    return {chunk: mass / total for chunk, mass in masses.items()} if total else {}


@dataclass
class ProbeEntry:
    draft_layer: int
    probe: dict[str, torch.Tensor]


class ExpertProbeBank:
    def __init__(self, entries: dict[int, ProbeEntry]) -> None:
        self.entries = entries
        self._parameter_batches: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}

    def predict(self, target_layer: int, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        entry = self.entries[target_layer]
        features = to_cpu_float(hidden_states[entry.draft_layer + 1][:, -1])
        return self.predict_features(target_layer, features)

    def predict_features(self, target_layer: int, features: torch.Tensor) -> torch.Tensor:
        """使用已搬到 CPU 的特征，避免相同 Draft 层重复同步。"""
        entry = self.entries[target_layer]
        scores = predict_probe(entry.probe, features)
        return torch.sigmoid(scores)

    def predict_feature_batch(
        self,
        target_layers: list[int],
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate aligned layers as one batched probe GEMM.

        ``features`` has shape ``(target_layers, samples, hidden_size)``.
        """
        key = tuple(target_layers)
        if features.ndim != 3 or len(features) != len(key):
            raise ValueError("batched probe features must align with target layers")
        parameters = self._parameter_batches.get(key)
        if parameters is None:
            probes = [self.entries[layer].probe for layer in key]
            coefficients = []
            biases = []
            for probe in probes:
                scale = probe["x_scale"].squeeze(0)
                mean = probe["x_mean"].squeeze(0)
                coefficient = probe["coef"] / scale[:, None]
                bias = probe["y_mean"] - (mean / scale) @ probe["coef"]
                coefficients.append(coefficient)
                biases.append(bias)
            parameters = (torch.stack(coefficients), torch.stack(biases))
            self._parameter_batches[key] = parameters
        coefficients, biases = parameters
        return torch.sigmoid(torch.bmm(features.float(), coefficients) + biases)

    def save(self, path: str | Path) -> None:
        payload = {
            layer: {"draft_layer": entry.draft_layer, "probe": entry.probe}
            for layer, entry in self.entries.items()
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> ExpertProbeBank:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            {
                int(layer): ProbeEntry(value["draft_layer"], value["probe"])
                for layer, value in payload.items()
            }
        )


@dataclass
class DraftPredictionPlan:
    horizons: list[StepPredictions]
    token_ids: torch.Tensor


class DraftSignalProvider:
    """Stateful draft rollout that rolls its cache back after each lookahead."""

    def __init__(self, model, probe_bank: ExpertProbeBank | None, lookahead: int = 4) -> None:
        if lookahead <= 0:
            raise ValueError("lookahead must be positive")
        self.model = model.eval()
        self.probe_bank = probe_bank
        self.lookahead = lookahead
        self.cache = None
        self.next_logits = None
        self.request_ids: list[str] = []

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def initialize(self, input_ids: torch.Tensor, request_ids: list[str]) -> None:
        if input_ids.ndim != 2 or len(input_ids) != len(request_ids):
            raise ValueError("draft input must be a uniform request batch")
        output = self.model(
            input_ids=input_ids.to(self.device),
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        self.cache = output.past_key_values
        self.next_logits = output.logits[:, -1]
        self.request_ids = list(request_ids)

    def split_requests(self) -> list[DraftSignalProvider]:
        """将批量前缀状态拆为可独立推进的单请求 provider。"""
        if self.cache is None or self.next_logits is None or not self.request_ids:
            raise RuntimeError("拆分前必须先初始化 Draft provider")
        if len(self.request_ids) == 1:
            return [self]
        split_cache = getattr(self.cache, "batch_split", None)
        if not callable(split_cache):
            raise TypeError("Draft KV cache 不支持按 batch 拆分")
        caches = split_cache(len(self.request_ids), 1)
        if len(caches) != len(self.request_ids):
            raise RuntimeError("Draft KV cache 拆分数量与请求数不一致")
        providers = []
        for index, (request_id, cache) in enumerate(zip(self.request_ids, caches)):
            provider = DraftSignalProvider(self.model, self.probe_bank, self.lookahead)
            provider.cache = cache
            provider.next_logits = self.next_logits[index : index + 1]
            provider.request_ids = [request_id]
            providers.append(provider)
        return providers

    @classmethod
    def merge_requests(cls, providers: list[DraftSignalProvider]) -> DraftSignalProvider:
        """合并兼容的单请求 provider，用于批量推进和 rollout。"""
        if not providers:
            raise ValueError("至少需要一个 Draft provider")
        first = providers[0]
        if first.cache is None or first.next_logits is None:
            raise RuntimeError("合并前必须先初始化 Draft provider")
        cache_length = first.cache.get_seq_length()
        for provider in providers:
            if provider.cache is None or provider.next_logits is None:
                raise RuntimeError("合并前必须先初始化全部 Draft provider")
            if len(provider.request_ids) != 1:
                raise ValueError("只能合并单请求 Draft provider")
            if (
                provider.model is not first.model
                or provider.probe_bank is not first.probe_bank
                or provider.lookahead != first.lookahead
                or provider.cache.get_seq_length() != cache_length
                or provider.next_logits.shape[0] != 1
            ):
                raise ValueError("Draft provider 的模型、配置或缓存长度不兼容")
        cache_type = type(first.cache)
        merge_cache = getattr(cache_type, "from_batch_splits", None)
        if not callable(merge_cache):
            raise TypeError("Draft KV cache 不支持合并 batch split")
        merged = cls(first.model, first.probe_bank, first.lookahead)
        merged.cache = merge_cache([provider.cache for provider in providers])
        merged.next_logits = torch.cat([provider.next_logits for provider in providers])
        merged.request_ids = [provider.request_ids[0] for provider in providers]
        return merged

    @torch.inference_mode()
    def advance(self, actual_token_ids: torch.Tensor) -> None:
        if self.cache is None or len(actual_token_ids) != len(self.request_ids):
            raise RuntimeError("initialize draft state before advancing")
        if actual_token_ids.ndim == 1:
            actual_token_ids = actual_token_ids[:, None]
        elif actual_token_ids.ndim != 2:
            raise ValueError("actual tokens must have shape (batch,) or (batch, tokens)")
        output = self.model(
            input_ids=actual_token_ids.to(self.device),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        self.cache = output.past_key_values
        self.next_logits = output.logits[:, -1]

    @torch.inference_mode()
    def predict(self, target_state: BatchState) -> DraftPredictionPlan:
        if self.cache is None or self.next_logits is None:
            raise RuntimeError("draft provider is not initialized")
        if target_state.request_ids != self.request_ids:
            raise ValueError("draft and target request order differs")
        base_length = self.cache.get_seq_length()
        # Full-resident Target KV never consumes draft attention ranks.  Asking
        # Transformers for attentions in that mode materializes every draft
        # layer's (batch, heads, 1, context) tensor and then needlessly copies it
        # to CPU.  Keep hidden states independently enabled for expert probes.
        predict_kv = not target_state.kv or any(
            cache.config.kv_storage != "resident" for cache in target_state.kv.values()
        )
        predict_experts = self.probe_bank is not None
        token = self.next_logits.argmax(dim=-1)
        proposed = []
        outputs = []
        for _ in range(self.lookahead):
            proposed.append(token)
            output = self.model(
                input_ids=token[:, None],
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
                output_attentions=predict_kv,
                output_hidden_states=predict_experts,
                return_dict=True,
            )
            outputs.append(output)
            token = output.logits[:, -1].argmax(dim=-1)
        self.cache.crop(base_length)

        target_layers = len({layer for _, layer in target_state.kv})
        horizons = [StepPredictions() for _ in outputs]
        if predict_kv and target_layers:
            draft_layers = len(outputs[0].attentions)
            target_draft_layers = [
                map_layer(target_layer, target_layers, draft_layers)
                for target_layer in range(target_layers)
            ]
            unique_draft_layers = sorted(set(target_draft_layers))
            draft_offsets = {layer: index for index, layer in enumerate(unique_draft_layers)}
            for prediction, output in zip(horizons, outputs):
                # All last-token attention rows in one horizon have the same
                # shape. Transfer every mapped draft layer in one D2H operation
                # instead of synchronizing once per layer.
                cpu_attentions = torch.stack(
                    [output.attentions[layer][:, :, -1] for layer in unique_draft_layers]
                )
                cpu_attentions = to_cpu_float(cpu_attentions)
                for target_layer, draft_layer in enumerate(target_draft_layers):
                    attention = cpu_attentions[draft_offsets[draft_layer]]
                    for request_index, request_id in enumerate(self.request_ids):
                        ranges = target_state.kv[(request_id, target_layer)].old_ranges
                        prediction.kv[(request_id, target_layer)] = aggregate_old_chunk_mass(
                            attention[request_index], ranges
                        )
        if self.probe_bank is not None and target_layers:
            target_feature_layers = [
                self.probe_bank.entries[target_layer].draft_layer
                for target_layer in range(target_layers)
            ]
            unique_feature_layers = sorted(set(target_feature_layers))
            feature_offsets = {layer: index for index, layer in enumerate(unique_feature_layers)}
            # Hidden rows are uniform across both horizons and layers. A single
            # snapshot amortizes D2H synchronization for the complete rollout.
            cpu_features = to_cpu_float(
                torch.stack(
                    [
                        torch.stack(
                            [
                                output.hidden_states[layer + 1][:, -1]
                                for layer in unique_feature_layers
                            ]
                        )
                        for output in outputs
                    ]
                )
            )
            target_features = torch.stack(
                [
                    cpu_features[:, feature_offsets[feature_layer]].flatten(0, 1)
                    for feature_layer in target_feature_layers
                ]
            )
            probabilities = self.probe_bank.predict_feature_batch(
                list(range(target_layers)),
                target_features,
            ).unflatten(1, (len(outputs), len(self.request_ids)))
            for target_layer, layer_probabilities in enumerate(probabilities):
                for horizon, values in zip(horizons, layer_probabilities):
                    horizon.experts[target_layer] = values
        return DraftPredictionPlan(horizons, torch.stack(proposed, dim=1).detach().cpu())
