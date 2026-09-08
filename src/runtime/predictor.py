from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.metrics import predict_probe
from src.runtime.qwen3_engine import BatchState, StepPredictions
from src.trace import map_layer


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

    def predict(self, target_layer: int, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        entry = self.entries[target_layer]
        features = hidden_states[entry.draft_layer + 1][:, -1].float().cpu()
        return self.predict_features(target_layer, features)

    def predict_features(self, target_layer: int, features: torch.Tensor) -> torch.Tensor:
        """使用已搬到 CPU 的特征，避免相同 Draft 层重复同步。"""
        entry = self.entries[target_layer]
        scores = predict_probe(entry.probe, features)
        return torch.sigmoid(scores)

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
        output = self.model(input_ids=input_ids.to(self.device), use_cache=True, return_dict=True)
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
        token = self.next_logits.argmax(dim=-1)
        proposed = []
        outputs = []
        for _ in range(self.lookahead):
            proposed.append(token)
            output = self.model(
                input_ids=token[:, None],
                past_key_values=self.cache,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            outputs.append(output)
            token = output.logits[:, -1].argmax(dim=-1)
        self.cache.crop(base_length)

        target_layers = len({layer for _, layer in target_state.kv})
        draft_layers = len(outputs[0].attentions)
        horizons = []
        for output in outputs:
            prediction = StepPredictions()
            cpu_attentions: dict[int, torch.Tensor] = {}
            cpu_features: dict[int, torch.Tensor] = {}
            for target_layer in range(target_layers):
                draft_layer = map_layer(target_layer, target_layers, draft_layers)
                if draft_layer not in cpu_attentions:
                    cpu_attentions[draft_layer] = (
                        output.attentions[draft_layer][:, :, -1].detach().float().cpu()
                    )
                attention = cpu_attentions[draft_layer]
                for request_index, request_id in enumerate(self.request_ids):
                    ranges = target_state.kv[(request_id, target_layer)].old_ranges
                    prediction.kv[(request_id, target_layer)] = aggregate_old_chunk_mass(
                        attention[request_index], ranges
                    )
                if self.probe_bank is not None:
                    feature_layer = self.probe_bank.entries[target_layer].draft_layer
                    if feature_layer not in cpu_features:
                        cpu_features[feature_layer] = (
                            output.hidden_states[feature_layer + 1][:, -1].detach().float().cpu()
                        )
                    prediction.experts[target_layer] = self.probe_bank.predict_features(
                        target_layer, cpu_features[feature_layer]
                    )
            horizons.append(prediction)
        return DraftPredictionPlan(horizons, torch.stack(proposed, dim=1).detach().cpu())
