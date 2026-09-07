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

    @torch.inference_mode()
    def advance(self, actual_token_ids: torch.Tensor) -> None:
        if self.cache is None or len(actual_token_ids) != len(self.request_ids):
            raise RuntimeError("initialize draft state before advancing")
        output = self.model(
            input_ids=actual_token_ids[:, None].to(self.device),
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
            proposed.append(token.detach().cpu())
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
            for target_layer in range(target_layers):
                draft_layer = map_layer(target_layer, target_layers, draft_layers)
                attention = output.attentions[draft_layer][:, :, -1]
                for request_index, request_id in enumerate(self.request_ids):
                    ranges = target_state.kv[(request_id, target_layer)].old_ranges
                    prediction.kv[(request_id, target_layer)] = aggregate_old_chunk_mass(
                        attention[request_index], ranges
                    )
                if self.probe_bank is not None:
                    prediction.experts[target_layer] = self.probe_bank.predict(
                        target_layer, output.hidden_states
                    )
            horizons.append(prediction)
        return DraftPredictionPlan(horizons, torch.stack(proposed, dim=1))
