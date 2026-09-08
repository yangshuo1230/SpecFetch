from __future__ import annotations

from dataclasses import dataclass

import torch

from src.runtime.config import RuntimeConfig
from src.runtime.hybrid_attention import (
    HybridStopController,
    attention_output,
    chunk_logsumexp,
    empty_partition,
    mean_target_marginal,
    update_partition,
)
from src.runtime.memory_queue import ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState
from src.runtime.transfer import OffloadRuntime, PrefetchRequest


def kv_key(request_id: str, layer: int, chunk: int) -> ResourceKey:
    return ResourceKey(ResourceKind.KV, layer=layer, object_id=chunk, request_id=request_id)


@dataclass
class SparseAttentionResult:
    output: torch.Tensor
    selected_old_chunks: list[int]
    predicted_mass: float
    target_marginals: list[float]


def _cpu_payload(key: torch.Tensor, value: torch.Tensor, pin: bool) -> tuple[torch.Tensor, ...]:
    result = []
    for tensor in (key, value):
        cpu = torch.empty_like(tensor, device="cpu", pin_memory=pin)
        cpu.copy_(tensor, non_blocking=tensor.is_cuda and pin)
        result.append(cpu)
    if key.is_cuda:
        torch.cuda.current_stream(key.device).synchronize()
    return tuple(result)


class RequestLayerKV:
    """Sink/recent GPU window plus CPU-resident old chunks for one request/layer."""

    def __init__(
        self,
        request_id: str,
        layer: int,
        config: RuntimeConfig,
        residency: ResidencyManager,
        runtime: OffloadRuntime,
        *,
        pin_cpu: bool | None = None,
    ) -> None:
        self.request_id = request_id
        self.layer = layer
        self.config = config
        self.residency = residency
        self.runtime = runtime
        self.pin_cpu = torch.cuda.is_available() if pin_cpu is None else pin_cpu
        self.sink: tuple[torch.Tensor, torch.Tensor] | None = None
        self.recent: tuple[torch.Tensor, torch.Tensor] | None = None
        self.old: dict[int, ResourceKey] = {}
        self.old_ranges: dict[int, tuple[int, int]] = {}
        self._next_chunk = 0
        self._recent_start = 0
        self._total_tokens = 0
        self._unwanted: set[ResourceKey] = set()

    def _register_old(self, key: torch.Tensor, value: torch.Tensor, start: int) -> int:
        chunk = self._next_chunk
        self._next_chunk += 1
        identity = kv_key(self.request_id, self.layer, chunk)
        payload = _cpu_payload(key, value, self.pin_cpu)
        size = sum(tensor.numel() * tensor.element_size() for tensor in payload)
        self.residency.register_cpu(identity, payload, size)
        self.old[chunk] = identity
        self.old_ranges[chunk] = (start, start + len(key))
        return chunk

    def initialize(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("KV tensors must have shape (tokens, kv_heads, head_dim)")
        if self.sink is not None:
            raise RuntimeError("KV cache is already initialized")
        tokens = len(key)
        sink_end = min(tokens, self.config.sink_tokens)
        old_tokens = max(0, tokens - sink_end - self.config.recent_tokens)
        old_tokens -= old_tokens % self.config.kv_chunk_tokens
        old_end = sink_end + old_tokens
        # Clone small resident windows so their views do not retain the full prefill tensor.
        self.sink = (key[:sink_end].contiguous().clone(), value[:sink_end].contiguous().clone())
        self.recent = (key[old_end:].contiguous().clone(), value[old_end:].contiguous().clone())
        self._recent_start = old_end
        self._total_tokens = tokens
        for start in range(sink_end, old_end, self.config.kv_chunk_tokens):
            end = start + self.config.kv_chunk_tokens
            self._register_old(key[start:end], value[start:end], start)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if self.recent is None or key.shape != value.shape or key.ndim != 3:
            raise ValueError("initialize first and append aligned 3-D KV tensors")
        recent_key = torch.cat((self.recent[0], key))
        recent_value = torch.cat((self.recent[1], value))
        limit = self.config.recent_tokens + self.config.kv_chunk_tokens - 1
        while len(recent_key) > limit:
            width = self.config.kv_chunk_tokens
            self._register_old(recent_key[:width], recent_value[:width], self._recent_start)
            recent_key, recent_value = recent_key[width:], recent_value[width:]
            self._recent_start += width
        self.recent = (recent_key, recent_value)
        self._total_tokens += len(key)

    def enqueue(
        self,
        draft_mass: dict[int, float],
        deadline: int,
        miss_cost_ms: float,
        *,
        consumer: str | None = None,
    ) -> list[tuple[ResourceKey, str]]:
        queued = []
        requests = []
        consumer = consumer or self.request_id
        for chunk, probability in draft_mass.items():
            if chunk not in self.old:
                continue
            requests.append(
                PrefetchRequest(
                    self.old[chunk], consumer, probability, deadline, miss_cost_ms
                )
            )
            queued.append((self.old[chunk], consumer))
        self.runtime.prefetch_many(requests)
        return queued

    def predicted_keep_set(self, draft_mass: dict[int, float]) -> set[int]:
        keep = set()
        cumulative = 0.0
        for chunk in sorted(draft_mass, key=draft_mass.get, reverse=True):
            if chunk not in self.old:
                continue
            keep.add(chunk)
            cumulative += draft_mass[chunk]
            if cumulative >= self.config.predicted_mass_threshold and len(keep) >= min(
                self.config.minimum_old_chunks, len(self.old)
            ):
                break
        return keep

    def retain_predicted(self, predictions: list[dict[int, float]]) -> None:
        keep = set().union(*(self.predicted_keep_set(values) for values in predictions))
        for chunk, identity in self.old.items():
            if chunk in keep:
                self._unwanted.discard(identity)
            elif not self.residency.evict(identity):
                self._unwanted.add(identity)
        self.reap_unwanted()

    def sparse_attention(
        self,
        query: torch.Tensor,
        draft_mass: dict[int, float],
        *,
        miss_cost_ms: float,
    ) -> SparseAttentionResult:
        if self.sink is None or self.recent is None:
            raise RuntimeError("KV cache is not initialized")
        always = [part for part in (self.sink, self.recent) if len(part[0])]
        partition = empty_partition(len(query), query.device)
        for key, _ in always:
            partition = update_partition(partition, chunk_logsumexp(query, key))
        controller = HybridStopController(
            self.config.predicted_mass_threshold,
            self.config.marginal_mass_threshold,
            self.config.marginal_patience,
            min(self.config.minimum_old_chunks, len(self.old)),
        )
        selected = []
        marginals = []
        selected_payloads: list[tuple[torch.Tensor, torch.Tensor]] = []
        for chunk in sorted(draft_mass, key=draft_mass.get, reverse=True):
            identity = self.old.get(chunk)
            if identity is None:
                continue
            key, value = self.runtime.demand(
                identity,
                consumer=self.request_id,
                miss_cost_ms=miss_cost_ms,
            )
            chunk_lse = chunk_logsumexp(query, key)
            marginal = mean_target_marginal(partition, chunk_lse)
            partition = update_partition(partition, chunk_lse)
            selected.append(chunk)
            marginals.append(marginal)
            selected_payloads.append((key, value))
            if controller.observe(draft_mass[chunk], marginal):
                break
        output = attention_output(query, always + selected_payloads)
        for chunk in selected:
            self.runtime.release(self.old[chunk])
        return SparseAttentionResult(
            output,
            selected,
            controller.predicted_mass,
            marginals,
        )

    def reconcile(
        self, next_draft_mass: dict[int, float], deadline: int, miss_cost_ms: float
    ) -> None:
        """Keep next-decode candidates; cancel or evict every other old chunk."""
        keep = self.predicted_keep_set(next_draft_mass)
        for chunk, identity in self.old.items():
            if chunk in keep:
                self.runtime.prefetch(
                    identity,
                    consumer=self.request_id,
                    probability=next_draft_mass[chunk],
                    deadline=deadline,
                    miss_cost_ms=miss_cost_ms,
                )
                self._unwanted.discard(identity)
            else:
                self.runtime.cancel(identity, self.request_id)
                if not self.residency.evict(identity):
                    self._unwanted.add(identity)
        self.reap_unwanted()

    def reap_unwanted(self) -> None:
        for identity in list(self._unwanted):
            if self.residency.state(identity) == ResourceState.GPU_RESIDENT:
                self.residency.evict(identity)
                self._unwanted.remove(identity)

    def close(self) -> None:
        """Release all request-private offloaded chunks."""
        for identity in self.old.values():
            self.runtime.drop(identity)
        self.old.clear()
        self.old_ranges.clear()
        self._unwanted.clear()
        self.sink = None
        self.recent = None
