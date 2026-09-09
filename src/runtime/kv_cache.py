from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from src.runtime.config import RuntimeConfig
from src.runtime.hybrid_attention import (
    HybridStopController,
    attention_output,
    chunk_logsumexp,
    empty_partition,
    mean_target_marginal,
    sequence_target_marginals,
    update_partition,
)
from src.runtime.memory_queue import ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState
from src.runtime.transfer import DemandRequest, OffloadRuntime, PrefetchRequest


def kv_key(request_id: str, layer: int, chunk: int) -> ResourceKey:
    return ResourceKey(ResourceKind.KV, layer=layer, object_id=chunk, request_id=request_id)


@dataclass
class SparseAttentionResult:
    output: torch.Tensor
    selected_old_chunks: list[int]
    predicted_mass: float
    target_marginals: list[float]
    target_mass_coverage: float | None = None
    relative_l2_error: float | None = None
    cosine_similarity: float | None = None
    shadow_seconds: float = 0.0
    threshold_sweep: dict[str, dict[str, float]] | None = None


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
        self._resident_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
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
        if self.config.kv_storage == "resident":
            capacity = self.config.resident_kv_capacity_tokens
            assert capacity is not None
            if tokens > capacity:
                raise ValueError(
                    f"prefix has {tokens} tokens but resident KV capacity is {capacity}"
                )
            buffers = tuple(
                torch.empty((capacity, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
                for tensor in (key, value)
            )
            for buffer, tensor in zip(buffers, (key, value)):
                buffer[:tokens].copy_(tensor)
            self._resident_buffers = buffers
            # Keep the public resident window view usable by diagnostics while avoiding
            # duplicate prefix storage. Resident mode never creates offloaded chunks.
            self.sink = tuple(buffer[:0] for buffer in buffers)
            self.recent = tuple(buffer[:tokens] for buffer in buffers)
            self._recent_start = 0
            self._total_tokens = tokens
            return
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

    def initialize_resident(
        self,
        buffers: tuple[torch.Tensor, torch.Tensor],
        tokens: int,
    ) -> None:
        """Bind a row of a batch-owned resident allocation without copying its prefix."""
        if self.config.kv_storage != "resident":
            raise RuntimeError("external resident buffers require resident KV storage")
        if self.sink is not None:
            raise RuntimeError("KV cache is already initialized")
        capacity = self.config.resident_kv_capacity_tokens
        assert capacity is not None
        if tokens < 0 or tokens > capacity:
            raise ValueError(f"resident token count {tokens} exceeds capacity {capacity}")
        if any(buffer.ndim != 3 or len(buffer) != capacity for buffer in buffers):
            raise ValueError("resident buffers must have shape (capacity, kv_heads, head_dim)")
        if buffers[0].shape != buffers[1].shape:
            raise ValueError("resident key and value buffers must have matching shapes")
        self._resident_buffers = buffers
        self.sink = tuple(buffer[:0] for buffer in buffers)
        self.recent = tuple(buffer[:tokens] for buffer in buffers)
        self._recent_start = 0
        self._total_tokens = tokens

    def commit_resident_append(self, tokens: int = 1) -> None:
        """Publish tokens already written by the owning batch allocation."""
        if self._resident_buffers is None or tokens <= 0:
            raise RuntimeError("resident KV cache must be initialized before advancing")
        end = self._total_tokens + tokens
        if end > len(self._resident_buffers[0]):
            raise RuntimeError(
                f"resident KV capacity {len(self._resident_buffers[0])} exceeded by token {end}"
            )
        self._total_tokens = end
        self.recent = tuple(buffer[:end] for buffer in self._resident_buffers)

    def rebind_resident(self, buffers: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Rebind after an owning batch compacts rows for completed requests."""
        if self._resident_buffers is None:
            raise RuntimeError("only an initialized resident cache can be rebound")
        if any(
            buffer.shape != self._resident_buffers[index].shape
            for index, buffer in enumerate(buffers)
        ):
            raise ValueError("replacement resident buffers must preserve the cache shape")
        self._resident_buffers = buffers
        self.sink = tuple(buffer[:0] for buffer in buffers)
        self.recent = tuple(buffer[: self._total_tokens] for buffer in buffers)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if self.recent is None or key.shape != value.shape or key.ndim != 3:
            raise ValueError("initialize first and append aligned 3-D KV tensors")
        if self._resident_buffers is not None:
            end = self._total_tokens + len(key)
            if end > len(self._resident_buffers[0]):
                raise RuntimeError(
                    f"resident KV capacity {len(self._resident_buffers[0])} exceeded by token {end}"
                )
            for buffer, tensor in zip(self._resident_buffers, (key, value)):
                buffer[self._total_tokens : end].copy_(tensor)
            self._total_tokens = end
            self.recent = tuple(buffer[:end] for buffer in self._resident_buffers)
            return
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
        requests, queued = self.prefetch_requests(
            draft_mass,
            deadline,
            miss_cost_ms,
            consumer=consumer,
        )
        self.runtime.prefetch_many(requests)
        return queued

    def prefetch_requests(
        self,
        draft_mass: dict[int, float],
        deadline: int,
        miss_cost_ms: float,
        *,
        consumer: str | None = None,
    ) -> tuple[list[PrefetchRequest], list[tuple[ResourceKey, str]]]:
        """构造 KV 预测请求，供调用方跨请求、跨层合并提交。"""
        queued = []
        requests = []
        consumer = consumer or self.request_id
        for chunk, probability in draft_mass.items():
            if chunk not in self.old:
                continue
            requests.append(
                PrefetchRequest(self.old[chunk], consumer, probability, deadline, miss_cost_ms)
            )
            queued.append((self.old[chunk], consumer))
        return requests, queued

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

    def ordered_old_chunks(self, draft_mass: dict[int, float]) -> list[int]:
        return [
            chunk
            for chunk in sorted(draft_mass, key=draft_mass.get, reverse=True)
            if chunk in self.old
        ]

    def guaranteed_chunks(self, draft_mass: dict[int, float]) -> list[int]:
        """Return the Draft prefix required before Target marginal stopping can apply."""
        guaranteed = []
        predicted_mass = 0.0
        for chunk in self.ordered_old_chunks(draft_mass):
            guaranteed.append(chunk)
            predicted_mass = min(1.0, predicted_mass + draft_mass[chunk])
            if predicted_mass >= self.config.predicted_mass_threshold and len(guaranteed) >= min(
                self.config.minimum_old_chunks, len(self.old)
            ):
                break
        return guaranteed

    def guaranteed_demand_requests(
        self, draft_mass: dict[int, float], miss_cost_ms: float
    ) -> list[DemandRequest]:
        return [
            DemandRequest(self.old[chunk], self.request_id, miss_cost_ms)
            for chunk in self.guaranteed_chunks(draft_mass)
        ]

    def sparse_attention(
        self,
        query: torch.Tensor,
        draft_mass: dict[int, float],
        *,
        miss_cost_ms: float,
        guaranteed_payloads: dict[ResourceKey, tuple[torch.Tensor, torch.Tensor]] | None = None,
        shadow: bool = False,
        shadow_thresholds: tuple[float, ...] = (),
    ) -> SparseAttentionResult:
        if self.sink is None or self.recent is None:
            raise RuntimeError("KV cache is not initialized")
        if self._resident_buffers is not None:
            key, value = (buffer[: self._total_tokens] for buffer in self._resident_buffers)
            q = query[None, :, None]
            k = key.transpose(0, 1)[None]
            v = value.transpose(0, 1)[None]
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                enable_gqa=q.shape[1] != k.shape[1],
            )[0, :, 0]
            return SparseAttentionResult(output, [], 1.0, [])
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
        ordered_chunks = self.ordered_old_chunks(draft_mass)
        guaranteed = self.guaranteed_chunks(draft_mass)
        if guaranteed_payloads is None:
            guaranteed_payloads = self.runtime.demand_many(
                self.guaranteed_demand_requests(draft_mass, miss_cost_ms)
            )
        guaranteed_values = [guaranteed_payloads[self.old[chunk]] for chunk in guaranteed]
        guaranteed_lses = [chunk_logsumexp(query, key) for key, _ in guaranteed_values]
        partition, guaranteed_marginals = sequence_target_marginals(partition, guaranteed_lses)
        stopped = False
        for chunk, payload, marginal in zip(guaranteed, guaranteed_values, guaranteed_marginals):
            selected.append(chunk)
            marginals.append(marginal)
            selected_payloads.append(payload)
            stopped = controller.observe(draft_mass[chunk], marginal)
        for chunk in ordered_chunks[len(guaranteed) :]:
            if stopped:
                break
            identity = self.old[chunk]
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
            stopped = controller.observe(draft_mass[chunk], marginal)
        output = attention_output(query, always + selected_payloads)
        coverage = None
        relative_l2 = None
        cosine = None
        shadow_seconds = 0.0
        threshold_sweep = None
        if shadow or shadow_thresholds:
            shadow_start = time.perf_counter()
            cpu_query = query.detach().float().cpu()
            cpu_always = [
                (key.detach().float().cpu(), value.detach().float().cpu()) for key, value in always
            ]
            cpu_old = {
                chunk: tuple(
                    tensor.detach().float().cpu()
                    for tensor in self.residency.record(identity).cpu_value
                )
                for chunk, identity in self.old.items()
            }
            full_chunks = cpu_always + list(cpu_old.values())
            selected_chunks = cpu_always + [cpu_old[chunk] for chunk in selected]
            exact = attention_output(cpu_query, full_chunks)
            approximate = output.detach().float().cpu()
            difference = torch.linalg.vector_norm(approximate - exact)
            relative_l2 = float(difference / torch.linalg.vector_norm(exact).clamp_min(1e-12))
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    approximate.reshape(1, -1), exact.reshape(1, -1)
                )[0]
            )
            full_partition = empty_partition(len(cpu_query))
            selected_partition = empty_partition(len(cpu_query))
            for key, _ in full_chunks:
                full_partition = update_partition(full_partition, chunk_logsumexp(cpu_query, key))
            for key, _ in selected_chunks:
                selected_partition = update_partition(
                    selected_partition, chunk_logsumexp(cpu_query, key)
                )
            coverage = float(torch.exp(selected_partition - full_partition).mean())
            threshold_sweep = {}
            ordered_chunks = [
                chunk
                for chunk in sorted(draft_mass, key=draft_mass.get, reverse=True)
                if chunk in cpu_old
            ]
            for threshold in shadow_thresholds:
                sweep_controller = HybridStopController(
                    threshold,
                    self.config.marginal_mass_threshold,
                    self.config.marginal_patience,
                    min(self.config.minimum_old_chunks, len(self.old)),
                )
                sweep_partition = empty_partition(len(cpu_query))
                for key, _ in cpu_always:
                    sweep_partition = update_partition(
                        sweep_partition, chunk_logsumexp(cpu_query, key)
                    )
                sweep_selected = []
                for chunk in ordered_chunks:
                    chunk_lse = chunk_logsumexp(cpu_query, cpu_old[chunk][0])
                    marginal = mean_target_marginal(sweep_partition, chunk_lse)
                    sweep_partition = update_partition(sweep_partition, chunk_lse)
                    sweep_selected.append(chunk)
                    if sweep_controller.observe(draft_mass[chunk], marginal):
                        break
                sweep_chunks = cpu_always + [cpu_old[chunk] for chunk in sweep_selected]
                sweep_output = attention_output(cpu_query, sweep_chunks)
                sweep_difference = torch.linalg.vector_norm(sweep_output - exact)
                label = f"{threshold:.4g}"
                threshold_sweep[label] = {
                    "selected_old_chunks": float(len(sweep_selected)),
                    "target_mass_coverage": float(
                        torch.exp(sweep_partition - full_partition).mean()
                    ),
                    "relative_l2_error": float(
                        sweep_difference / torch.linalg.vector_norm(exact).clamp_min(1e-12)
                    ),
                    "cosine_similarity": float(
                        torch.nn.functional.cosine_similarity(
                            sweep_output.reshape(1, -1), exact.reshape(1, -1)
                        )[0]
                    ),
                }
            shadow_seconds = time.perf_counter() - shadow_start
        for chunk in selected:
            self.runtime.release(self.old[chunk])
        return SparseAttentionResult(
            output,
            selected,
            controller.predicted_mass,
            marginals,
            coverage,
            relative_l2,
            cosine,
            shadow_seconds,
            threshold_sweep,
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
        self._resident_buffers = None
