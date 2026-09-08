from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Protocol

import torch

from src.runtime.memory_queue import MemoryRequestQueue, QueueUpdate, ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState


class TransferBackend(Protocol):
    def copy_to_gpu(self, key: ResourceKey, cpu_value: Any) -> Any: ...


class PackedExpertSlots:
    """Fixed GPU storage for identically-shaped expert dataclass payloads."""

    def __init__(self, capacity: int, device: str | torch.device) -> None:
        if capacity <= 0:
            raise ValueError("expert slot capacity must be positive")
        self.capacity = capacity
        self.device = torch.device(device)
        self._buffers: dict[str, torch.Tensor] | None = None
        self._payload_type: type | None = None
        self._free = list(reversed(range(capacity)))
        self._assigned: dict[ResourceKey, int] = {}

    @property
    def allocated(self) -> bool:
        return self._buffers is not None

    def _initialize(self, payload: Any) -> None:
        if not is_dataclass(payload):
            raise TypeError("packed expert payload must be a dataclass")
        tensors = {
            item.name: getattr(payload, item.name)
            for item in fields(payload)
            if isinstance(getattr(payload, item.name), torch.Tensor)
        }
        if not tensors:
            raise TypeError("packed expert payload has no tensor fields")
        self._payload_type = type(payload)
        self._buffers = {
            name: torch.empty(
                (self.capacity, *value.shape), dtype=value.dtype, device=self.device
            )
            for name, value in tensors.items()
        }

    def acquire(self, key: ResourceKey, payload: Any) -> Any:
        if key in self._assigned:
            raise RuntimeError(f"expert already owns a packed slot: {key}")
        if self._buffers is None:
            self._initialize(payload)
        assert self._payload_type is not None and self._buffers is not None
        if type(payload) is not self._payload_type:
            raise TypeError("expert payload type changed after slot allocation")
        if not self._free:
            raise RuntimeError("no free packed expert slot")
        slot = self._free.pop()
        values = {}
        for item in fields(payload):
            source = getattr(payload, item.name)
            if isinstance(source, torch.Tensor):
                destination = self._buffers[item.name][slot]
                if source.shape != destination.shape or source.dtype != destination.dtype:
                    self._free.append(slot)
                    raise ValueError("expert tensor shape or dtype changed after slot allocation")
                destination.copy_(source, non_blocking=self.device.type == "cuda")
                values[item.name] = destination
            else:
                values[item.name] = source
        self._assigned[key] = slot
        return self._payload_type(**values)

    def release(self, key: ResourceKey) -> bool:
        slot = self._assigned.pop(key, None)
        if slot is None:
            return False
        self._free.append(slot)
        return True


class CudaTransferBackend:
    """Copy nested tensor payloads on one dedicated CUDA stream."""

    def __init__(
        self,
        device: str | torch.device = "cuda:0",
        *,
        expert_slots: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.expert_slots = (
            PackedExpertSlots(expert_slots, self.device) if expert_slots is not None else None
        )

    def _copy(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, tuple):
            return tuple(self._copy(item) for item in value)
        if isinstance(value, dict):
            return {name: self._copy(item) for name, item in value.items()}
        if is_dataclass(value):
            return type(value)(
                **{item.name: self._copy(getattr(value, item.name)) for item in fields(value)}
            )
        raise TypeError(f"unsupported transfer payload {type(value)}")

    def copy_to_gpu(self, key: ResourceKey, cpu_value: Any) -> Any:
        with torch.cuda.stream(self.stream):
            if key.kind == ResourceKind.EXPERT and self.expert_slots is not None:
                gpu_value = self.expert_slots.acquire(key, cpu_value)
            else:
                gpu_value = self._copy(cpu_value)
        self.stream.synchronize()
        return gpu_value

    def release_gpu(self, key: ResourceKey, gpu_value: Any) -> None:
        del gpu_value
        if key.kind == ResourceKind.EXPERT and self.expert_slots is not None:
            self.expert_slots.release(key)


@dataclass
class TransferMetrics:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    bytes: int = 0
    transfer_ms: float = 0.0
    demand_wait_ms: float = 0.0
    speculative_transfers: int = 0
    demand_transfers: int = 0
    dropped_speculative: int = 0
    demand_requests: int = 0
    demand_hits: int = 0
    demand_misses: int = 0
    prefetch_requests: int = 0
    prefetch_enqueued: int = 0
    prefetch_batches: int = 0
    prefetch_enqueue_ms: float = 0.0

    def delta(self, earlier: TransferMetrics) -> TransferMetrics:
        """Return the per-field increase since an earlier snapshot."""
        return type(self)(
            **{
                item.name: getattr(self, item.name) - getattr(earlier, item.name)
                for item in fields(self)
            }
        )


@dataclass(frozen=True)
class PrefetchRequest:
    key: ResourceKey
    consumer: str
    probability: float
    deadline: int
    miss_cost_ms: float


class TransferWorker:
    """Single owner of memory movement; compute never performs copies directly."""

    def __init__(
        self,
        queue: MemoryRequestQueue,
        residency: ResidencyManager,
        backend: TransferBackend,
    ) -> None:
        self.queue = queue
        self.residency = residency
        self.backend = backend
        release_gpu = getattr(backend, "release_gpu", None)
        if callable(release_gpu):
            residency.set_eviction_callback(release_gpu)
        self.metrics = TransferMetrics()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("transfer worker already started")
        self._thread = threading.Thread(target=self._run, name="specfetch-h2d", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            request = self.queue.pop(block=True)
            if request is None:
                return
            if not self.residency.begin_transfer(
                request.key,
                priority=request.priority(self.queue.current_step),
                deadline=request.deadline,
                demand=request.demand,
            ):
                if not request.demand:
                    self.metrics.dropped_speculative += 1
                continue
            self.metrics.submitted += 1
            start = time.perf_counter()
            try:
                record = self.residency.record(request.key)
                value = self.backend.copy_to_gpu(request.key, record.cpu_value)
                self.residency.complete_transfer(request.key, value)
                self.metrics.completed += 1
                self.metrics.bytes += request.size_bytes
                if request.demand:
                    self.metrics.demand_transfers += 1
                else:
                    self.metrics.speculative_transfers += 1
            except Exception as error:  # noqa: BLE001 - surface backend failures to compute
                self._error = error
                self.metrics.failed += 1
                self.residency.fail_transfer(request.key)
            finally:
                self.metrics.transfer_ms += (time.perf_counter() - start) * 1000

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("transfer worker failed") from self._error

    def metrics_snapshot(self) -> TransferMetrics:
        """Copy monotonic counters for phase-level reporting.

        A transfer completing across a phase boundary is charged to the phase in
        which its counter becomes visible. This preserves additive totals without
        synchronizing away intended transfer/compute overlap.
        """
        return replace(self.metrics)

    def close(self, *, drain: bool = False) -> None:
        self.queue.close(discard=not drain)
        if self._thread is not None:
            self._thread.join()
        self.check()


class OffloadRuntime:
    """Compute-facing facade for speculative requests and blocking demand misses."""

    def __init__(
        self,
        queue: MemoryRequestQueue,
        residency: ResidencyManager,
        worker: TransferWorker,
    ) -> None:
        self.queue = queue
        self.residency = residency
        self.worker = worker

    def prefetch(
        self,
        key: ResourceKey,
        *,
        consumer: str,
        probability: float,
        deadline: int,
        miss_cost_ms: float,
    ) -> None:
        self.prefetch_many(
            [PrefetchRequest(key, consumer, probability, deadline, miss_cost_ms)]
        )

    def prefetch_many(
        self,
        requests: list[PrefetchRequest],
    ) -> None:
        """Submit one prediction group with a single queue lock acquisition."""
        if not requests:
            return
        start = time.perf_counter()
        self.worker.metrics.prefetch_batches += 1
        self.worker.metrics.prefetch_requests += len(requests)
        queued: list[QueueUpdate] = []
        leases: dict[ResourceKey, tuple[float, int, float]] = {}
        for request in requests:
            key = request.key
            if not 0 <= request.probability <= 1:
                raise ValueError("probability must be in [0, 1]")
            if request.miss_cost_ms < 0:
                raise ValueError("miss_cost_ms must be non-negative")
            state = self.residency.state(key)
            record = self.residency.record(key)
            if state in (ResourceState.GPU_RESIDENT, ResourceState.IN_FLIGHT):
                expected_uses, earliest, largest_cost = leases.get(
                    key, (0.0, request.deadline, request.miss_cost_ms)
                )
                leases[key] = (
                    expected_uses + request.probability,
                    min(earliest, request.deadline),
                    max(largest_cost, request.miss_cost_ms),
                )
                continue
            self.residency.mark_queued(key)
            queued.append(
                QueueUpdate(
                    key,
                    request.consumer,
                    request.probability,
                    request.deadline,
                    record.size_bytes,
                    request.miss_cost_ms,
                )
            )
        current_step = self.queue.current_step
        for key, (expected_uses, deadline, miss_cost_ms) in leases.items():
            record = self.residency.record(key)
            urgency = 1 / max(1, deadline - current_step)
            priority = (
                miss_cost_ms
                * expected_uses
                * urgency
                / max(record.size_bytes / 2**20, 1e-6)
            )
            self.residency.update_lease(key, priority, deadline)
        self.queue.upsert_many(queued)
        self.worker.metrics.prefetch_enqueued += len(queued)
        self.worker.metrics.prefetch_enqueue_ms += (time.perf_counter() - start) * 1000

    def demand(
        self,
        key: ResourceKey,
        *,
        consumer: str,
        miss_cost_ms: float,
        timeout: float | None = None,
    ) -> Any:
        self.worker.metrics.demand_requests += 1
        state = self.residency.state(key)
        if state == ResourceState.GPU_RESIDENT:
            self.worker.metrics.demand_hits += 1
            return self.residency.get_gpu(key)
        self.worker.metrics.demand_misses += 1
        record = self.residency.record(key)
        if state != ResourceState.IN_FLIGHT:
            self.residency.mark_queued(key)
            self.queue.promote_demand(
                key,
                consumer=consumer,
                size_bytes=record.size_bytes,
                miss_cost_ms=miss_cost_ms,
            )
        start = time.perf_counter()
        ready = self.residency.wait_resident(key, timeout)
        self.worker.metrics.demand_wait_ms += (time.perf_counter() - start) * 1000
        self.worker.check()
        if not ready:
            raise TimeoutError(f"resource did not become resident: {key}")
        return self.residency.get_gpu(key)

    def cancel(self, key: ResourceKey, consumer: str | None = None) -> bool:
        removed = self.queue.cancel(key, consumer)
        if removed and not self.queue.contains(key):
            self.residency.unqueue(key)
        return removed

    def release(self, key: ResourceKey) -> None:
        self.residency.release(key)
