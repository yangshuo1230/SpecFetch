from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol

import torch

from src.runtime.memory_queue import MemoryRequestQueue, ResourceKey
from src.runtime.residency import ResidencyManager, ResourceState


class TransferBackend(Protocol):
    def copy_to_gpu(self, key: ResourceKey, cpu_value: Any) -> Any: ...


class CudaTransferBackend:
    """Copy nested tensor payloads on one dedicated CUDA stream."""

    def __init__(self, device: str | torch.device = "cuda:0") -> None:
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)

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
        del key
        with torch.cuda.stream(self.stream):
            gpu_value = self._copy(cpu_value)
        self.stream.synchronize()
        return gpu_value


@dataclass
class TransferMetrics:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    bytes: int = 0
    transfer_ms: float = 0.0
    demand_wait_ms: float = 0.0


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
            if not self.residency.begin_transfer(request.key):
                continue
            self.metrics.submitted += 1
            start = time.perf_counter()
            try:
                record = self.residency.record(request.key)
                value = self.backend.copy_to_gpu(request.key, record.cpu_value)
                self.residency.complete_transfer(request.key, value)
                self.metrics.completed += 1
                self.metrics.bytes += request.size_bytes
            except Exception as error:  # noqa: BLE001 - surface backend failures to compute
                self._error = error
                self.metrics.failed += 1
                self.residency.fail_transfer(request.key)
            finally:
                self.metrics.transfer_ms += (time.perf_counter() - start) * 1000

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("transfer worker failed") from self._error

    def close(self) -> None:
        self.queue.close()
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
        state = self.residency.state(key)
        if state in (ResourceState.GPU_RESIDENT, ResourceState.IN_FLIGHT):
            return
        record = self.residency.record(key)
        self.residency.mark_queued(key)
        self.queue.upsert(
            key,
            consumer=consumer,
            probability=probability,
            deadline=deadline,
            size_bytes=record.size_bytes,
            miss_cost_ms=miss_cost_ms,
        )

    def demand(
        self,
        key: ResourceKey,
        *,
        consumer: str,
        miss_cost_ms: float,
        timeout: float | None = None,
    ) -> Any:
        state = self.residency.state(key)
        if state == ResourceState.GPU_RESIDENT:
            return self.residency.get_gpu(key)
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
