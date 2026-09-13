from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, NamedTuple, Protocol

import torch

from src.runtime.memory_queue import (
    MemoryRequestQueue,
    ResourceKey,
    ResourceKind,
)
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
        self._expert_layout = False
        self._expert_width = 0
        self._free = list(reversed(range(capacity)))
        self._assigned: dict[ResourceKey, int] = {}
        self.copy_operations = 0

    @staticmethod
    def _joined_gate_up(payload: Any) -> torch.Tensor | None:
        gate, up = payload.gate, payload.up
        if not (gate.is_contiguous() and up.is_contiguous()):
            return None
        if gate.untyped_storage().data_ptr() != up.untyped_storage().data_ptr():
            return None
        if gate.storage_offset() + gate.numel() != up.storage_offset():
            return None
        return gate.as_strided(
            (gate.shape[0] + up.shape[0], *gate.shape[1:]),
            gate.stride(),
            gate.storage_offset(),
        )

    def _copy_into(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        destination.copy_(source, non_blocking=self.device.type == "cuda")
        self.copy_operations += 1

    @property
    def allocated(self) -> bool:
        return self._buffers is not None

    def initialize(self, payload: Any) -> None:
        """Allocate fixed storage without consuming a logical slot."""
        if self._buffers is None:
            self._initialize(payload)

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
        self._expert_layout = (
            set(tensors) == {"gate", "up", "down"}
            and tensors["gate"].shape == tensors["up"].shape
            and tensors["down"].shape == (tensors["gate"].shape[1], tensors["gate"].shape[0])
            and len({value.dtype for value in tensors.values()}) == 1
        )
        if self._expert_layout:
            self._expert_width = tensors["gate"].shape[0]
            self._buffers = {
                "gate_up": torch.empty(
                    (self.capacity, self._expert_width * 2, tensors["gate"].shape[1]),
                    dtype=tensors["gate"].dtype,
                    device=self.device,
                ),
                "down": torch.empty(
                    (self.capacity, *tensors["down"].shape),
                    dtype=tensors["down"].dtype,
                    device=self.device,
                ),
            }
        else:
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
        if self._expert_layout:
            gate = self._buffers["gate_up"][slot, : self._expert_width]
            up = self._buffers["gate_up"][slot, self._expert_width :]
            down = self._buffers["down"][slot]
            sources = (payload.gate, payload.up, payload.down)
            destinations = (gate, up, down)
            if any(
                source.shape != destination.shape or source.dtype != destination.dtype
                for source, destination in zip(sources, destinations)
            ):
                self._free.append(slot)
                raise ValueError("expert tensor shape or dtype changed after slot allocation")
            joined_gate_up = self._joined_gate_up(payload)
            if joined_gate_up is None:
                self._copy_into(gate, payload.gate)
                self._copy_into(up, payload.up)
            else:
                self._copy_into(self._buffers["gate_up"][slot], joined_gate_up)
            self._copy_into(down, payload.down)
            self._assigned[key] = slot
            return self._payload_type(gate=gate, up=up, down=down)
        values = {}
        for item in fields(payload):
            source = getattr(payload, item.name)
            if isinstance(source, torch.Tensor):
                destination = self._buffers[item.name][slot]
                if source.shape != destination.shape or source.dtype != destination.dtype:
                    self._free.append(slot)
                    raise ValueError("expert tensor shape or dtype changed after slot allocation")
                self._copy_into(destination, source)
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

    def slot_for(self, key: ResourceKey) -> int:
        return self._assigned[key]

    def fused_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._buffers is None or not self._expert_layout:
            raise RuntimeError("packed payload does not use the fused expert layout")
        return self._buffers["gate_up"], self._buffers["down"]


class ExpertSlotMap:
    """Persistent logical-to-physical expert maps, updated only on slot changes."""

    def __init__(
        self,
        device: str | torch.device,
        *,
        staging_capacity: int = 0,
    ) -> None:
        if staging_capacity < 0:
            raise ValueError("expert map staging capacity must be non-negative")
        self.device = torch.device(device)
        self._assignments: dict[int, dict[int, int]] = {}
        self._maps: dict[int, torch.Tensor] = {}
        self._sizes: dict[int, int] = {}
        self._pending_removals: dict[int, set[int]] = {}
        self._lock = threading.Lock()
        self._staging_capacity = 0
        self._host_indices: torch.Tensor | None = None
        self._host_slots: torch.Tensor | None = None
        self._device_indices: torch.Tensor | None = None
        self._device_slots: torch.Tensor | None = None
        if staging_capacity:
            self._allocate_staging(staging_capacity)

    def _allocate_staging(self, capacity: int) -> None:
        pin_memory = self.device.type == "cuda"
        self._host_indices = torch.empty(capacity, dtype=torch.long, pin_memory=pin_memory)
        self._host_slots = torch.empty(capacity, dtype=torch.int32, pin_memory=pin_memory)
        if pin_memory:
            self._device_indices = torch.empty(capacity, dtype=torch.long, device=self.device)
            self._device_slots = torch.empty(capacity, dtype=torch.int32, device=self.device)
        else:
            self._device_indices = self._host_indices
            self._device_slots = self._host_slots
        self._staging_capacity = capacity

    def _apply_staged_updates(
        self,
        mapping: torch.Tensor,
        updates: dict[int, int],
        offset: int,
    ) -> None:
        size = len(updates)
        if offset + size > self._staging_capacity:
            raise RuntimeError("expert map update exceeds prepared staging capacity")
        assert self._host_indices is not None and self._host_slots is not None
        assert self._device_indices is not None and self._device_slots is not None
        end = offset + size
        for position, (expert, slot) in enumerate(updates.items(), start=offset):
            self._host_indices[position] = expert
            self._host_slots[position] = slot
        indices = self._device_indices[offset:end]
        slots = self._device_slots[offset:end]
        if self.device.type == "cuda":
            indices.copy_(self._host_indices[offset:end], non_blocking=True)
            slots.copy_(self._host_slots[offset:end], non_blocking=True)
        mapping.index_copy_(0, indices, slots)

    @staticmethod
    def _validate_key(key: ResourceKey) -> None:
        if key.kind != ResourceKind.EXPERT or key.request_id:
            raise ValueError("expert slot maps require shared expert resource keys")

    def update(self, assignments: list[tuple[ResourceKey, int]]) -> None:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for key, slot in assignments:
            self._validate_key(key)
            if key.layer < 0 or key.object_id < 0 or slot < 0:
                raise ValueError("expert layer, ID, and slot must be non-negative")
            grouped.setdefault(key.layer, []).append((key.object_id, slot))
        with self._lock:
            for layer, updates in grouped.items():
                mapping = self._maps.get(layer)
                if mapping is not None and any(expert >= len(mapping) for expert, _ in updates):
                    raise ValueError("logical expert ID exceeds persistent map size")
            changed_layers = set(grouped) | set(self._pending_removals)
            for layer, updates in grouped.items():
                logical = self._assignments.setdefault(layer, {})
                logical.update(updates)
            staged_updates: list[tuple[torch.Tensor, dict[int, int]]] = []
            for layer in changed_layers:
                mapping = self._maps.get(layer)
                if mapping is None:
                    continue
                device_updates = {expert: -1 for expert in self._pending_removals.get(layer, set())}
                device_updates.update(dict(grouped.get(layer, [])))
                if not device_updates:
                    continue
                staged_updates.append((mapping, device_updates))
            total_updates = sum(len(updates) for _, updates in staged_updates)
            if total_updates > self._staging_capacity:
                self._allocate_staging(max(total_updates, max(1, self._staging_capacity * 2)))
            offset = 0
            for mapping, device_updates in staged_updates:
                self._apply_staged_updates(mapping, device_updates, offset)
                offset += len(device_updates)
            self._pending_removals.clear()

    def remove(self, key: ResourceKey, *, defer_device: bool = False) -> None:
        self._validate_key(key)
        with self._lock:
            self._assignments.get(key.layer, {}).pop(key.object_id, None)
            mapping = self._maps.get(key.layer)
            if mapping is not None and key.object_id < len(mapping):
                if defer_device:
                    self._pending_removals.setdefault(key.layer, set()).add(key.object_id)
                else:
                    mapping[key.object_id] = -1

    def get(self, layer: int, num_experts: int) -> torch.Tensor:
        if layer < 0 or num_experts <= 0:
            raise ValueError("layer must be non-negative and num_experts must be positive")
        with self._lock:
            existing = self._maps.get(layer)
            if existing is not None:
                if self._sizes[layer] != num_experts:
                    raise ValueError("global expert count changed for an initialized layer")
                pending = self._pending_removals.pop(layer, set())
                if pending:
                    indices = torch.tensor(list(pending), dtype=torch.long, device=self.device)
                    existing.index_fill_(0, indices, -1)
                return existing
            mapping = torch.full((num_experts,), -1, dtype=torch.int32, device=self.device)
            assignments = self._assignments.get(layer, {})
            if any(expert >= num_experts for expert in assignments):
                raise ValueError("logical expert ID exceeds persistent map size")
            if assignments:
                indices = torch.tensor(list(assignments), dtype=torch.long, device=self.device)
                slots = torch.tensor(
                    list(assignments.values()), dtype=torch.int32, device=self.device
                )
                mapping.index_copy_(0, indices, slots)
            self._maps[layer] = mapping
            self._sizes[layer] = num_experts
            return mapping


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
        self.expert_slot_maps = (
            ExpertSlotMap(self.device, staging_capacity=expert_slots * 2)
            if expert_slots is not None
            else None
        )
        self._use_events: dict[ResourceKey, torch.cuda.Event] = {}
        self._use_event_references: dict[torch.cuda.Event, int] = {}
        self._retired_use_events: list[torch.cuda.Event] = []
        self._use_event_lock = threading.Lock()
        self._waited_use_events: set[torch.cuda.Event] = set()

    def _retire_use_event_reference(self, event: torch.cuda.Event) -> None:
        references = self._use_event_references[event] - 1
        if references:
            self._use_event_references[event] = references
        else:
            del self._use_event_references[event]
            self._retired_use_events.append(event)

    def _acquire_use_event(self) -> torch.cuda.Event:
        for index, event in enumerate(self._retired_use_events):
            if event not in self._waited_use_events and event.query():
                self._retired_use_events.pop(index)
                return event
        return torch.cuda.Event()

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
        return self.copy_many_to_gpu([(key, cpu_value)])[0]

    def copy_many_to_gpu(self, items: list[tuple[ResourceKey, Any]]) -> list[Any]:
        acquired = []
        values = []
        with torch.cuda.stream(self.stream):
            try:
                for key, cpu_value in items:
                    if key.kind == ResourceKind.EXPERT and self.expert_slots is not None:
                        gpu_value = self.expert_slots.acquire(key, cpu_value)
                        acquired.append(key)
                    else:
                        gpu_value = self._copy(cpu_value)
                    values.append(gpu_value)
                if acquired:
                    assert self.expert_slots is not None and self.expert_slot_maps is not None
                    self.expert_slot_maps.update(
                        [(key, self.expert_slots.slot_for(key)) for key in acquired]
                    )
                self.stream.synchronize()
            except Exception:
                if self.expert_slots is not None:
                    for key in acquired:
                        self.expert_slots.release(key)
                raise
            finally:
                with self._use_event_lock:
                    self._waited_use_events.clear()
        return values

    def release_gpu(self, key: ResourceKey, gpu_value: Any) -> None:
        del gpu_value
        if key.kind == ResourceKind.EXPERT and self.expert_slots is not None:
            with self._use_event_lock:
                event = self._use_events.pop(key, None)
                if event is not None:
                    if event not in self._waited_use_events:
                        self.stream.wait_event(event)
                        self._waited_use_events.add(event)
                    self._retire_use_event_reference(event)
            assert self.expert_slot_maps is not None
            self.expert_slot_maps.remove(key, defer_device=True)
            self.expert_slots.release(key)

    def record_use(self, key: ResourceKey) -> None:
        """Prevent H2D slot reuse until compute-stream readers have finished."""
        self.record_uses([key])

    def record_uses(self, keys: list[ResourceKey]) -> None:
        """Protect one compute batch with a shared completion event."""
        experts = [
            key
            for key in dict.fromkeys(keys)
            if key.kind == ResourceKind.EXPERT and self.expert_slots is not None
        ]
        if not experts:
            return
        with self._use_event_lock:
            for key in experts:
                previous = self._use_events.pop(key, None)
                if previous is not None:
                    self._retire_use_event_reference(previous)
            event = self._acquire_use_event()
            event.record(torch.cuda.current_stream(self.device))
            self._use_event_references[event] = len(experts)
            for key in experts:
                self._use_events[key] = event

    def expert_slot(self, key: ResourceKey) -> int:
        if self.expert_slots is None:
            raise RuntimeError("fixed expert slots are disabled")
        return self.expert_slots.slot_for(key)

    def initialize_expert_storage(self, payload: Any) -> bool:
        if self.expert_slots is None:
            return False
        self.expert_slots.initialize(payload)
        return True

    def packed_expert_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.expert_slots is None:
            raise RuntimeError("fixed expert slots are disabled")
        return self.expert_slots.fused_weights()

    def expert_map(self, layer: int, num_experts: int) -> torch.Tensor:
        if self.expert_slot_maps is None:
            raise RuntimeError("fixed expert slots are disabled")
        return self.expert_slot_maps.get(layer, num_experts)


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
    prefetch_candidates: int = 0
    prefetch_budget_dropped: int = 0
    maximum_prefetch_candidates: int = 0
    maximum_prefetch_admitted: int = 0
    prefetch_enqueued: int = 0
    prefetch_batches: int = 0
    prefetch_enqueue_ms: float = 0.0
    transfer_batches: int = 0
    maximum_transfer_batch: int = 0

    def delta(self, earlier: TransferMetrics) -> TransferMetrics:
        """Return the per-field increase since an earlier snapshot."""
        return type(self)(
            **{
                item.name: getattr(self, item.name) - getattr(earlier, item.name)
                for item in fields(self)
            }
        )


class PrefetchRequest(NamedTuple):
    key: ResourceKey
    consumer: str
    probability: float
    deadline: int
    miss_cost_ms: float


class DemandRequest(NamedTuple):
    key: ResourceKey
    consumer: str
    miss_cost_ms: float


class TransferWorker:
    """Single owner of memory movement; compute never performs copies directly."""

    def __init__(
        self,
        queue: MemoryRequestQueue,
        residency: ResidencyManager,
        backend: TransferBackend,
        max_batch_size: int = 32,
        max_speculative_batch_size: int | None = None,
        max_speculative_batch_bytes: int | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("transfer batch size must be positive")
        if max_speculative_batch_size is not None and max_speculative_batch_size <= 0:
            raise ValueError("speculative transfer batch size must be positive")
        if max_speculative_batch_bytes is not None and max_speculative_batch_bytes <= 0:
            raise ValueError("speculative transfer batch bytes must be positive")
        self.queue = queue
        self.residency = residency
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_speculative_batch_size = max_speculative_batch_size
        self.max_speculative_batch_bytes = max_speculative_batch_bytes
        release_gpu = getattr(backend, "release_gpu", None)
        if callable(release_gpu):
            residency.set_eviction_callback(release_gpu)
        self.metrics = TransferMetrics()
        self._phase_maximum_transfer_batch = 0
        self._phase_maximum_prefetch_candidates = 0
        self._phase_maximum_prefetch_admitted = 0
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("transfer worker already started")
        self._thread = threading.Thread(target=self._run, name="specfetch-h2d", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            requests, current_step, batch_bytes = self.queue.pop_many_with_metadata(
                self.max_batch_size,
                block=True,
                speculative_maximum=self.max_speculative_batch_size,
                speculative_maximum_bytes=self.max_speculative_batch_bytes,
            )
            if not requests:
                return
            demand_batch = requests[0].demand
            if demand_batch:
                keys = [request.key for request in requests]
                admitted_requests = self.residency.begin_demand_transfers(
                    keys,
                    keys_are_unique=True,
                )
            else:
                admissions = []
                keys = []
                for request in requests:
                    mib = max(request.size_bytes / 2**20, 1e-6)
                    consumer_leases = {
                        consumer: (
                            request.miss_cost_ms
                            * probability
                            / max(
                                1,
                                request.consumer_deadlines[consumer] - current_step,
                            )
                            / mib,
                            request.consumer_deadlines[consumer],
                        )
                        for consumer, probability in request.consumer_probabilities.items()
                    }
                    admissions.append(
                        (
                            request.key,
                            request.priority(current_step),
                            request.deadline,
                            False,
                            consumer_leases,
                        )
                    )
                    keys.append(request.key)
                admitted_requests = self.residency.begin_transfers(admissions)
            if False not in admitted_requests:
                accepted = requests
                accepted_bytes = batch_bytes
            else:
                accepted = []
                accepted_keys = []
                accepted_bytes = 0
                for request, admitted in zip(requests, admitted_requests):
                    if admitted:
                        accepted.append(request)
                        accepted_keys.append(request.key)
                        accepted_bytes += request.size_bytes
                    elif not demand_batch:
                        self.metrics.dropped_speculative += 1
                keys = accepted_keys
            if not accepted:
                continue
            self.metrics.submitted += len(accepted)
            self.metrics.transfer_batches += 1
            self.metrics.maximum_transfer_batch = max(
                self.metrics.maximum_transfer_batch, len(accepted)
            )
            self._phase_maximum_transfer_batch = max(
                self._phase_maximum_transfer_batch, len(accepted)
            )
            start = time.perf_counter()
            try:
                items = self.residency.cpu_items(keys)
                copy_many = getattr(self.backend, "copy_many_to_gpu", None)
                if callable(copy_many):
                    values = copy_many(items)
                else:
                    values = [self.backend.copy_to_gpu(key, cpu_value) for key, cpu_value in items]
                if len(values) != len(accepted):
                    raise RuntimeError("transfer backend returned the wrong batch length")
                self.residency.complete_transfer_values(keys, values)
                accepted_count = len(accepted)
                self.metrics.completed += accepted_count
                self.metrics.bytes += accepted_bytes
                if demand_batch:
                    self.metrics.demand_transfers += accepted_count
                else:
                    self.metrics.speculative_transfers += accepted_count
            except Exception as error:  # noqa: BLE001 - surface backend failures to compute
                self._error = error
                release_gpu = getattr(self.backend, "release_gpu", None)
                for request in accepted:
                    if self.residency.state(request.key) == ResourceState.IN_FLIGHT:
                        if callable(release_gpu):
                            release_gpu(request.key, None)
                        self.metrics.failed += 1
                        self.residency.fail_transfer(request.key)
            finally:
                self.metrics.transfer_ms += (time.perf_counter() - start) * 1000
            if self._error is not None:
                return

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("transfer worker failed") from self._error

    def record_prefetch_admission(self, candidate_count: int, admitted_count: int) -> None:
        self.metrics.maximum_prefetch_candidates = max(
            self.metrics.maximum_prefetch_candidates, candidate_count
        )
        self.metrics.maximum_prefetch_admitted = max(
            self.metrics.maximum_prefetch_admitted, admitted_count
        )
        self._phase_maximum_prefetch_candidates = max(
            self._phase_maximum_prefetch_candidates, candidate_count
        )
        self._phase_maximum_prefetch_admitted = max(
            self._phase_maximum_prefetch_admitted, admitted_count
        )

    def metrics_snapshot(self) -> TransferMetrics:
        """Copy monotonic counters for phase-level reporting.

        A transfer completing across a phase boundary is charged to the phase in
        which its counter becomes visible. This preserves additive totals without
        synchronizing away intended transfer/compute overlap.
        """
        return replace(self.metrics)

    def phase_metrics_since(
        self, earlier: TransferMetrics
    ) -> tuple[TransferMetrics, TransferMetrics]:
        """Return a new total snapshot and an additive delta with a phase-local peak."""
        current = self.metrics_snapshot()
        delta = current.delta(earlier)
        delta.maximum_transfer_batch = self._phase_maximum_transfer_batch
        delta.maximum_prefetch_candidates = self._phase_maximum_prefetch_candidates
        delta.maximum_prefetch_admitted = self._phase_maximum_prefetch_admitted
        self._phase_maximum_transfer_batch = 0
        self._phase_maximum_prefetch_candidates = 0
        self._phase_maximum_prefetch_admitted = 0
        return current, delta

    def close(self, *, drain: bool = False) -> None:
        discarded = [] if drain else [request.key for request in self.queue.snapshot()]
        self.queue.close(discard=not drain)
        self.residency.unqueue_many(discarded)
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
        self.prefetch_many([PrefetchRequest(key, consumer, probability, deadline, miss_cost_ms)])

    def prefetch_many(
        self,
        requests: list[PrefetchRequest],
        *,
        candidate_count: int | None = None,
    ) -> None:
        """Submit one prediction group with a single queue lock acquisition."""
        if not requests:
            return
        candidate_count = len(requests) if candidate_count is None else candidate_count
        if candidate_count < len(requests):
            raise ValueError("candidate count cannot be smaller than admitted requests")
        start = time.perf_counter()
        self.worker.metrics.prefetch_batches += 1
        self.worker.metrics.prefetch_candidates += candidate_count
        self.worker.metrics.prefetch_budget_dropped += candidate_count - len(requests)
        self.worker.metrics.prefetch_requests += len(requests)
        self.worker.record_prefetch_admission(candidate_count, len(requests))
        for request in requests:
            if not 0 <= request.probability <= 1:
                raise ValueError("probability must be in [0, 1]")
            if request.miss_cost_ms < 0:
                raise ValueError("miss_cost_ms must be non-negative")
        current_step = self.queue.current_step
        queued, queue_sizes = self.residency.prepare_prefetches(
            requests,
            current_step=current_step,
        )
        self.queue.upsert_prefetches(queued, queue_sizes)
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
        return self.demand_many(
            [DemandRequest(key, consumer, miss_cost_ms)],
            timeout=timeout,
            keys_are_unique=True,
        )[key]

    def demand_many(
        self,
        requests: list[DemandRequest],
        *,
        timeout: float | None = None,
        keys_are_unique: bool = False,
    ) -> dict[ResourceKey, Any]:
        """Promote all dependencies before waiting so H2D can form a batch."""
        if not requests:
            return {}
        keys = []
        for request in requests:
            if request.miss_cost_ms < 0:
                raise ValueError("miss_cost_ms must be non-negative")
            keys.append(request.key)
        self.worker.metrics.demand_requests += len(requests)
        values, pending, queue_sizes, demand_hits = self.residency.prepare_demands(
            keys,
            keys_are_unique=keys_are_unique,
        )
        self.worker.metrics.demand_hits += demand_hits
        self.worker.metrics.demand_misses += len(requests) - demand_hits
        if not pending:
            self.worker.check()
            return values
        if queue_sizes:
            queued = []
            queued_sizes = []
            for request in requests:
                size_bytes = queue_sizes.get(request.key)
                if size_bytes is not None:
                    queued.append(request)
                    queued_sizes.append(size_bytes)
            if len(queued) == 1:
                self.queue.upsert_demand_intent(queued[0], queued_sizes[0])
            else:
                self.queue.upsert_demand_intents(queued, queued_sizes)
        start = time.perf_counter()
        completed = self.residency.wait_resident_many(
            pending,
            timeout,
            keys_are_unique=True,
        )
        self.worker.check()
        if completed is None:
            missing = next((key for key in pending if key not in values), pending[0])
            raise TimeoutError(f"resource did not become resident: {missing}")
        values.update(completed)
        self.worker.metrics.demand_wait_ms += (time.perf_counter() - start) * 1000
        return values

    def cancel(self, key: ResourceKey, consumer: str | None = None) -> bool:
        removed = self.queue.cancel(key, consumer)
        self.residency.cancel_lease(key, consumer)
        if removed and not self.queue.contains(key):
            self.residency.unqueue(key)
        return removed

    def cancel_many(self, cancellations: list[tuple[ResourceKey, str]]) -> None:
        """批量撤销同一预测窗口产生的队列项和驻留 lease。"""
        if not cancellations:
            return
        depleted = self.queue.cancel_many(cancellations)
        self.residency.cancel_leases(cancellations)
        self.residency.unqueue_many(depleted)

    def release(self, key: ResourceKey) -> None:
        self.release_many([key], keys_are_unique=True)

    def release_many(
        self,
        keys: list[ResourceKey],
        *,
        keys_are_unique: bool = False,
    ) -> None:
        if not keys:
            return
        record_uses = getattr(self.worker.backend, "record_uses", None)
        if callable(record_uses):
            record_uses(keys)
        else:
            record_use = getattr(self.worker.backend, "record_use", None)
            if callable(record_use):
                for key in keys:
                    record_use(key)
        self.residency.release_many(keys, keys_are_unique=keys_are_unique)

    def drop(self, key: ResourceKey, timeout: float | None = None) -> None:
        """Cancel and free a request-private resource at request completion."""
        removed = self.queue.cancel(key)
        if removed and not self.queue.contains(key):
            self.residency.unqueue(key)
        state = self.residency.state(key)
        if state == ResourceState.QUEUED:
            if not self.residency.wait_not_queued(key, timeout):
                raise TimeoutError(f"queued resource did not settle during drop: {key}")
            state = self.residency.state(key)
        if state == ResourceState.IN_FLIGHT:
            self.residency.wait_resident(key, timeout)
            self.worker.check()
            state = self.residency.state(key)
        if state == ResourceState.GPU_RESIDENT:
            self.residency.evict(key)
        self.residency.unregister_cpu(key)
