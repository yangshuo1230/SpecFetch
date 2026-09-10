from __future__ import annotations

import heapq
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from src.runtime.memory_queue import QueueUpdate, ResourceKey, ResourceKind


class ResourceState(str, Enum):
    CPU_ONLY = "cpu_only"
    QUEUED = "queued"
    IN_FLIGHT = "in_flight"
    GPU_RESIDENT = "gpu_resident"


class CacheFullError(RuntimeError):
    pass


class PrefetchIntent(Protocol):
    key: ResourceKey
    consumer: str
    probability: float
    deadline: int
    miss_cost_ms: float


@dataclass
class ResourceRecord:
    key: ResourceKey
    cpu_value: Any
    size_bytes: int
    state: ResourceState = ResourceState.CPU_ONLY
    gpu_value: Any = None
    pinned: bool = False
    priority: float = 0.0
    deadline: int = 0
    speculative: bool = False
    used: bool = False
    demand_active: bool = False
    demand_count: int = 0
    consumer_leases: dict[str, tuple[float, int]] = field(default_factory=dict)
    lease_priority: float = 0.0
    lease_deadline: int = 0


class ResidencyManager:
    """Own resource state and LRU eviction independently of transfer policy."""

    def __init__(self, capacities: dict[ResourceKind, int]):
        if any(value <= 0 for value in capacities.values()):
            raise ValueError("all cache capacities must be positive")
        self.capacities = dict(capacities)
        self._records: dict[ResourceKey, ResourceRecord] = {}
        self._resident: dict[ResourceKind, OrderedDict[ResourceKey, None]] = {
            kind: OrderedDict() for kind in capacities
        }
        self._reserved: dict[ResourceKind, set[ResourceKey]] = {kind: set() for kind in capacities}
        self._resident_layer_counts: dict[ResourceKind, dict[int, int]] = {
            kind: {} for kind in capacities
        }
        self._condition = threading.Condition()
        self._eviction_callback: Callable[[ResourceKey, Any], None] | None = None
        self.evictions = 0
        self.wasted_prefetches = 0

    def set_eviction_callback(self, callback: Callable[[ResourceKey, Any], None] | None) -> None:
        """Register storage cleanup used by fixed-slot transfer backends."""
        with self._condition:
            self._eviction_callback = callback

    def register_cpu(
        self, key: ResourceKey, value: Any, size_bytes: int, *, pinned: bool = False
    ) -> None:
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        with self._condition:
            existing = self._records.get(key)
            if existing and existing.state != ResourceState.CPU_ONLY:
                raise RuntimeError(f"cannot replace active resource {key}")
            self._records[key] = ResourceRecord(key, value, size_bytes, pinned=pinned)

    def record(self, key: ResourceKey) -> ResourceRecord:
        with self._condition:
            try:
                return self._records[key]
            except KeyError as error:
                raise KeyError(f"unregistered resource {key}") from error

    def contains(self, key: ResourceKey) -> bool:
        with self._condition:
            return key in self._records

    def unregister_cpu(self, key: ResourceKey) -> None:
        """Forget an inactive CPU source after its owning request completes."""
        with self._condition:
            record = self._records[key]
            if record.state != ResourceState.CPU_ONLY:
                raise RuntimeError(f"cannot unregister active resource {key}")
            del self._records[key]

    def state(self, key: ResourceKey) -> ResourceState:
        return self.record(key).state

    def mark_queued(self, key: ResourceKey) -> bool:
        with self._condition:
            record = self._records[key]
            if record.state == ResourceState.CPU_ONLY:
                record.state = ResourceState.QUEUED
                self._condition.notify_all()
                return True
            return record.state == ResourceState.QUEUED

    def prepare_demands(
        self, keys: list[ResourceKey]
    ) -> tuple[dict[ResourceKey, Any], list[ResourceKey], dict[ResourceKey, int]]:
        """Acquire resident hits and promote misses under one residency lock."""
        values: dict[ResourceKey, Any] = {}
        pending: list[ResourceKey] = []
        queue_sizes: dict[ResourceKey, int] = {}
        transitioned = False
        with self._condition:
            for key in dict.fromkeys(keys):
                record = self._records[key]
                record.demand_count += 1
                if record.state == ResourceState.GPU_RESIDENT:
                    record.demand_active = True
                    record.used = True
                    record.priority = float("inf")
                    self._resident[key.kind].move_to_end(key)
                    values[key] = record.gpu_value
                    continue
                pending.append(key)
                if record.state == ResourceState.IN_FLIGHT:
                    record.demand_active = True
                    record.used = True
                    record.priority = float("inf")
                    continue
                if record.state == ResourceState.CPU_ONLY:
                    record.state = ResourceState.QUEUED
                    transitioned = True
                queue_sizes[key] = record.size_bytes
            if transitioned:
                self._condition.notify_all()
        return values, pending, queue_sizes

    def prepare_prefetches(
        self,
        requests: list[PrefetchIntent],
        *,
        current_step: int,
    ) -> list[QueueUpdate]:
        """Prepare one speculative batch under one residency lock.

        CPU-only resources become queued and are returned for queue admission.
        Queued, resident and in-flight records retain the current consumer leases so
        cancellation stays linearizable while a worker claims a popped request.
        """
        queued: list[QueueUpdate] = []
        transitioned = False
        with self._condition:
            for request in requests:
                key = request.key
                try:
                    record = self._records[key]
                except KeyError as error:
                    raise KeyError(f"unregistered resource {key}") from error
                urgency = 1 / max(1, request.deadline - current_step)
                mib = max(record.size_bytes / 2**20, 1e-6)
                priority = request.miss_cost_ms * request.probability * urgency / mib
                if record.state in (
                    ResourceState.QUEUED,
                    ResourceState.GPU_RESIDENT,
                    ResourceState.IN_FLIGHT,
                ):
                    self._set_lease(record, request.consumer, priority, request.deadline)
                    record.speculative = True
                    record.used = False
                    if record.state in (
                        ResourceState.GPU_RESIDENT,
                        ResourceState.IN_FLIGHT,
                    ):
                        continue
                if record.state == ResourceState.CPU_ONLY:
                    record.state = ResourceState.QUEUED
                    transitioned = True
                    self._set_lease(record, request.consumer, priority, request.deadline)
                    record.speculative = True
                    record.used = False
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
            if transitioned:
                self._condition.notify_all()
        return queued

    def unqueue(self, key: ResourceKey) -> bool:
        return key in self.unqueue_many([key])

    def unqueue_many(self, keys: list[ResourceKey] | set[ResourceKey]) -> set[ResourceKey]:
        """Return queued resources to CPU-only state under one lock."""
        transitioned = set()
        with self._condition:
            for key in keys:
                record = self._records.get(key)
                if record is None or record.state != ResourceState.QUEUED:
                    continue
                record.state = ResourceState.CPU_ONLY
                record.consumer_leases.clear()
                record.speculative = False
                record.used = False
                record.lease_priority = 0.0
                record.lease_deadline = 0
                record.priority = 0.0
                record.deadline = 0
                transitioned.add(key)
            if transitioned:
                self._condition.notify_all()
        return transitioned

    def _eviction_candidate(
        self, kind: ResourceKind, protected: set[ResourceKey]
    ) -> ResourceKey | None:
        lru = self._resident[kind]
        victim = None
        victim_rank: tuple[float, int, int, int, int] | None = None
        layer_counts = self._resident_layer_counts[kind]
        for lru_order, key in enumerate(lru):
            record = self._records[key]
            if key in protected or record.pinned:
                continue
            # Sequential MoE decode scans every layer once per token. Plain
            # global LRU thrashes when that cyclic working set is larger than
            # the cache. Among equally urgent experts, evict from the most
            # represented layer first so every layer retains reusable entries.
            layer_balance = -layer_counts.get(key.layer, 0) if kind == ResourceKind.EXPERT else 0
            rank = (
                record.priority,
                -record.deadline,
                layer_balance,
                record.demand_count,
                lru_order,
            )
            if victim_rank is None or rank < victim_rank:
                victim = key
                victim_rank = rank
        return victim

    def _eviction_candidates(
        self,
        kind: ResourceKind,
        protected: set[ResourceKey],
        maximum: int,
    ) -> list[ResourceKey]:
        """Plan sequentially equivalent victims without rescanning the LRU."""
        if maximum <= 0:
            return []
        groups: dict[
            tuple[float, int],
            dict[int, list[tuple[int, int, ResourceKey]]],
        ] = {}
        layer_counts = dict(self._resident_layer_counts[kind])
        for lru_order, key in enumerate(self._resident[kind]):
            record = self._records[key]
            if key in protected or record.pinned:
                continue
            primary_rank = (record.priority, -record.deadline)
            layers = groups.setdefault(primary_rank, {})
            heapq.heappush(
                layers.setdefault(key.layer, []),
                (record.demand_count, lru_order, key),
            )
        victims = []
        for primary_rank in sorted(groups):
            layers = groups[primary_rank]
            layer_heap = []
            for layer, candidates in layers.items():
                top = candidates[0]
                balance = -layer_counts[layer] if kind == ResourceKind.EXPERT else 0
                layer_heap.append((balance, top[0], top[1], layer))
            heapq.heapify(layer_heap)
            while layer_heap and len(victims) < maximum:
                _, _, _, layer = heapq.heappop(layer_heap)
                _, _, victim = heapq.heappop(layers[layer])
                victims.append(victim)
                layer_counts[layer] -= 1
                if layers[layer]:
                    top = layers[layer][0]
                    balance = -layer_counts[layer] if kind == ResourceKind.EXPERT else 0
                    heapq.heappush(layer_heap, (balance, top[0], top[1], layer))
            if len(victims) == maximum:
                break
        return victims

    @staticmethod
    def _refresh_priority(record: ResourceRecord) -> None:
        record.lease_priority = sum(priority for priority, _ in record.consumer_leases.values())
        record.lease_deadline = min(
            (deadline for _, deadline in record.consumer_leases.values()),
            default=0,
        )
        record.priority = float("inf") if record.demand_active else record.lease_priority
        record.deadline = record.lease_deadline

    @staticmethod
    def _set_lease(
        record: ResourceRecord,
        consumer: str,
        priority: float,
        deadline: int,
    ) -> None:
        previous = record.consumer_leases.get(consumer)
        record.consumer_leases[consumer] = (priority, deadline)
        record.lease_priority += priority - (previous[0] if previous is not None else 0.0)
        if previous is None:
            record.lease_deadline = (
                deadline
                if len(record.consumer_leases) == 1
                else min(record.lease_deadline, deadline)
            )
        elif deadline <= record.lease_deadline:
            record.lease_deadline = deadline
        elif previous[1] == record.lease_deadline:
            record.lease_deadline = min(value[1] for value in record.consumer_leases.values())
        record.priority = float("inf") if record.demand_active else record.lease_priority
        record.deadline = record.lease_deadline

    @staticmethod
    def _remove_lease(record: ResourceRecord, consumer: str) -> None:
        previous = record.consumer_leases.pop(consumer, None)
        if previous is None:
            return
        if not record.consumer_leases:
            record.lease_priority = 0.0
            record.lease_deadline = 0
        else:
            record.lease_priority -= previous[0]
            if previous[1] == record.lease_deadline:
                record.lease_deadline = min(value[1] for value in record.consumer_leases.values())
        record.priority = float("inf") if record.demand_active else record.lease_priority
        record.deadline = record.lease_deadline

    def _evict_victim(self, kind: ResourceKind, victim: ResourceKey) -> None:
        lru = self._resident[kind]
        lru.pop(victim)
        record = self._records[victim]
        self._decrement_layer_count(victim)
        if record.speculative and not record.used:
            self.wasted_prefetches += 1
        if self._eviction_callback is not None:
            self._eviction_callback(victim, record.gpu_value)
        record.gpu_value = None
        record.state = ResourceState.CPU_ONLY
        self.evictions += 1

    def _decrement_layer_count(self, key: ResourceKey) -> None:
        counts = self._resident_layer_counts[key.kind]
        remaining = counts[key.layer] - 1
        if remaining:
            counts[key.layer] = remaining
        else:
            del counts[key.layer]

    def _begin_transfer_locked(
        self,
        key: ResourceKey,
        *,
        priority: float = 0.0,
        deadline: int = 0,
        demand: bool = False,
        consumer_leases: dict[str, tuple[float, int]] | None = None,
        protected: set[ResourceKey] | None = None,
    ) -> tuple[bool, bool]:
        record = self._records[key]
        if record.state in (ResourceState.GPU_RESIDENT, ResourceState.IN_FLIGHT):
            return False, False
        effective_consumer_leases = consumer_leases
        if not demand and record.state == ResourceState.QUEUED:
            if not record.consumer_leases:
                record.state = ResourceState.CPU_ONLY
                record.speculative = False
                record.used = False
                self._refresh_priority(record)
                return False, True
            effective_consumer_leases = dict(record.consumer_leases)
            if consumer_leases is not None:
                for consumer in effective_consumer_leases.keys() & consumer_leases.keys():
                    effective_consumer_leases[consumer] = consumer_leases[consumer]
            priority = sum(value for value, _ in effective_consumer_leases.values())
            deadline = min(value for _, value in effective_consumer_leases.values())
        kind = key.kind
        used = len(self._resident[kind]) + len(self._reserved[kind])
        if used >= self.capacities[kind]:
            effective_protected = {key} if protected is None else protected | {key}
            victim = self._eviction_candidate(kind, effective_protected)
            if victim is None:
                raise CacheFullError(f"no evictable {kind.value} cache slot")
            if not demand and self._records[victim].priority >= priority:
                record.state = ResourceState.CPU_ONLY
                return False, True
            self._evict_victim(kind, victim)
        self._reserved[kind].add(key)
        record.state = ResourceState.IN_FLIGHT
        record.priority = priority
        record.deadline = deadline
        record.speculative = not demand
        record.used = demand
        record.demand_active = demand
        if demand:
            record.consumer_leases.clear()
            record.lease_priority = 0.0
            record.lease_deadline = 0
            record.priority = float("inf")
            record.deadline = 0
        elif effective_consumer_leases is None:
            record.consumer_leases = {"__transfer__": (priority, deadline)}
            self._refresh_priority(record)
        else:
            record.consumer_leases = dict(effective_consumer_leases)
            self._refresh_priority(record)
        return True, True

    def begin_transfer(
        self,
        key: ResourceKey,
        *,
        priority: float = 0.0,
        deadline: int = 0,
        demand: bool = False,
        consumer_leases: dict[str, tuple[float, int]] | None = None,
        protected: set[ResourceKey] | None = None,
    ) -> bool:
        with self._condition:
            accepted, changed = self._begin_transfer_locked(
                key,
                priority=priority,
                deadline=deadline,
                demand=demand,
                consumer_leases=consumer_leases,
                protected=protected,
            )
            if changed:
                self._condition.notify_all()
            return accepted

    def _prepare_demand_evictions_locked(self, keys: list[ResourceKey]) -> bool | None:
        """Apply a complete victim plan, or return None to request scalar fallback."""
        eligible_by_kind: dict[ResourceKind, set[ResourceKey]] = {}
        try:
            for key in keys:
                if self._records[key].state not in (
                    ResourceState.GPU_RESIDENT,
                    ResourceState.IN_FLIGHT,
                ):
                    eligible_by_kind.setdefault(key.kind, set()).add(key)
        except KeyError:
            return None
        victim_plan = []
        for kind, eligible in eligible_by_kind.items():
            used = len(self._resident[kind]) + len(self._reserved[kind])
            required = max(0, used + len(eligible) - self.capacities[kind])
            victims = self._eviction_candidates(kind, set(), required)
            if len(victims) != required:
                return None
            victim_plan.extend((kind, victim) for victim in victims)
        for kind, victim in victim_plan:
            self._evict_victim(kind, victim)
        return bool(victim_plan)

    def begin_demand_transfers(self, keys: list[ResourceKey]) -> list[bool]:
        """Admit one unconditional demand batch without generic admission tuples."""
        if not keys:
            return []
        changed = False
        results = []
        with self._condition:
            try:
                planned_change = self._prepare_demand_evictions_locked(keys)
                if planned_change is not None:
                    changed = planned_change
                    for key in keys:
                        record = self._records[key]
                        if record.state in (
                            ResourceState.GPU_RESIDENT,
                            ResourceState.IN_FLIGHT,
                        ):
                            results.append(False)
                            continue
                        self._reserved[key.kind].add(key)
                        record.state = ResourceState.IN_FLIGHT
                        record.speculative = False
                        record.used = True
                        record.demand_active = True
                        record.consumer_leases.clear()
                        record.lease_priority = 0.0
                        record.lease_deadline = 0
                        record.priority = float("inf")
                        record.deadline = 0
                        changed = True
                        results.append(True)
                else:
                    for key in keys:
                        accepted, item_changed = self._begin_transfer_locked(key, demand=True)
                        changed = changed or item_changed
                        results.append(accepted)
            finally:
                if changed:
                    self._condition.notify_all()
        return results

    def begin_transfers(
        self,
        admissions: list[
            tuple[
                ResourceKey,
                float,
                int,
                bool,
                dict[str, tuple[float, int]] | None,
            ]
        ],
    ) -> list[bool]:
        """Apply one worker admission batch under one residency lock."""
        if not admissions:
            return []
        changed = False
        results = []
        with self._condition:
            try:
                if all(demand for _, _, _, demand, _ in admissions):
                    planned_change = self._prepare_demand_evictions_locked(
                        [key for key, _, _, _, _ in admissions]
                    )
                    if planned_change is not None:
                        changed = planned_change
                for key, priority, deadline, demand, consumer_leases in admissions:
                    accepted, item_changed = self._begin_transfer_locked(
                        key,
                        priority=priority,
                        deadline=deadline,
                        demand=demand,
                        consumer_leases=consumer_leases,
                    )
                    changed = changed or item_changed
                    results.append(accepted)
            finally:
                if changed:
                    self._condition.notify_all()
        return results

    def complete_transfer(self, key: ResourceKey, gpu_value: Any) -> None:
        self.complete_transfers([(key, gpu_value)])

    def cpu_values(self, keys: list[ResourceKey]) -> list[Any]:
        """Read one transfer batch's CPU payloads under one residency lock."""
        with self._condition:
            return [self._records[key].cpu_value for key in keys]

    def complete_transfers(self, items: list[tuple[ResourceKey, Any]]) -> None:
        """Publish one completed transfer batch atomically and notify once."""
        if not items:
            return
        with self._condition:
            records = [self._records[key] for key, _ in items]
            invalid = next(
                (record.state for record in records if record.state != ResourceState.IN_FLIGHT),
                None,
            )
            if invalid is not None:
                raise RuntimeError(f"transfer completed from invalid state {invalid}")
            for (key, gpu_value), record in zip(items, records):
                self._reserved[key.kind].remove(key)
                record.gpu_value = gpu_value
                record.state = ResourceState.GPU_RESIDENT
                self._resident[key.kind][key] = None
                counts = self._resident_layer_counts[key.kind]
                counts[key.layer] = counts.get(key.layer, 0) + 1
            self._condition.notify_all()

    def fail_transfer(self, key: ResourceKey) -> None:
        with self._condition:
            record = self._records[key]
            self._reserved[key.kind].discard(key)
            record.state = ResourceState.CPU_ONLY
            self._condition.notify_all()

    def get_gpu(self, key: ResourceKey) -> Any:
        with self._condition:
            record = self._records[key]
            if record.state != ResourceState.GPU_RESIDENT:
                raise KeyError(f"resource is not GPU resident: {key}")
            lru = self._resident[key.kind]
            lru.move_to_end(key)
            record.used = True
            return record.gpu_value

    def set_pinned(self, key: ResourceKey, pinned: bool) -> None:
        with self._condition:
            self._records[key].pinned = pinned

    def update_lease(
        self,
        key: ResourceKey,
        priority: float,
        deadline: int,
        consumer: str,
    ) -> None:
        with self._condition:
            record = self._records[key]
            self._set_lease(record, consumer, priority, deadline)
            record.speculative = True
            record.used = False

    def cancel_lease(self, key: ResourceKey, consumer: str | None = None) -> None:
        with self._condition:
            record = self._records[key]
            if consumer is None:
                record.consumer_leases.clear()
                record.lease_priority = 0.0
                record.lease_deadline = 0
                record.priority = float("inf") if record.demand_active else 0.0
                record.deadline = 0
            else:
                self._remove_lease(record, consumer)

    def cancel_leases(self, cancellations: list[tuple[ResourceKey, str]]) -> None:
        """在一次驻留锁内批量撤销 consumer lease。"""
        grouped: dict[ResourceKey, set[str]] = {}
        for key, consumer in cancellations:
            grouped.setdefault(key, set()).add(consumer)
        with self._condition:
            for key, consumers in grouped.items():
                record = self._records[key]
                for consumer in consumers:
                    self._remove_lease(record, consumer)

    def mark_demand(self, key: ResourceKey) -> None:
        with self._condition:
            record = self._records[key]
            record.demand_active = True
            record.used = True
            record.priority = float("inf")

    def release(self, key: ResourceKey) -> None:
        self.release_many([key])

    def release_many(self, keys: list[ResourceKey]) -> None:
        """End one compute batch's demand scopes under one residency lock."""
        with self._condition:
            for key in dict.fromkeys(keys):
                record = self._records[key]
                record.demand_active = False
                record.priority = record.lease_priority
                record.deadline = record.lease_deadline

    def evict(self, key: ResourceKey) -> bool:
        with self._condition:
            record = self._records[key]
            if record.state != ResourceState.GPU_RESIDENT or record.pinned:
                return False
            self._resident[key.kind].pop(key, None)
            self._decrement_layer_count(key)
            if record.speculative and not record.used:
                self.wasted_prefetches += 1
            if self._eviction_callback is not None:
                self._eviction_callback(key, record.gpu_value)
            record.gpu_value = None
            record.state = ResourceState.CPU_ONLY
            self.evictions += 1
            return True

    def wait_resident(self, key: ResourceKey, timeout: float | None = None) -> bool:
        with self._condition:
            return (
                self._condition.wait_for(
                    lambda: (
                        self._records[key].state
                        in (ResourceState.GPU_RESIDENT, ResourceState.CPU_ONLY)
                    ),
                    timeout=timeout,
                )
                and self._records[key].state == ResourceState.GPU_RESIDENT
            )

    def wait_resident_many(
        self, keys: list[ResourceKey], timeout: float | None = None
    ) -> dict[ResourceKey, Any] | None:
        """Wait once for a demand batch and acquire all completed GPU values."""
        unique = list(dict.fromkeys(keys))
        if not unique:
            return {}
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    any(self._records[key].state == ResourceState.CPU_ONLY for key in unique)
                    or all(self._records[key].state == ResourceState.GPU_RESIDENT for key in unique)
                ),
                timeout=timeout,
            )
            if not ready or any(
                self._records[key].state != ResourceState.GPU_RESIDENT for key in unique
            ):
                return None
            values = {}
            for key in unique:
                record = self._records[key]
                record.used = True
                self._resident[key.kind].move_to_end(key)
                values[key] = record.gpu_value
            return values

    def wait_not_queued(self, key: ResourceKey, timeout: float | None = None) -> bool:
        with self._condition:
            return self._condition.wait_for(
                lambda: self._records[key].state != ResourceState.QUEUED,
                timeout=timeout,
            )

    def resident_keys(self, kind: ResourceKind | None = None) -> set[ResourceKey]:
        with self._condition:
            if kind is not None:
                return set(self._resident[kind])
            return set().union(*(set(values) for values in self._resident.values()))
