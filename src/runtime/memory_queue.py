from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Protocol


class ResourceKind(str, Enum):
    KV = "kv"
    EXPERT = "expert"


@dataclass(frozen=True)
class ResourceKey:
    """Identity of an offloaded object.

    KV keys use ``request_id`` and ``object_id=chunk_index``. Experts are shared
    across requests and therefore use an empty request ID and expert ID.
    """

    kind: ResourceKind
    layer: int
    object_id: int
    request_id: str = ""
    _cached_hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_cached_hash",
            hash((self.kind, self.layer, self.object_id, self.request_id)),
        )

    def __hash__(self) -> int:
        return self._cached_hash


@dataclass
class MemoryRequest:
    key: ResourceKey
    size_bytes: int
    miss_cost_ms: float
    deadline: int
    consumer_probabilities: dict[str, float] = field(default_factory=dict)
    consumer_deadlines: dict[str, int] = field(default_factory=dict)
    demand: bool = False
    version: int = 0
    expected_uses: float = 0.0

    def priority(self, current_step: int) -> float:
        if self.demand:
            return math.inf
        urgency = 1.0 / max(1, self.deadline - current_step)
        mib = max(self.size_bytes / 2**20, 1e-6)
        return self.miss_cost_ms * self.expected_uses * urgency / mib


class QueueUpdate(NamedTuple):
    """One consumer's contribution to a queued resource request."""

    key: ResourceKey
    consumer: str
    probability: float
    deadline: int
    size_bytes: int
    miss_cost_ms: float
    demand: bool = False


class PrefetchQueueIntent(Protocol):
    """Fields consumed when residency forwards an existing prefetch intent."""

    key: ResourceKey
    consumer: str
    probability: float
    deadline: int
    miss_cost_ms: float


class DemandQueueUpdate(NamedTuple):
    """Minimal queue fields needed by an unconditional demand miss."""

    key: ResourceKey
    size_bytes: int
    miss_cost_ms: float


class DemandQueueIntent(Protocol):
    """Fields consumed when demand admission forwards its existing intent."""

    key: ResourceKey
    miss_cost_ms: float


class MemoryRequestQueue:
    """Thread-safe updatable priority queue shared by KV and expert requests."""

    def __init__(self) -> None:
        self._requests: dict[ResourceKey, MemoryRequest] = {}
        self._heap: list[tuple[float, int, int, int, ResourceKey]] = []
        self._sequence = 0
        self._step = 0
        self._heap_dirty = False
        self._closed = False
        self._condition = threading.Condition()

    def __len__(self) -> int:
        with self._condition:
            return len(self._requests)

    @property
    def current_step(self) -> int:
        with self._condition:
            return self._step

    def contains(self, key: ResourceKey) -> bool:
        with self._condition:
            return key in self._requests

    def _push(self, request: MemoryRequest) -> None:
        request.version += 1
        self._sequence += 1
        heapq.heappush(
            self._heap,
            (
                -request.priority(self._step),
                request.deadline,
                self._sequence,
                request.version,
                request.key,
            ),
        )

    def _rebuild_heap_locked(self) -> None:
        rebuilt = []
        for request in self._requests.values():
            self._sequence += 1
            rebuilt.append(
                (
                    -request.priority(self._step),
                    request.deadline,
                    self._sequence,
                    request.version,
                    request.key,
                )
            )
        heapq.heapify(rebuilt)
        self._heap = rebuilt
        self._heap_dirty = False

    def upsert(
        self,
        key: ResourceKey,
        *,
        consumer: str,
        probability: float,
        deadline: int,
        size_bytes: int,
        miss_cost_ms: float,
        demand: bool = False,
    ) -> MemoryRequest:
        return self.upsert_many(
            [
                QueueUpdate(
                    key,
                    consumer,
                    probability,
                    deadline,
                    size_bytes,
                    miss_cost_ms,
                    demand,
                )
            ]
        )[0]

    @staticmethod
    def _validate(update: QueueUpdate) -> None:
        if not 0 <= update.probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if update.size_bytes <= 0 or update.miss_cost_ms < 0:
            raise ValueError("size_bytes must be positive and miss_cost_ms non-negative")

    def _merge_locked(
        self,
        update: PrefetchQueueIntent,
        size_bytes: int,
        *,
        demand: bool = False,
        update_aggregates: bool = False,
    ) -> MemoryRequest:
        """Merge fields that can be updated incrementally under the queue lock."""
        request = self._requests.get(update.key)
        if request is None:
            request = MemoryRequest(
                update.key,
                size_bytes,
                update.miss_cost_ms,
                update.deadline,
            )
            self._requests[update.key] = request
        elif request.size_bytes != size_bytes:
            raise ValueError(f"size changed for existing resource {update.key}")
        previous_probability = request.consumer_probabilities.get(update.consumer)
        previous_deadline = request.consumer_deadlines.get(update.consumer)
        request.consumer_probabilities[update.consumer] = update.probability
        request.consumer_deadlines[update.consumer] = update.deadline
        if update_aggregates:
            request.expected_uses += update.probability - (
                previous_probability if previous_probability is not None else 0.0
            )
            if previous_deadline is None:
                request.deadline = (
                    update.deadline
                    if len(request.consumer_deadlines) == 1
                    else min(request.deadline, update.deadline)
                )
            elif update.deadline <= request.deadline:
                request.deadline = update.deadline
            elif previous_deadline == request.deadline:
                request.deadline = min(request.consumer_deadlines.values())
        request.miss_cost_ms = max(request.miss_cost_ms, update.miss_cost_ms)
        request.demand = request.demand or demand
        return request

    def upsert_many(self, updates: list[QueueUpdate]) -> list[MemoryRequest]:
        """Apply a prediction batch under one lock and wake the worker once."""
        if not updates:
            return []
        for update in updates:
            self._validate(update)
        with self._condition:
            if self._closed:
                raise RuntimeError("queue is closed")
            sizes: dict[ResourceKey, int] = {}
            for update in updates:
                expected_size = sizes.setdefault(update.key, update.size_bytes)
                existing = self._requests.get(update.key)
                if expected_size != update.size_bytes or (
                    existing is not None and existing.size_bytes != update.size_bytes
                ):
                    raise ValueError(f"size changed for existing resource {update.key}")
            requests = []
            unique: dict[ResourceKey, MemoryRequest] = {}
            for update in updates:
                request = self._merge_locked(
                    update,
                    update.size_bytes,
                    demand=update.demand,
                )
                requests.append(request)
                unique[request.key] = request
            for request in unique.values():
                request.expected_uses = sum(request.consumer_probabilities.values())
                request.deadline = min(request.consumer_deadlines.values())
                self._push(request)
            self._condition.notify()
            return requests

    def upsert_prefetches(
        self,
        intents: list[PrefetchQueueIntent],
        size_bytes: list[int],
    ) -> list[MemoryRequest]:
        """Queue existing prefetch intents without allocating forwarding records."""
        if not intents:
            if size_bytes:
                raise ValueError("prefetch intents and sizes must have equal lengths")
            return []
        if len(intents) != len(size_bytes):
            raise ValueError("prefetch intents and sizes must have equal lengths")
        for intent, size in zip(intents, size_bytes):
            if not 0 <= intent.probability <= 1:
                raise ValueError("probability must be in [0, 1]")
            if size <= 0 or intent.miss_cost_ms < 0:
                raise ValueError("size_bytes must be positive and miss_cost_ms non-negative")
        with self._condition:
            if self._closed:
                raise RuntimeError("queue is closed")
            sizes: dict[ResourceKey, int] = {}
            for intent, size in zip(intents, size_bytes):
                expected_size = sizes.setdefault(intent.key, size)
                existing = self._requests.get(intent.key)
                if expected_size != size or (existing is not None and existing.size_bytes != size):
                    raise ValueError(f"size changed for existing resource {intent.key}")
            requests = []
            unique: dict[ResourceKey, MemoryRequest] = {}
            update_aggregates = len(sizes) == len(intents)
            for intent, size in zip(intents, size_bytes):
                request = self._merge_locked(
                    intent,
                    size,
                    update_aggregates=update_aggregates,
                )
                requests.append(request)
                unique[request.key] = request
            if not update_aggregates:
                for request in unique.values():
                    request.expected_uses = sum(request.consumer_probabilities.values())
                    request.deadline = min(request.consumer_deadlines.values())
            for request in unique.values():
                self._push(request)
            self._condition.notify()
            return requests

    def upsert_demands(self, updates: list[DemandQueueUpdate]) -> list[MemoryRequest]:
        """Atomically promote demands without unused speculative consumer maps."""
        if not updates:
            return []
        for update in updates:
            if update.size_bytes <= 0 or update.miss_cost_ms < 0:
                raise ValueError("size_bytes must be positive and miss_cost_ms non-negative")
        with self._condition:
            if self._closed:
                raise RuntimeError("queue is closed")
            sizes: dict[ResourceKey, int] = {}
            for update in updates:
                expected_size = sizes.setdefault(update.key, update.size_bytes)
                existing = self._requests.get(update.key)
                if expected_size != update.size_bytes or (
                    existing is not None and existing.size_bytes != update.size_bytes
                ):
                    raise ValueError(f"size changed for existing resource {update.key}")
            requests = []
            unique: dict[ResourceKey, MemoryRequest] = {}
            for update in updates:
                request = self._requests.get(update.key)
                if request is None:
                    request = MemoryRequest(
                        update.key,
                        update.size_bytes,
                        update.miss_cost_ms,
                        self._step,
                        demand=True,
                    )
                    self._requests[update.key] = request
                else:
                    request.miss_cost_ms = max(request.miss_cost_ms, update.miss_cost_ms)
                    request.deadline = min(request.deadline, self._step)
                    request.demand = True
                requests.append(request)
                unique[request.key] = request
            for request in unique.values():
                self._push(request)
            self._condition.notify()
            return requests

    def upsert_demand_intents(
        self,
        intents: list[DemandQueueIntent],
        size_bytes: list[int],
    ) -> list[MemoryRequest]:
        """Promote existing demand intents without allocating forwarding records."""
        if not intents:
            if size_bytes:
                raise ValueError("demand intents and sizes must have equal lengths")
            return []
        if len(intents) != len(size_bytes):
            raise ValueError("demand intents and sizes must have equal lengths")
        for intent, size in zip(intents, size_bytes):
            if size <= 0 or intent.miss_cost_ms < 0:
                raise ValueError("size_bytes must be positive and miss_cost_ms non-negative")
        with self._condition:
            if self._closed:
                raise RuntimeError("queue is closed")
            sizes: dict[ResourceKey, int] = {}
            for intent, size in zip(intents, size_bytes):
                expected_size = sizes.setdefault(intent.key, size)
                existing = self._requests.get(intent.key)
                if expected_size != size or (existing is not None and existing.size_bytes != size):
                    raise ValueError(f"size changed for existing resource {intent.key}")
            requests = []
            unique: dict[ResourceKey, MemoryRequest] = {}
            for intent, size in zip(intents, size_bytes):
                request = self._requests.get(intent.key)
                if request is None:
                    request = MemoryRequest(
                        intent.key,
                        size,
                        intent.miss_cost_ms,
                        self._step,
                        demand=True,
                    )
                    self._requests[intent.key] = request
                else:
                    request.miss_cost_ms = max(request.miss_cost_ms, intent.miss_cost_ms)
                    request.deadline = min(request.deadline, self._step)
                    request.demand = True
                requests.append(request)
                unique[request.key] = request
            for request in unique.values():
                self._push(request)
            self._condition.notify()
            return requests

    def upsert_demand_intent(
        self,
        intent: DemandQueueIntent,
        size_bytes: int,
    ) -> MemoryRequest:
        """Promote one existing demand intent without batch bookkeeping."""
        if size_bytes <= 0 or intent.miss_cost_ms < 0:
            raise ValueError("size_bytes must be positive and miss_cost_ms non-negative")
        with self._condition:
            if self._closed:
                raise RuntimeError("queue is closed")
            request = self._requests.get(intent.key)
            if request is None:
                request = MemoryRequest(
                    intent.key,
                    size_bytes,
                    intent.miss_cost_ms,
                    self._step,
                    demand=True,
                )
                self._requests[intent.key] = request
            elif request.size_bytes != size_bytes:
                raise ValueError(f"size changed for existing resource {intent.key}")
            else:
                request.miss_cost_ms = max(request.miss_cost_ms, intent.miss_cost_ms)
                request.deadline = min(request.deadline, self._step)
                request.demand = True
            self._push(request)
            self._condition.notify()
            return request

    def promote_demand(
        self,
        key: ResourceKey,
        *,
        consumer: str,
        size_bytes: int,
        miss_cost_ms: float,
    ) -> MemoryRequest:
        del consumer
        return self.upsert_demands([DemandQueueUpdate(key, size_bytes, miss_cost_ms)])[0]

    def set_step(self, step: int) -> None:
        """Advance logical time; the transfer worker rebuilds urgency lazily."""
        with self._condition:
            if step < self._step:
                raise ValueError("queue step cannot move backwards")
            if step != self._step:
                self._step = step
                self._heap_dirty = True
                self._condition.notify_all()

    def cancel(self, key: ResourceKey, consumer: str | None = None) -> bool:
        with self._condition:
            request = self._requests.get(key)
            if request is None:
                return False
            if consumer is None:
                del self._requests[key]
            else:
                request.consumer_probabilities.pop(consumer, None)
                request.consumer_deadlines.pop(consumer, None)
                if request.demand:
                    return True
                if request.consumer_probabilities:
                    request.expected_uses = sum(request.consumer_probabilities.values())
                    request.deadline = min(request.consumer_deadlines.values())
                    self._push(request)
                else:
                    del self._requests[key]
            return True

    def cancel_many(self, cancellations: list[tuple[ResourceKey, str]]) -> set[ResourceKey]:
        """批量移除 consumer，每个受影响资源最多重建一次堆项。"""
        grouped: dict[ResourceKey, set[str]] = {}
        for key, consumer in cancellations:
            grouped.setdefault(key, set()).add(consumer)
        depleted = set()
        with self._condition:
            for key, consumers in grouped.items():
                request = self._requests.get(key)
                if request is None:
                    continue
                for consumer in consumers:
                    request.consumer_probabilities.pop(consumer, None)
                    request.consumer_deadlines.pop(consumer, None)
                if request.demand:
                    continue
                if request.consumer_probabilities:
                    request.expected_uses = sum(request.consumer_probabilities.values())
                    request.deadline = min(request.consumer_deadlines.values())
                    self._push(request)
                else:
                    del self._requests[key]
                    depleted.add(key)
        return depleted

    def pop(self, block: bool = False) -> MemoryRequest | None:
        batch = self.pop_many(1, block=block)
        return batch[0] if batch else None

    def _peek_valid_locked(self) -> MemoryRequest | None:
        if self._heap_dirty:
            self._rebuild_heap_locked()
        while self._heap:
            _, _, _, version, key = self._heap[0]
            request = self._requests.get(key)
            if request is not None and request.version == version:
                return request
            heapq.heappop(self._heap)
        return None

    def _pop_valid_locked(self) -> MemoryRequest | None:
        request = self._peek_valid_locked()
        if request is None:
            return None
        heapq.heappop(self._heap)
        del self._requests[request.key]
        return request

    def pop_many(
        self,
        maximum: int,
        block: bool = False,
        *,
        speculative_maximum: int | None = None,
        speculative_maximum_bytes: int | None = None,
    ) -> list[MemoryRequest]:
        requests, _ = self.pop_many_with_step(
            maximum,
            block,
            speculative_maximum=speculative_maximum,
            speculative_maximum_bytes=speculative_maximum_bytes,
        )
        return requests

    def pop_many_with_step(
        self,
        maximum: int,
        block: bool = False,
        *,
        speculative_maximum: int | None = None,
        speculative_maximum_bytes: int | None = None,
    ) -> tuple[list[MemoryRequest], int]:
        """Pop one same-class transfer batch without mixing demand and speculation."""
        if maximum <= 0:
            raise ValueError("maximum batch size must be positive")
        if speculative_maximum is not None and speculative_maximum <= 0:
            raise ValueError("speculative maximum batch size must be positive")
        if speculative_maximum_bytes is not None and speculative_maximum_bytes <= 0:
            raise ValueError("speculative maximum batch bytes must be positive")
        with self._condition:
            while True:
                first = self._pop_valid_locked()
                if first is not None:
                    requests = [first]
                    limit = (
                        maximum
                        if first.demand or speculative_maximum is None
                        else min(maximum, speculative_maximum)
                    )
                    batch_bytes = first.size_bytes
                    while len(requests) < limit:
                        next_request = self._peek_valid_locked()
                        if next_request is None or next_request.demand != first.demand:
                            break
                        if (
                            not first.demand
                            and speculative_maximum_bytes is not None
                            and batch_bytes + next_request.size_bytes > speculative_maximum_bytes
                        ):
                            break
                        requests.append(self._pop_valid_locked())
                        batch_bytes += next_request.size_bytes
                    return requests, self._step
                if not block or self._closed:
                    return [], self._step
                self._condition.wait()

    def snapshot(self) -> list[MemoryRequest]:
        with self._condition:
            return sorted(
                self._requests.values(),
                key=lambda item: (-item.priority(self._step), item.deadline),
            )

    def close(self, *, discard: bool = False) -> None:
        with self._condition:
            self._closed = True
            if discard:
                self._requests.clear()
                self._heap.clear()
                self._heap_dirty = False
            self._condition.notify_all()
