from __future__ import annotations

import heapq
import math
import threading
from dataclasses import dataclass, field
from enum import Enum


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

    @property
    def expected_uses(self) -> float:
        return sum(self.consumer_probabilities.values())

    def priority(self, current_step: int) -> float:
        if self.demand:
            return math.inf
        urgency = 1.0 / max(1, self.deadline - current_step)
        mib = max(self.size_bytes / 2**20, 1e-6)
        return self.miss_cost_ms * self.expected_uses * urgency / mib


@dataclass(frozen=True)
class QueueUpdate:
    """One consumer's contribution to a queued resource request."""

    key: ResourceKey
    consumer: str
    probability: float
    deadline: int
    size_bytes: int
    miss_cost_ms: float
    demand: bool = False


class MemoryRequestQueue:
    """Thread-safe updatable priority queue shared by KV and expert requests."""

    def __init__(self) -> None:
        self._requests: dict[ResourceKey, MemoryRequest] = {}
        self._heap: list[tuple[float, int, int, int, ResourceKey]] = []
        self._sequence = 0
        self._step = 0
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

    def _merge_locked(self, update: QueueUpdate) -> MemoryRequest:
        """Merge fields that can be updated incrementally under the queue lock."""
        request = self._requests.get(update.key)
        if request is None:
            request = MemoryRequest(
                update.key,
                update.size_bytes,
                update.miss_cost_ms,
                update.deadline,
            )
            self._requests[update.key] = request
        elif request.size_bytes != update.size_bytes:
            raise ValueError(f"size changed for existing resource {update.key}")
        request.consumer_probabilities[update.consumer] = update.probability
        request.consumer_deadlines[update.consumer] = update.deadline
        request.miss_cost_ms = max(request.miss_cost_ms, update.miss_cost_ms)
        request.demand = request.demand or update.demand
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
                request = self._merge_locked(update)
                requests.append(request)
                unique[request.key] = request
            for request in unique.values():
                request.deadline = min(request.consumer_deadlines.values())
                self._push(request)
            self._condition.notify()
            return requests

    def promote_demand(
        self,
        key: ResourceKey,
        *,
        consumer: str,
        size_bytes: int,
        miss_cost_ms: float,
    ) -> MemoryRequest:
        return self.upsert(
            key,
            consumer=consumer,
            probability=1.0,
            deadline=self._step,
            size_bytes=size_bytes,
            miss_cost_ms=miss_cost_ms,
            demand=True,
        )

    def set_step(self, step: int) -> None:
        """Advance logical time and rebuild priorities because urgency changed."""
        with self._condition:
            if step < self._step:
                raise ValueError("queue step cannot move backwards")
            self._step = step
            self._heap.clear()
            for request in self._requests.values():
                self._push(request)
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
                if request.consumer_probabilities:
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
                if request.consumer_probabilities:
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

    def pop_many(self, maximum: int, block: bool = False) -> list[MemoryRequest]:
        """Pop one same-class transfer batch without mixing demand and speculation."""
        if maximum <= 0:
            raise ValueError("maximum batch size must be positive")
        with self._condition:
            while True:
                first = self._pop_valid_locked()
                if first is not None:
                    requests = [first]
                    while len(requests) < maximum:
                        next_request = self._peek_valid_locked()
                        if next_request is None or next_request.demand != first.demand:
                            break
                        requests.append(self._pop_valid_locked())
                    return requests
                if not block or self._closed:
                    return []
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
            self._condition.notify_all()
