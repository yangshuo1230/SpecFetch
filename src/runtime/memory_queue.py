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
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if size_bytes <= 0 or miss_cost_ms < 0:
            raise ValueError("size_bytes must be positive and miss_cost_ms non-negative")
        with self._condition:
            if self._closed:
                raise RuntimeError("queue is closed")
            request = self._requests.get(key)
            if request is None:
                request = MemoryRequest(key, size_bytes, miss_cost_ms, deadline)
                self._requests[key] = request
            elif request.size_bytes != size_bytes:
                raise ValueError(f"size changed for existing resource {key}")
            request.consumer_probabilities[consumer] = probability
            request.consumer_deadlines[consumer] = deadline
            request.deadline = min(request.consumer_deadlines.values())
            request.miss_cost_ms = max(request.miss_cost_ms, miss_cost_ms)
            request.demand = request.demand or demand
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

    def pop(self, block: bool = False) -> MemoryRequest | None:
        with self._condition:
            while True:
                while self._heap:
                    _, _, _, version, key = heapq.heappop(self._heap)
                    request = self._requests.get(key)
                    if request is None or request.version != version:
                        continue
                    del self._requests[key]
                    return request
                if not block or self._closed:
                    return None
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
