"""Focused CPU microbenchmarks for runtime bookkeeping hot paths."""

from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable

from src.runtime.memory_queue import (
    MemoryRequest,
    MemoryRequestQueue,
    QueueUpdate,
    ResourceKey,
    ResourceKind,
)
from src.runtime.residency import ResidencyManager


def _measure(function: Callable[[], object], iterations: int, repeat: int) -> float:
    samples = timeit.repeat(function, number=iterations, repeat=repeat)
    return statistics.median(samples) / iterations * 1e6


def benchmark_residency_eviction(iterations: int, repeat: int) -> None:
    """Compare the pre-optimization two-scan path with current admission."""
    kind = ResourceKind.EXPERT
    manager = ResidencyManager({kind: 512})
    for index in range(513):
        key = ResourceKey(kind, layer=index // 64, object_id=index)
        manager.register_cpu(key, index, 1)
        if index < 512:
            manager.begin_transfer(key, priority=float(index % 17), deadline=index % 31)
            manager.complete_transfer(key, index)
    incoming = ResourceKey(kind, layer=8, object_id=512)
    protected = {incoming}

    def legacy_candidate() -> ResourceKey | None:
        lru = manager._resident[kind]
        order = {key: index for index, key in enumerate(lru)}
        candidates = [
            key for key in lru if key not in protected and not manager._records[key].pinned
        ]
        return min(
            candidates,
            key=lambda key: (
                manager._records[key].priority,
                -manager._records[key].deadline,
                order[key],
            ),
            default=None,
        )

    def legacy_full_cache_path() -> ResourceKey | None:
        victim = legacy_candidate()
        if victim is not None:
            return legacy_candidate()
        return None

    def current_full_cache_path() -> ResourceKey | None:
        return manager._eviction_candidate(kind, protected)

    legacy = _measure(legacy_full_cache_path, iterations, repeat)
    current = _measure(current_full_cache_path, iterations, repeat)
    print(f"residency eviction, 512 slots: legacy_two_scan_us={legacy:.3f}")
    print(f"residency eviction, 512 slots: current_one_scan_us={current:.3f}")
    print(f"residency eviction speedup: {legacy / current:.2f}x")


def _legacy_upsert_many(queue: MemoryRequestQueue, updates: list[QueueUpdate]) -> None:
    """Reproduce the former per-consumer deadline reduction for comparison."""
    for update in updates:
        queue._validate(update)
    with queue._condition:
        if queue._closed:
            raise RuntimeError("queue is closed")
        sizes: dict[ResourceKey, int] = {}
        for update in updates:
            expected_size = sizes.setdefault(update.key, update.size_bytes)
            existing = queue._requests.get(update.key)
            if expected_size != update.size_bytes or (
                existing is not None and existing.size_bytes != update.size_bytes
            ):
                raise ValueError(f"size changed for existing resource {update.key}")
        requests: list[MemoryRequest] = []
        for update in updates:
            request = queue._requests.get(update.key)
            if request is None:
                request = MemoryRequest(
                    update.key,
                    update.size_bytes,
                    update.miss_cost_ms,
                    update.deadline,
                )
                queue._requests[update.key] = request
            request.consumer_probabilities[update.consumer] = update.probability
            request.consumer_deadlines[update.consumer] = update.deadline
            request.deadline = min(request.consumer_deadlines.values())
            request.miss_cost_ms = max(request.miss_cost_ms, update.miss_cost_ms)
            request.demand = request.demand or update.demand
            requests.append(request)
        for request in {item.key: item for item in requests}.values():
            queue._push(request)
        queue._condition.notify()


def benchmark_shared_queue_upsert(consumers: int, iterations: int, repeat: int) -> None:
    """Compare deadline reduction for many consumers sharing one resource."""
    resource = ResourceKey(ResourceKind.EXPERT, layer=0, object_id=0)
    updates = [
        QueueUpdate(
            resource,
            consumer=f"request-{index}",
            probability=0.5,
            deadline=consumers - index,
            size_bytes=1024,
            miss_cost_ms=1.0,
        )
        for index in range(consumers)
    ]
    legacy_queue = MemoryRequestQueue()
    current_queue = MemoryRequestQueue()

    legacy = _measure(lambda: _legacy_upsert_many(legacy_queue, updates), iterations, repeat)
    current = _measure(lambda: current_queue.upsert_many(updates), iterations, repeat)
    print(f"shared queue upsert, {consumers} consumers: legacy_per_consumer_min_us={legacy:.3f}")
    print(f"shared queue upsert, {consumers} consumers: current_per_resource_min_us={current:.3f}")
    print(f"shared queue upsert speedup: {legacy / current:.2f}x")


def benchmark_worker_step_snapshot(accesses: int, iterations: int, repeat: int) -> None:
    """Compare repeated locked logical-step reads with one batch snapshot."""
    queue = MemoryRequestQueue()

    def legacy_per_consumer_reads() -> int:
        current_step = 0
        for _ in range(accesses):
            current_step = queue.current_step
        return current_step

    def current_batch_snapshot() -> int:
        return queue.current_step

    legacy = _measure(legacy_per_consumer_reads, iterations, repeat)
    current = _measure(current_batch_snapshot, iterations, repeat)
    print(f"worker logical step, {accesses} uses: legacy_repeated_lock_us={legacy:.3f}")
    print(f"worker logical step, {accesses} uses: current_batch_snapshot_us={current:.3f}")
    print(f"worker logical-step read speedup: {legacy / current:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--queue-iterations", type=int, default=20)
    parser.add_argument("--queue-consumers", type=int, default=2_000)
    parser.add_argument("--step-accesses", type=int, default=100_000)
    parser.add_argument("--step-iterations", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    benchmark_residency_eviction(args.iterations, args.repeat)
    benchmark_shared_queue_upsert(args.queue_consumers, args.queue_iterations, args.repeat)
    benchmark_worker_step_snapshot(args.step_accesses, args.step_iterations, args.repeat)


if __name__ == "__main__":
    main()
