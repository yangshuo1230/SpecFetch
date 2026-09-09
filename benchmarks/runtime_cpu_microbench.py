"""Focused CPU microbenchmarks for runtime bookkeeping hot paths."""

from __future__ import annotations

import argparse
import statistics
import timeit
from collections.abc import Callable

from src.runtime.memory_queue import ResourceKey, ResourceKind
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    benchmark_residency_eviction(args.iterations, args.repeat)


if __name__ == "__main__":
    main()
