import time

import torch

from src.runtime.memory_queue import MemoryRequestQueue, ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState
from src.runtime.transfer import (
    OffloadRuntime,
    PackedExpertSlots,
    TransferMetrics,
    TransferWorker,
)


class FakeBackend:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.copies = []

    def copy_to_gpu(self, key, value):
        time.sleep(self.delay)
        self.copies.append(key)
        return f"gpu:{value}"


def resource(index, kind=ResourceKind.EXPERT):
    return ResourceKey(kind, layer=0, object_id=index)


def build_runtime(capacity=2, delay=0.0):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: capacity, ResourceKind.KV: capacity})
    backend = FakeBackend(delay)
    worker = TransferWorker(queue, residency, backend)
    runtime = OffloadRuntime(queue, residency, worker)
    for index in range(4):
        residency.register_cpu(resource(index), f"cpu:{index}", 1024)
    worker.start()
    return runtime, residency, worker, backend


def test_prefetch_and_demand_use_background_worker():
    runtime, residency, worker, backend = build_runtime()
    runtime.prefetch(resource(0), consumer="r0", probability=0.8, deadline=2, miss_cost_ms=1)
    assert runtime.demand(resource(0), consumer="r0", miss_cost_ms=1) == "gpu:cpu:0"
    assert backend.copies == [resource(0)]
    assert residency.state(resource(0)) == ResourceState.GPU_RESIDENT
    worker.close()


def test_lru_evicts_unpinned_resource():
    runtime, residency, worker, _ = build_runtime(capacity=1)
    runtime.demand(resource(0), consumer="r0", miss_cost_ms=1)
    runtime.demand(resource(1), consumer="r0", miss_cost_ms=1)
    assert residency.state(resource(0)) == ResourceState.CPU_ONLY
    assert residency.state(resource(1)) == ResourceState.GPU_RESIDENT
    assert residency.evictions == 1
    worker.close()


def test_demand_promotes_queued_resource():
    runtime, _, worker, backend = build_runtime(delay=0.005)
    runtime.prefetch(resource(0), consumer="r0", probability=0.9, deadline=1, miss_cost_ms=1)
    runtime.prefetch(resource(1), consumer="r0", probability=0.1, deadline=10, miss_cost_ms=1)
    value = runtime.demand(resource(1), consumer="r0", miss_cost_ms=1)
    assert value == "gpu:cpu:1"
    assert resource(1) in backend.copies
    worker.close()


def test_transfer_metric_snapshots_have_additive_deltas():
    earlier = TransferMetrics(submitted=2, bytes=10, transfer_ms=1.5, demand_hits=1)
    later = TransferMetrics(submitted=5, bytes=42, transfer_ms=4.0, demand_hits=3)
    delta = later.delta(earlier)
    assert delta.submitted == 3
    assert delta.bytes == 32
    assert delta.transfer_ms == 2.5
    assert delta.demand_hits == 2


def test_packed_expert_slots_reuse_released_storage():
    from src.runtime.expert import ExpertWeights

    slots = PackedExpertSlots(1, "cpu")
    first_key = resource(10)
    second_key = resource(11)
    first = ExpertWeights(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, 2))
    second = ExpertWeights(
        torch.full((2, 3), 2.0),
        torch.full((2, 3), 3.0),
        torch.full((3, 2), 4.0),
    )
    resident = slots.acquire(first_key, first)
    gate_pointer = resident.gate.data_ptr()
    assert slots.release(first_key)
    replaced = slots.acquire(second_key, second)
    assert replaced.gate.data_ptr() == gate_pointer
    assert torch.equal(replaced.gate, second.gate)


def test_residency_eviction_returns_packed_expert_slot():
    from src.runtime.expert import ExpertWeights

    class PackedBackend:
        def __init__(self):
            self.slots = PackedExpertSlots(1, "cpu")

        def copy_to_gpu(self, key, value):
            return self.slots.acquire(key, value)

        def release_gpu(self, key, value):
            del value
            self.slots.release(key)

    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    backend = PackedBackend()
    worker = TransferWorker(queue, residency, backend)
    runtime = OffloadRuntime(queue, residency, worker)
    value = ExpertWeights(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, 2))
    residency.register_cpu(resource(0), value, value.size_bytes)
    residency.register_cpu(resource(1), value, value.size_bytes)
    worker.start()
    first = runtime.demand(resource(0), consumer="r0", miss_cost_ms=1)
    first_pointer = first.gate.data_ptr()
    second = runtime.demand(resource(1), consumer="r0", miss_cost_ms=1)
    assert second.gate.data_ptr() == first_pointer
    worker.close()


def test_drop_removes_request_private_resource_record():
    runtime, residency, worker, _ = build_runtime()
    key = resource(0)
    runtime.demand(key, consumer="r0", miss_cost_ms=1)
    runtime.drop(key)
    assert not residency.contains(key)
    worker.close()


def test_consumer_lease_cancellation_and_demand_scope_recompute_priority():
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    key = resource(0)
    residency.register_cpu(key, "cpu", 1)
    assert residency.begin_transfer(
        key,
        priority=5,
        deadline=1,
        consumer_leases={"a": (2, 1), "b": (3, 2)},
    )
    residency.complete_transfer(key, "gpu")
    assert residency.record(key).priority == 5
    residency.cancel_lease(key, "a")
    assert residency.record(key).priority == 3
    residency.mark_demand(key)
    assert residency.record(key).priority == float("inf")
    residency.release(key)
    assert residency.record(key).priority == 3


def test_low_priority_prefetch_cannot_evict_high_priority_lease():
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    residency.register_cpu(resource(0), "a", 1)
    residency.register_cpu(resource(1), "b", 1)
    assert residency.begin_transfer(resource(0), priority=10, deadline=1)
    residency.complete_transfer(resource(0), "gpu:a")
    assert not residency.begin_transfer(resource(1), priority=1, deadline=5)
    assert residency.state(resource(0)) == ResourceState.GPU_RESIDENT
    assert residency.state(resource(1)) == ResourceState.CPU_ONLY
