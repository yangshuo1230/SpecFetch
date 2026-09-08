import time

import torch

from src.runtime.memory_queue import MemoryRequestQueue, ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState
from src.runtime.transfer import (
    DemandRequest,
    OffloadRuntime,
    PackedExpertSlots,
    PrefetchRequest,
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


def test_phase_metrics_preserve_local_peak_instead_of_subtracting_gauge():
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    worker = TransferWorker(queue, residency, FakeBackend())
    earlier = worker.metrics_snapshot()
    worker.metrics.submitted = 5
    worker.metrics.maximum_transfer_batch = 4
    worker._phase_maximum_transfer_batch = 4
    current, first = worker.phase_metrics_since(earlier)
    assert first.submitted == 5
    assert first.maximum_transfer_batch == 4
    worker.metrics.submitted = 7
    worker._phase_maximum_transfer_batch = 2
    _, second = worker.phase_metrics_since(current)
    assert second.submitted == 2
    assert second.maximum_transfer_batch == 2


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
    w1, w2 = slots.fused_weights()
    assert resident.gate.data_ptr() == w1[0, :2].data_ptr()
    assert resident.down.data_ptr() == w2[0].data_ptr()
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


def test_prefetch_many_updates_resident_shared_leases_in_one_batch():
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    key = resource(0)
    residency.register_cpu(key, "cpu", 2**20)
    assert residency.begin_transfer(key, demand=True)
    residency.complete_transfer(key, "gpu")
    residency.release(key)
    worker = TransferWorker(queue, residency, FakeBackend())
    runtime = OffloadRuntime(queue, residency, worker)

    runtime.prefetch_many(
        [
            PrefetchRequest(key, "a", 0.5, 2, 4.0),
            PrefetchRequest(key, "b", 0.25, 4, 8.0),
        ]
    )

    record = residency.record(key)
    assert record.consumer_leases == {"a": (1.0, 2), "b": (0.5, 4)}
    assert record.priority == 1.5
    assert record.deadline == 2
    assert record.speculative
    assert not record.used
    assert len(queue) == 0


def test_prefetch_many_queues_each_consumer_and_marks_resource_once():
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    key = resource(0)
    residency.register_cpu(key, "cpu", 1024)
    worker = TransferWorker(queue, residency, FakeBackend())
    runtime = OffloadRuntime(queue, residency, worker)

    runtime.prefetch_many(
        [
            PrefetchRequest(key, "a", 0.6, 3, 2.0),
            PrefetchRequest(key, "b", 0.3, 5, 1.0),
        ]
    )

    [request] = queue.snapshot()
    assert residency.state(key) == ResourceState.QUEUED
    assert request.consumer_probabilities == {"a": 0.6, "b": 0.3}
    assert request.deadline == 3


def test_demand_many_uses_one_backend_transfer_batch():
    class BatchBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.batches = []

        def copy_many_to_gpu(self, items):
            self.batches.append([key for key, _ in items])
            return [f"gpu:{value}" for _, value in items]

    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 3, ResourceKind.KV: 1})
    backend = BatchBackend()
    worker = TransferWorker(queue, residency, backend)
    runtime = OffloadRuntime(queue, residency, worker)
    for index in range(3):
        residency.register_cpu(resource(index), f"cpu:{index}", 1024)
    worker.start()
    values = runtime.demand_many([DemandRequest(resource(index), "r0", 1.0) for index in range(3)])
    assert values == {resource(index): f"gpu:cpu:{index}" for index in range(3)}
    assert len(backend.batches) == 1
    assert set(backend.batches[0]) == {resource(0), resource(1), resource(2)}
    assert worker.metrics.transfer_batches == 1
    assert worker.metrics.maximum_transfer_batch == 3
    worker.close()


def test_discarding_worker_close_unqueues_pending_resources():
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    residency.register_cpu(resource(0), "cpu", 1024)
    worker = TransferWorker(queue, residency, FakeBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    runtime.prefetch(resource(0), consumer="r0", probability=1.0, deadline=1, miss_cost_ms=1)
    assert residency.state(resource(0)) == ResourceState.QUEUED
    worker.close()
    assert residency.state(resource(0)) == ResourceState.CPU_ONLY


def test_batch_cancel_returns_depleted_queued_resources_to_cpu_state():
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    for index in range(2):
        residency.register_cpu(resource(index), f"cpu:{index}", 1024)
    worker = TransferWorker(queue, residency, FakeBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    runtime.prefetch(resource(0), consumer="a", probability=0.8, deadline=1, miss_cost_ms=1)
    runtime.prefetch(resource(1), consumer="b", probability=0.7, deadline=1, miss_cost_ms=1)

    runtime.cancel_many([(resource(0), "a"), (resource(1), "b")])

    assert len(queue) == 0
    assert residency.state(resource(0)) == ResourceState.CPU_ONLY
    assert residency.state(resource(1)) == ResourceState.CPU_ONLY
    worker.close()


def test_low_priority_prefetch_cannot_evict_high_priority_lease():
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    residency.register_cpu(resource(0), "a", 1)
    residency.register_cpu(resource(1), "b", 1)
    assert residency.begin_transfer(resource(0), priority=10, deadline=1)
    residency.complete_transfer(resource(0), "gpu:a")
    assert not residency.begin_transfer(resource(1), priority=1, deadline=5)
    assert residency.state(resource(0)) == ResourceState.GPU_RESIDENT
    assert residency.state(resource(1)) == ResourceState.CPU_ONLY
