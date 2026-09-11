import threading
import time

import pytest
import torch

from src.runtime.memory_queue import MemoryRequestQueue, ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState
from src.runtime.transfer import (
    CudaTransferBackend,
    DemandRequest,
    ExpertSlotMap,
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


def test_resource_key_caches_stable_identity_hash():
    first = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=7, request_id="r0")
    same = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=7, request_id="r0")
    different = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=8, request_id="r0")

    assert hash(first) == first._cached_hash == hash(same)
    assert first == same
    assert first != different
    assert {first: "value"}[same] == "value"


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


def test_expert_eviction_preserves_cross_token_hits_across_layers():
    residency = ResidencyManager({ResourceKind.EXPERT: 3, ResourceKind.KV: 1})
    routes = [
        ResourceKey(ResourceKind.EXPERT, layer=0, object_id=0),
        ResourceKey(ResourceKind.EXPERT, layer=0, object_id=1),
        ResourceKey(ResourceKind.EXPERT, layer=1, object_id=0),
        ResourceKey(ResourceKind.EXPERT, layer=1, object_id=1),
    ]
    for key in routes:
        residency.register_cpu(key, f"cpu:{key.layer}:{key.object_id}", 1)

    def demand_cycle() -> int:
        hits = 0
        for key in routes:
            if residency.state(key) == ResourceState.GPU_RESIDENT:
                hits += 1
                residency.mark_demand(key)
                residency.get_gpu(key)
            else:
                assert residency.begin_transfer(key, demand=True)
                residency.complete_transfer(key, f"gpu:{key.layer}:{key.object_id}")
            residency.release(key)
        return hits

    assert demand_cycle() == 0
    # A global LRU cache of size three has zero hits on this cyclic trace of
    # four objects. Layer balancing retains one route from each layer.
    assert demand_cycle() == 2
    assert {key.layer for key in residency.resident_keys(ResourceKind.EXPERT)} == {0, 1}


def test_expert_eviction_retains_frequent_route_within_layer():
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    hot, cold, replacement = (resource(index) for index in range(3))
    for key in (hot, cold, replacement):
        residency.register_cpu(key, f"cpu:{key.object_id}", 1)

    def demand(key):
        values, pending, _, _ = residency.prepare_demands([key])
        if pending:
            assert residency.begin_transfer(key, demand=True)
            residency.complete_transfer(key, f"gpu:{key.object_id}")
        else:
            assert key in values
        residency.release(key)

    demand(hot)
    for _ in range(3):
        demand(hot)
    demand(cold)
    # hot is older in LRU order here, but its observed demand frequency is higher.
    demand(replacement)

    assert residency.state(hot) == ResourceState.GPU_RESIDENT
    assert residency.state(cold) == ResourceState.CPU_ONLY
    assert residency.record(hot).demand_count == 4


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
    worker.record_prefetch_admission(12, 5)
    current, first = worker.phase_metrics_since(earlier)
    assert first.submitted == 5
    assert first.maximum_transfer_batch == 4
    assert first.maximum_prefetch_candidates == 12
    assert first.maximum_prefetch_admitted == 5
    worker.metrics.submitted = 7
    worker._phase_maximum_transfer_batch = 2
    worker.record_prefetch_admission(3, 2)
    _, second = worker.phase_metrics_since(current)
    assert second.submitted == 2
    assert second.maximum_transfer_batch == 2
    assert second.maximum_prefetch_candidates == 3
    assert second.maximum_prefetch_admitted == 2


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


def test_packed_expert_storage_initializes_without_consuming_slot():
    from src.runtime.expert import ExpertWeights

    slots = PackedExpertSlots(2, "cpu")
    weights = ExpertWeights(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, 2))

    slots.initialize(weights)

    assert slots.allocated
    assert len(slots._free) == 2
    assert not slots._assigned


def test_packed_gate_up_source_uses_two_copies_per_expert():
    from src.runtime.expert import pack_expert_weights

    slots = PackedExpertSlots(1, "cpu")
    gate = torch.arange(6.0).reshape(2, 3)
    up = gate + 10
    down = torch.arange(6.0).reshape(3, 2)
    packed = pack_expert_weights(gate, up, down, pin_memory=False)

    resident = slots.acquire(resource(20), packed)

    assert packed.gate.untyped_storage().data_ptr() == packed.up.untyped_storage().data_ptr()
    assert packed.gate.storage_offset() + packed.gate.numel() == packed.up.storage_offset()
    assert slots.copy_operations == 2
    assert torch.equal(resident.gate, gate)
    assert torch.equal(resident.up, up)
    assert torch.equal(resident.down, down)


def test_persistent_expert_slot_map_tracks_late_updates_and_removal():
    slot_map = ExpertSlotMap("cpu")
    first = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=3)
    second = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=5)
    other_layer = ResourceKey(ResourceKind.EXPERT, layer=4, object_id=3)

    slot_map.update([(first, 7), (other_layer, 1)])
    layer_two = slot_map.get(2, 8)
    assert layer_two.tolist() == [-1, -1, -1, 7, -1, -1, -1, -1]
    assert slot_map.get(2, 8).data_ptr() == layer_two.data_ptr()

    slot_map.update([(second, 9)])
    assert layer_two.tolist() == [-1, -1, -1, 7, -1, 9, -1, -1]
    slot_map.remove(first)
    assert layer_two.tolist() == [-1, -1, -1, -1, -1, 9, -1, -1]
    assert slot_map.get(4, 8).tolist()[3] == 1


def test_deferred_expert_map_removals_merge_with_next_slot_batch():
    slot_map = ExpertSlotMap("cpu")
    first = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=1)
    second = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=4)
    replacement = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=6)
    slot_map.update([(first, 3), (second, 5)])
    mapping = slot_map.get(2, 8)

    slot_map.remove(first, defer_device=True)
    slot_map.remove(second, defer_device=True)
    slot_map.update([(replacement, 7)])

    assert mapping.tolist() == [-1, -1, -1, -1, -1, -1, 7, -1]


def test_expert_map_reuses_preallocated_update_staging():
    slot_map = ExpertSlotMap("cpu", staging_capacity=4)
    first = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=1)
    second = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=4)
    other_layer = ResourceKey(ResourceKind.EXPERT, layer=3, object_id=2)
    mapping = slot_map.get(2, 8)
    other_mapping = slot_map.get(3, 8)
    index_pointer = slot_map._device_indices.data_ptr()
    slot_pointer = slot_map._device_slots.data_ptr()

    slot_map.update([(first, 3)])
    slot_map.remove(first, defer_device=True)
    slot_map.update([(second, 7), (other_layer, 5)])

    assert mapping.tolist() == [-1, -1, -1, -1, 7, -1, -1, -1]
    assert other_mapping.tolist() == [-1, -1, 5, -1, -1, -1, -1, -1]
    assert slot_map._device_indices.data_ptr() == index_pointer
    assert slot_map._device_slots.data_ptr() == slot_pointer


def test_cuda_backend_reuses_only_completed_unreferenced_use_events(monkeypatch):
    class FakeEvent:
        def __init__(self):
            self.complete = False
            self.records = 0

        def query(self):
            return self.complete

        def record(self, stream):
            del stream
            self.complete = False
            self.records += 1

    created = []

    def make_event():
        event = FakeEvent()
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: device)
    backend = object.__new__(CudaTransferBackend)
    backend.device = torch.device("cpu")
    backend.expert_slots = object()
    backend._use_events = {}
    backend._use_event_references = {}
    backend._retired_use_events = []
    backend._use_event_lock = threading.Lock()
    backend._waited_use_events = set()
    first, second = resource(0), resource(1)

    backend.record_uses([first, second])
    shared = created[0]
    shared.complete = True
    backend._waited_use_events.add(shared)
    backend.record_uses([first, second])

    assert len(created) == 2
    assert backend._use_events[second] is created[1]
    backend._waited_use_events.clear()
    backend.record_uses([first, second])

    assert len(created) == 2
    assert backend._use_events[second] is shared
    assert shared.records == 2


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
    residency.cancel_lease(key, "b")
    residency.release(key)
    assert residency.record(key).priority == 0
    assert residency.record(key).deadline == 0


def test_incremental_lease_aggregate_handles_overwritten_earliest_deadline():
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    key = resource(0)
    residency.register_cpu(key, "cpu", 1)

    residency.update_lease(key, 2.0, 2, "a")
    residency.update_lease(key, 3.0, 5, "b")
    residency.update_lease(key, 4.0, 7, "a")

    record = residency.record(key)
    assert record.priority == 7.0
    assert record.deadline == 5
    residency.cancel_lease(key, "b")
    assert record.priority == 4.0
    assert record.deadline == 7


def test_release_many_records_one_backend_use_batch_and_clears_demands():
    class UseTrackingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.use_batches = []

        def record_uses(self, keys):
            self.use_batches.append(list(keys))

    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    keys = [resource(0), resource(1)]
    for key in keys:
        residency.register_cpu(key, f"cpu:{key.object_id}", 1)
        assert residency.begin_transfer(key, demand=True)
        residency.complete_transfer(key, f"gpu:{key.object_id}")
    backend = UseTrackingBackend()
    runtime = OffloadRuntime(
        MemoryRequestQueue(),
        residency,
        TransferWorker(MemoryRequestQueue(), residency, backend),
    )

    runtime.release_many(keys)

    assert backend.use_batches == [keys]
    assert all(not residency.record(key).demand_active for key in keys)


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


def test_request_intents_keep_named_immutable_fields():
    prefetch = PrefetchRequest(resource(0), "a", 0.5, 2, 4.0)
    demand = DemandRequest(resource(1), "b", 3.0)

    assert prefetch.key == resource(0)
    assert prefetch.consumer == "a"
    assert prefetch.probability == 0.5
    assert prefetch.deadline == 2
    assert prefetch.miss_cost_ms == 4.0
    assert demand == (resource(1), "b", 3.0)
    with pytest.raises(AttributeError):
        prefetch.deadline = 3


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
    pending_leases = residency.record(key).consumer_leases
    assert set(pending_leases) == {"a", "b"}
    assert {consumer: deadline for consumer, (_, deadline) in pending_leases.items()} == {
        "a": 3,
        "b": 5,
    }


def test_transfer_worker_uses_step_snapshot_from_popped_batch():
    class CountingQueue(MemoryRequestQueue):
        def __init__(self):
            super().__init__()
            self.current_step_reads = 0

        @property
        def current_step(self):
            self.current_step_reads += 1
            return super().current_step

    queue = CountingQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 3, ResourceKind.KV: 1})
    backend = FakeBackend()
    worker = TransferWorker(queue, residency, backend, max_batch_size=3)
    runtime = OffloadRuntime(queue, residency, worker)
    for index in range(3):
        residency.register_cpu(resource(index), f"cpu:{index}", 2**20)
    runtime.prefetch_many(
        [
            PrefetchRequest(resource(index), consumer, 0.5, index + 2, 4.0)
            for index in range(3)
            for consumer in ("a", "b")
        ]
    )
    queue.current_step_reads = 0
    queue.close()

    worker.start()
    worker.close(drain=True)

    assert queue.current_step_reads == 0
    assert set(backend.copies) == {resource(index) for index in range(3)}


def test_cancel_after_queue_pop_prevents_stale_speculative_transfer():
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    key = resource(0)
    residency.register_cpu(key, "cpu", 2**20)
    worker = TransferWorker(queue, residency, FakeBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    runtime.prefetch(key, consumer="a@1", probability=0.8, deadline=3, miss_cost_ms=4.0)
    [popped] = queue.pop_many(1)

    runtime.cancel_many([(key, "a@1")])
    consumer_leases = {
        consumer: (
            popped.miss_cost_ms
            * probability
            / max(1, popped.consumer_deadlines[consumer] - queue.current_step)
            / max(popped.size_bytes / 2**20, 1e-6),
            popped.consumer_deadlines[consumer],
        )
        for consumer, probability in popped.consumer_probabilities.items()
    }
    accepted = residency.begin_transfer(
        key,
        priority=popped.priority(queue.current_step),
        deadline=popped.deadline,
        consumer_leases=consumer_leases,
    )

    assert not accepted
    assert residency.state(key) == ResourceState.CPU_ONLY
    assert not residency.record(key).consumer_leases


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


def test_transfer_worker_publishes_residency_batch_without_scalar_calls(monkeypatch):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 3, ResourceKind.KV: 1})
    backend = FakeBackend()
    worker = TransferWorker(queue, residency, backend, max_batch_size=3)
    runtime = OffloadRuntime(queue, residency, worker)
    for index in range(3):
        residency.register_cpu(resource(index), f"cpu:{index}", 1024)

    def reject_scalar(*args, **kwargs):
        del args, kwargs
        raise AssertionError("worker must use batched residency operations")

    monkeypatch.setattr(residency, "record", reject_scalar)
    monkeypatch.setattr(residency, "begin_transfer", reject_scalar)
    monkeypatch.setattr(residency, "begin_transfers", reject_scalar)
    monkeypatch.setattr(residency, "complete_transfer", reject_scalar)
    monkeypatch.setattr(residency, "cpu_values", reject_scalar)
    monkeypatch.setattr(residency, "complete_transfers", reject_scalar)
    worker.start()

    values = runtime.demand_many([DemandRequest(resource(index), "r0", 1.0) for index in range(3)])

    assert values == {resource(index): f"gpu:cpu:{index}" for index in range(3)}
    worker.close()


def test_transfer_worker_uses_key_only_demand_admission(monkeypatch):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    worker = TransferWorker(queue, residency, FakeBackend(), max_batch_size=2)
    runtime = OffloadRuntime(queue, residency, worker)
    for index in range(2):
        residency.register_cpu(resource(index), f"cpu:{index}", 1024)
    demand_batches = []
    begin_demand_transfers = residency.begin_demand_transfers

    def capture_demands(keys):
        demand_batches.append(list(keys))
        return begin_demand_transfers(keys)

    monkeypatch.setattr(residency, "begin_demand_transfers", capture_demands)
    monkeypatch.setattr(
        residency,
        "begin_transfers",
        lambda admissions: (_ for _ in ()).throw(
            AssertionError(f"demand used generic admissions: {admissions}")
        ),
    )
    worker.start()

    runtime.demand_many([DemandRequest(resource(index), "r0", 1.0) for index in range(2)])

    assert demand_batches == [[resource(0), resource(1)]]
    worker.close()


def test_demand_many_uses_minimal_queue_updates(monkeypatch):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    worker = TransferWorker(queue, residency, FakeBackend(), max_batch_size=2)
    runtime = OffloadRuntime(queue, residency, worker)
    for index in range(2):
        residency.register_cpu(resource(index), f"cpu:{index}", 1024)
    demand_batches = []
    upsert_demand_intents = queue.upsert_demand_intents

    def capture_demands(requests, sizes):
        demand_batches.append((list(requests), list(sizes)))
        return upsert_demand_intents(requests, sizes)

    monkeypatch.setattr(queue, "upsert_demand_intents", capture_demands)
    monkeypatch.setattr(
        queue,
        "upsert_demands",
        lambda updates: (_ for _ in ()).throw(
            AssertionError(f"demand allocated forwarding updates: {updates}")
        ),
    )
    monkeypatch.setattr(
        queue,
        "upsert_many",
        lambda updates: (_ for _ in ()).throw(
            AssertionError(f"demand used speculative updates: {updates}")
        ),
    )
    worker.start()

    requests = [DemandRequest(resource(index), "r0", 1.0) for index in range(2)]
    runtime.demand_many(requests)

    assert demand_batches == [(requests, [1024, 1024])]
    worker.close()


def test_complete_transfers_validates_entire_batch_before_publishing():
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    first, second = resource(0), resource(1)
    for key in (first, second):
        residency.register_cpu(key, f"cpu:{key.object_id}", 1)
    assert residency.begin_transfer(first, demand=True)

    with pytest.raises(RuntimeError, match="invalid state"):
        residency.complete_transfers([(first, "gpu:0"), (second, "gpu:1")])
    with pytest.raises(RuntimeError, match="lengths do not match"):
        residency.complete_transfer_values([first, second], ["gpu:0"])

    assert residency.state(first) == ResourceState.IN_FLIGHT
    assert residency.state(second) == ResourceState.CPU_ONLY


def test_begin_transfers_preserves_ordered_speculative_and_demand_semantics():
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    speculative, demand = resource(0), resource(1)
    for key in (speculative, demand):
        residency.register_cpu(key, f"cpu:{key.object_id}", 1)
        assert residency.mark_queued(key)
    residency.update_lease(speculative, 1.0, 5, "forecast")

    accepted = residency.begin_transfers(
        [
            (speculative, 2.0, 3, False, {"forecast": (2.0, 3)}),
            (demand, float("inf"), 0, True, None),
        ]
    )

    assert accepted == [True, True]
    assert residency.state(speculative) == ResourceState.IN_FLIGHT
    assert residency.record(speculative).consumer_leases == {"forecast": (2.0, 3)}
    assert residency.record(speculative).priority == 2.0
    assert residency.state(demand) == ResourceState.IN_FLIGHT
    assert residency.record(demand).consumer_leases == {}
    assert residency.record(demand).demand_active
    assert residency.record(demand).priority == float("inf")
    assert residency.record(demand).deadline == 0
    assert residency.record(demand).lease_priority == 0.0
    assert residency.record(demand).lease_deadline == 0

    residency.complete_transfer(speculative, "gpu:0")
    residency.release(speculative)
    assert residency.evict(speculative)
    assert residency.mark_queued(speculative)
    residency.update_lease(speculative, 3.0, 7, "next")
    reused_leases = residency.record(speculative).consumer_leases
    assert residency.begin_demand_transfers([speculative]) == [True]
    assert residency.record(speculative).consumer_leases is reused_leases
    assert reused_leases == {}


def test_batched_demand_plans_same_layer_balanced_victims_as_repeated_selection():
    def populated():
        residency = ResidencyManager({ResourceKind.EXPERT: 8, ResourceKind.KV: 1})
        for index in range(8):
            key = ResourceKey(ResourceKind.EXPERT, layer=index % 3, object_id=index)
            residency.register_cpu(key, index, 1)
            assert residency.begin_transfer(key, demand=True)
            residency.complete_transfer(key, index)
            residency.release(key)
            record = residency.record(key)
            record.priority = float(index % 2)
            record.deadline = index % 4
            record.demand_count = index % 3
        return residency

    planned_residency = populated()
    repeated_residency = populated()
    planned = planned_residency._eviction_candidates(ResourceKind.EXPERT, set(), 6)
    repeated = []
    for _ in range(6):
        victim = repeated_residency._eviction_candidate(ResourceKind.EXPERT, set())
        assert victim is not None
        repeated.append(victim)
        repeated_residency._evict_victim(ResourceKind.EXPERT, victim)

    assert planned == repeated


def test_batched_kv_victim_plan_matches_repeated_selection():
    def populated():
        residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 8})
        for index in range(8):
            key = ResourceKey(ResourceKind.KV, layer=index % 3, object_id=index, request_id="r")
            residency.register_cpu(key, index, 1)
            assert residency.begin_transfer(key, demand=True)
            residency.complete_transfer(key, index)
            residency.release(key)
            record = residency.record(key)
            record.priority = float(index % 2)
            record.deadline = index % 4
            record.demand_count = index % 3
        return residency

    planned_residency = populated()
    repeated_residency = populated()
    planned = planned_residency._eviction_candidates(ResourceKind.KV, set(), 6)
    repeated = []
    for _ in range(6):
        victim = repeated_residency._eviction_candidate(ResourceKind.KV, set())
        assert victim is not None
        repeated.append(victim)
        repeated_residency._evict_victim(ResourceKind.KV, victim)

    assert planned == repeated


def test_batched_demand_capacity_avoids_repeated_victim_scans(monkeypatch):
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    resident = [resource(0), resource(1)]
    incoming = [resource(2), resource(3)]
    for key in resident + incoming:
        residency.register_cpu(key, key.object_id, 1)
    for key in resident:
        assert residency.begin_transfer(key, demand=True)
        residency.complete_transfer(key, key.object_id)
        residency.release(key)

    monkeypatch.setattr(
        residency,
        "_eviction_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("demand batch repeated scalar victim scan")
        ),
    )
    accepted = residency.begin_transfers([(key, float("inf"), 0, True, None) for key in incoming])

    assert accepted == [True, True]
    assert all(residency.state(key) == ResourceState.CPU_ONLY for key in resident)
    assert all(residency.state(key) == ResourceState.IN_FLIGHT for key in incoming)


def test_demand_many_batches_resident_hit_and_miss_state_transitions(monkeypatch):
    runtime, residency, worker, _ = build_runtime(capacity=2)
    assert runtime.demand(resource(0), consumer="warm", miss_cost_ms=1) == "gpu:cpu:0"

    def scalar_path(*_args, **_kwargs):
        raise AssertionError("demand_many must use batched residency operations")

    monkeypatch.setattr(residency, "state", scalar_path)
    monkeypatch.setattr(residency, "mark_demand", scalar_path)
    monkeypatch.setattr(residency, "get_gpu", scalar_path)
    monkeypatch.setattr(residency, "wait_resident", scalar_path)
    values = runtime.demand_many(
        [
            DemandRequest(resource(0), "a", 1.0),
            DemandRequest(resource(1), "b", 1.0),
        ]
    )

    assert values == {resource(0): "gpu:cpu:0", resource(1): "gpu:cpu:1"}
    assert worker.metrics.demand_hits == 1
    assert worker.metrics.demand_misses == 2  # Includes the initial warm miss.
    worker.close()


def test_all_resident_demand_batch_skips_queue_and_wait(monkeypatch):
    runtime, residency, worker, _ = build_runtime(capacity=2)
    keys = [resource(0), resource(1)]
    runtime.demand_many([DemandRequest(key, "warm", 1.0) for key in keys])
    runtime.release_many(keys)

    def reject_slow_path(*args, **kwargs):
        del args, kwargs
        raise AssertionError("resident demand batch must return before queue/wait")

    monkeypatch.setattr(runtime.queue, "upsert_many", reject_slow_path)
    monkeypatch.setattr(residency, "wait_resident_many", reject_slow_path)

    values = runtime.demand_many([DemandRequest(key, "hit", 1.0) for key in keys])

    assert values == {resource(index): f"gpu:cpu:{index}" for index in range(2)}
    runtime.release_many(keys)
    worker.close()


def test_duplicate_resident_demands_preserve_per_request_hit_metrics():
    runtime, residency, worker, _ = build_runtime(capacity=1)
    key = resource(0)
    runtime.demand(key, consumer="warm", miss_cost_ms=1.0)
    runtime.release(key)
    earlier_count = residency.record(key).demand_count

    values = runtime.demand_many(
        [
            DemandRequest(key, "a", 1.0),
            DemandRequest(key, "b", 1.0),
        ]
    )

    assert values == {key: "gpu:cpu:0"}
    assert worker.metrics.demand_hits == 2
    assert worker.metrics.demand_misses == 1
    assert residency.record(key).demand_count == earlier_count + 1
    runtime.release(key)
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


def test_batch_cancel_returns_depleted_queued_resources_to_cpu_state(monkeypatch):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    for index in range(2):
        residency.register_cpu(resource(index), f"cpu:{index}", 1024)
    worker = TransferWorker(queue, residency, FakeBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    runtime.prefetch(resource(0), consumer="a", probability=0.8, deadline=1, miss_cost_ms=1)
    runtime.prefetch(resource(1), consumer="b", probability=0.7, deadline=1, miss_cost_ms=1)
    calls = []
    unqueue_many = residency.unqueue_many

    def counted_unqueue(keys):
        calls.append(set(keys))
        return unqueue_many(keys)

    monkeypatch.setattr(residency, "unqueue_many", counted_unqueue)
    monkeypatch.setattr(
        residency,
        "unqueue",
        lambda _key: (_ for _ in ()).throw(AssertionError("scalar unqueue is forbidden")),
    )

    runtime.cancel_many([(resource(0), "a"), (resource(1), "b")])

    assert calls == [{resource(0), resource(1)}]
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


def test_eviction_prefers_priority_then_deadline_then_lru_and_honors_protection():
    residency = ResidencyManager({ResourceKind.EXPERT: 5, ResourceKind.KV: 1})
    for index in range(6):
        residency.register_cpu(resource(index), str(index), 1)
    leases = [
        (2.0, 1),
        (1.0, 4),
        (1.0, 6),
        (1.0, 6),
        (0.5, 10),
    ]
    for index, (priority, deadline) in enumerate(leases):
        assert residency.begin_transfer(resource(index), priority=priority, deadline=deadline)
        residency.complete_transfer(resource(index), f"gpu:{index}")
    residency.set_pinned(resource(4), True)

    assert residency._eviction_candidate(ResourceKind.EXPERT, set()) == resource(2)
    assert residency._eviction_candidate(ResourceKind.EXPERT, {resource(2)}) == resource(3)


def test_full_cache_prefetch_selects_eviction_candidate_once(monkeypatch):
    residency = ResidencyManager({ResourceKind.EXPERT: 1, ResourceKind.KV: 1})
    residency.register_cpu(resource(0), "a", 1)
    residency.register_cpu(resource(1), "b", 1)
    assert residency.begin_transfer(resource(0), priority=1, deadline=5)
    residency.complete_transfer(resource(0), "gpu:a")
    calls = 0
    original = residency._eviction_candidate

    def counted_candidate(kind, protected):
        nonlocal calls
        calls += 1
        return original(kind, protected)

    monkeypatch.setattr(residency, "_eviction_candidate", counted_candidate)
    assert residency.begin_transfer(resource(1), priority=2, deadline=1)
    assert calls == 1
    assert residency.state(resource(0)) == ResourceState.CPU_ONLY
    assert residency.state(resource(1)) == ResourceState.IN_FLIGHT
