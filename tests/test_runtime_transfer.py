import time

from src.runtime.memory_queue import MemoryRequestQueue, ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState
from src.runtime.transfer import OffloadRuntime, TransferWorker


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
