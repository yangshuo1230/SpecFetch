import pytest

from src.runtime.memory_queue import MemoryRequestQueue, QueueUpdate, ResourceKey, ResourceKind


def key(object_id, kind=ResourceKind.KV):
    return ResourceKey(kind=kind, layer=0, object_id=object_id, request_id="r0")


def add(queue, object_id, probability, deadline, consumer="r0"):
    return queue.upsert(
        key(object_id),
        consumer=consumer,
        probability=probability,
        deadline=deadline,
        size_bytes=1024,
        miss_cost_ms=1.0,
    )


def test_probability_and_urgency_order_speculative_requests():
    queue = MemoryRequestQueue()
    add(queue, 1, probability=0.9, deadline=4)
    add(queue, 2, probability=0.5, deadline=1)
    assert queue.pop().key == key(2)
    assert queue.pop().key == key(1)


def test_demand_miss_jumps_to_front():
    queue = MemoryRequestQueue()
    add(queue, 1, probability=1.0, deadline=1)
    queue.promote_demand(key(2), consumer="r0", size_bytes=1024, miss_cost_ms=1.0)
    request = queue.pop()
    assert request.key == key(2)
    assert request.demand


def test_duplicate_resource_merges_consumers_and_reorders():
    queue = MemoryRequestQueue()
    add(queue, 1, probability=0.2, deadline=5)
    add(queue, 2, probability=0.6, deadline=5)
    add(queue, 1, probability=0.7, deadline=2, consumer="r1")
    request = queue.pop()
    assert request.key == key(1)
    assert request.expected_uses == pytest.approx(0.9)
    assert request.deadline == 2


def test_cancel_one_consumer_keeps_shared_request():
    queue = MemoryRequestQueue()
    add(queue, 1, 0.5, 3)
    add(queue, 1, 0.7, 3, consumer="r1")
    assert queue.cancel(key(1), consumer="r0")
    assert queue.snapshot()[0].consumer_probabilities == {"r1": 0.7}


def test_step_cannot_move_backwards():
    queue = MemoryRequestQueue()
    queue.set_step(3)
    with pytest.raises(ValueError, match="backwards"):
        queue.set_step(2)


def test_batch_upsert_merges_consumers_before_worker_observes_queue():
    queue = MemoryRequestQueue()
    queue.upsert_many(
        [
            QueueUpdate(key(1), "r0", 0.2, 5, 1024, 1.0),
            QueueUpdate(key(1), "r1", 0.7, 2, 1024, 1.0),
            QueueUpdate(key(2), "r0", 0.5, 3, 1024, 1.0),
        ]
    )
    first = queue.pop()
    assert first.key == key(1)
    assert first.expected_uses == pytest.approx(0.9)
    assert first.deadline == 2


def test_pop_many_batches_demands_without_delaying_them_for_speculation():
    queue = MemoryRequestQueue()
    add(queue, 1, 0.9, 1)
    queue.promote_demand(key(2), consumer="r0", size_bytes=1024, miss_cost_ms=1)
    queue.promote_demand(key(3), consumer="r0", size_bytes=1024, miss_cost_ms=1)
    demands = queue.pop_many(8)
    assert {item.key for item in demands} == {key(2), key(3)}
    assert all(item.demand for item in demands)
    assert queue.pop().key == key(1)
