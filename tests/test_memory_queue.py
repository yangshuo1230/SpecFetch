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


def test_batch_cancel_groups_consumers_and_reports_depleted_resources():
    queue = MemoryRequestQueue()
    add(queue, 1, 0.5, 3)
    add(queue, 1, 0.7, 2, consumer="r1")
    add(queue, 2, 0.4, 4)
    depleted = queue.cancel_many([(key(1), "r0"), (key(1), "r0"), (key(2), "r0")])

    assert depleted == {key(2)}
    remaining = queue.snapshot()
    assert len(remaining) == 1
    assert remaining[0].key == key(1)
    assert remaining[0].consumer_probabilities == {"r1": 0.7}


def test_step_cannot_move_backwards():
    queue = MemoryRequestQueue()
    queue.set_step(3)
    with pytest.raises(ValueError, match="backwards"):
        queue.set_step(2)


def test_step_rebuild_uses_linear_heapify_without_incremental_push(monkeypatch):
    queue = MemoryRequestQueue()
    add(queue, 1, probability=0.4, deadline=8)
    add(queue, 2, probability=0.6, deadline=5)
    add(queue, 3, probability=0.9, deadline=9)

    def unexpected_push(_request):
        raise AssertionError("step rebuild must not perform repeated heappush")

    monkeypatch.setattr(queue, "_push", unexpected_push)
    queue.set_step(4)

    assert [queue.pop().key for _ in range(3)] == [key(2), key(3), key(1)]


def test_multiple_step_updates_coalesce_before_worker_pop(monkeypatch):
    queue = MemoryRequestQueue()
    add(queue, 1, probability=0.4, deadline=8)
    add(queue, 2, probability=0.6, deadline=5)
    rebuilds = 0
    rebuild = queue._rebuild_heap_locked

    def counted_rebuild():
        nonlocal rebuilds
        rebuilds += 1
        rebuild()

    monkeypatch.setattr(queue, "_rebuild_heap_locked", counted_rebuild)
    queue.set_step(1)
    queue.set_step(2)
    queue.set_step(4)

    assert rebuilds == 0
    assert queue.pop().key == key(2)
    assert rebuilds == 1


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


def test_step_rebuild_reuses_cached_expected_uses(monkeypatch):
    queue = MemoryRequestQueue()
    add(queue, 1, 0.5, 8, consumer="a")
    add(queue, 1, 0.25, 6, consumer="b")
    request = queue.snapshot()[0]
    assert request.expected_uses == pytest.approx(0.75)

    class UnexpectedIteration(dict):
        def values(self):
            raise AssertionError("step rebuild must not rescan consumer probabilities")

    request.consumer_probabilities = UnexpectedIteration(request.consumer_probabilities)
    queue.set_step(4)
    assert queue.pop() is request


def test_batch_upsert_matches_sequential_duplicate_consumer_merges():
    updates = [
        QueueUpdate(key(1), "r0", 0.2, 8, 1024, 1.5),
        QueueUpdate(key(1), "r1", 0.7, 2, 1024, 3.0),
        QueueUpdate(key(1), "r0", 0.9, 6, 1024, 2.0, demand=True),
        QueueUpdate(key(2), "r2", 0.4, 4, 1024, 0.5),
    ]
    batched = MemoryRequestQueue()
    sequential = MemoryRequestQueue()

    returned = batched.upsert_many(updates)
    for update in updates:
        sequential.upsert(
            update.key,
            consumer=update.consumer,
            probability=update.probability,
            deadline=update.deadline,
            size_bytes=update.size_bytes,
            miss_cost_ms=update.miss_cost_ms,
            demand=update.demand,
        )

    assert returned[0] is returned[1] is returned[2]
    actual = {request.key: request for request in batched.snapshot()}
    expected = {request.key: request for request in sequential.snapshot()}
    assert actual.keys() == expected.keys()
    for resource_key, left in actual.items():
        right = expected[resource_key]
        assert left.consumer_probabilities == right.consumer_probabilities
        assert left.consumer_deadlines == right.consumer_deadlines
        assert left.deadline == right.deadline
        assert left.miss_cost_ms == right.miss_cost_ms
        assert left.demand == right.demand


def test_batch_upsert_pushes_each_shared_resource_once(monkeypatch):
    queue = MemoryRequestQueue()
    pushes = []
    original_push = queue._push

    def counting_push(request):
        pushes.append(request.key)
        original_push(request)

    monkeypatch.setattr(queue, "_push", counting_push)
    queue.upsert_many(
        [QueueUpdate(key(1), f"r{index}", 0.5, 20 - index, 1024, 1.0) for index in range(10)]
        + [QueueUpdate(key(2), "r0", 0.5, 3, 1024, 1.0)]
    )

    assert pushes == [key(1), key(2)]
    assert {request.key: request.deadline for request in queue.snapshot()} == {
        key(1): 11,
        key(2): 3,
    }


def test_pop_many_batches_demands_without_delaying_them_for_speculation():
    queue = MemoryRequestQueue()
    add(queue, 1, 0.9, 1)
    queue.promote_demand(key(2), consumer="r0", size_bytes=1024, miss_cost_ms=1)
    queue.promote_demand(key(3), consumer="r0", size_bytes=1024, miss_cost_ms=1)
    demands = queue.pop_many(8)
    assert {item.key for item in demands} == {key(2), key(3)}
    assert all(item.demand for item in demands)
    assert queue.pop().key == key(1)


def test_pop_many_bounds_only_speculative_microbatch():
    speculative = MemoryRequestQueue()
    for index in range(10):
        add(speculative, index, 0.5, index + 1)
    assert len(speculative.pop_many(8, speculative_maximum=3)) == 3
    assert len(speculative) == 7

    demands = MemoryRequestQueue()
    for index in range(10):
        demands.promote_demand(key(index), consumer="r0", size_bytes=1024, miss_cost_ms=1)
    batch = demands.pop_many(8, speculative_maximum=3)
    assert len(batch) == 8
    assert all(request.demand for request in batch)


def test_pop_many_bounds_speculation_by_bytes_without_throttling_small_objects():
    large = MemoryRequestQueue()
    large.upsert_many(
        [QueueUpdate(key(index), "r0", 0.5, index + 1, 9 * 2**20, 1.0) for index in range(10)]
    )
    assert len(large.pop_many(32, speculative_maximum_bytes=72 * 2**20)) == 8

    small = MemoryRequestQueue()
    small.upsert_many(
        [QueueUpdate(key(index), "r0", 0.5, index + 1, 128 * 2**10, 1.0) for index in range(32)]
    )
    assert len(small.pop_many(32, speculative_maximum_bytes=72 * 2**20)) == 32
