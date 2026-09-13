import pytest

from src.runtime.batch_scheduler import ContinuousBatchScheduler, ServingRequest


def request(name, output=2):
    return ServingRequest(name, [1, 2], output)


def test_continuous_batch_backfills_finished_slots():
    scheduler = ContinuousBatchScheduler(2)
    scheduler.submit(request("a", 1))
    scheduler.submit(request("b", 2))
    scheduler.submit(request("c", 1))
    assert [item.request_id for item in scheduler.admit()] == ["a", "b"]
    assert [item.request_id for item in scheduler.record_decode(["a"])] == ["a"]
    assert [item.request_id for item in scheduler.admit()] == ["c"]
    scheduler.record_decode(["b", "c"])
    scheduler.record_decode(["b"])
    assert scheduler.done
    assert [item.request_id for item in scheduler.completed] == ["a", "c", "b"]


def test_duplicate_request_ids_are_rejected():
    scheduler = ContinuousBatchScheduler(1)
    scheduler.submit(request("same"))
    with pytest.raises(ValueError, match="duplicate"):
        scheduler.submit(request("same"))
