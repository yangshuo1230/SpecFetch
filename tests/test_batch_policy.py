import pytest

from src.batch_policy import (
    allocate_private,
    allocate_shared,
    private_utility,
    shared_utility,
)


def test_dynamic_private_budget_moves_slots_between_requests():
    scores = [{0: 0.9, 1: 0.8}, {0: 0.2, 1: 0.1}]
    static, dynamic = allocate_private(scores, per_request_budget=1)
    assert static == [{0}, {0}]
    assert dynamic == [{0, 1}, set()]
    assert private_utility(dynamic, scores, required=1)["transfers"] == 2


def test_dynamic_shared_budget_rewards_batch_reuse():
    scores = [{0: 0.6, 1: 0.5}, {1: 0.5, 2: 0.6}]
    static, dynamic = allocate_shared(scores, per_request_budget=1)
    assert static == {0, 2}
    assert 1 in dynamic
    assert len(dynamic) == 2
    utility = shared_utility(dynamic, [{1: 1.0}, {1: 1.0}], required=1)
    assert utility["recall"] == 1.0
    assert utility["hits_per_transfer"] == 1.0


def test_private_utility_reports_attention_mass():
    utility = private_utility([{1}], [{0: 0.25, 1: 0.75}], required=1)
    assert utility["recall"] == 1.0
    assert utility["target_mass"] == pytest.approx(0.75)
