from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from src.runtime.expert import (
    ExpertRegistry,
    ExpertWeights,
    OffloadedExpertExecutor,
    enqueue_expert_predictions,
    expert_key,
    expert_prediction_requests,
)
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKind
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import OffloadRuntime, TransferWorker


class Source:
    def __init__(self, experts):
        self.experts = experts

    def get(self, layer, expert):
        return self.experts[expert]


class IdentityBackend:
    def copy_to_gpu(self, key, value):
        return value


def runtime_with(experts, capacity=None, backend=None):
    queue = MemoryRequestQueue()
    residency = ResidencyManager(
        {ResourceKind.EXPERT: capacity or len(experts), ResourceKind.KV: 2}
    )
    worker = TransferWorker(queue, residency, backend or IdentityBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    registry = ExpertRegistry(Source(experts), residency)
    worker.start()
    return runtime, registry, worker


def make_expert(seed):
    generator = torch.Generator().manual_seed(seed)
    return ExpertWeights(
        gate=torch.randn(3, 4, generator=generator),
        up=torch.randn(3, 4, generator=generator),
        down=torch.randn(4, 3, generator=generator),
    )


def test_offloaded_executor_matches_direct_moe():
    experts = [make_expert(0), make_expert(1), make_expert(2)]
    runtime, registry, worker = runtime_with(experts)
    hidden = torch.randn(2, 4, generator=torch.Generator().manual_seed(4))
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 2.0, 3.0]])
    executor = OffloadedExpertExecutor(runtime, registry, top_k=2)
    actual = executor(hidden, logits, layer=0, request_ids=["a", "b"])

    probabilities = torch.softmax(logits, dim=-1)
    weights, selected = probabilities.topk(2, dim=-1)
    weights /= weights.sum(-1, keepdim=True)
    expected = torch.zeros_like(hidden)
    for token in range(2):
        for route in range(2):
            expert = experts[selected[token, route]]
            value = F.silu(F.linear(hidden[token], expert.gate))
            value *= F.linear(hidden[token], expert.up)
            expected[token] += F.linear(value, expert.down) * weights[token, route]
    assert torch.allclose(actual, expected, atol=1e-5)
    worker.close()


def test_grouped_and_vectorized_executors_match():
    experts = [make_expert(0), make_expert(1), make_expert(2)]
    hidden = torch.randn(2, 4, generator=torch.Generator().manual_seed(7))
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 2.0, 3.0]])
    first_runtime, first_registry, first_worker = runtime_with(experts)
    second_runtime, second_registry, second_worker = runtime_with(experts)
    vectorized = OffloadedExpertExecutor(
        first_runtime, first_registry, top_k=2, vectorized_token_limit=8
    )
    grouped = OffloadedExpertExecutor(
        second_runtime, second_registry, top_k=2, vectorized_token_limit=0
    )
    first = vectorized(hidden, logits, layer=0, request_ids=["a", "b"])
    second = grouped(hidden, logits, layer=0, request_ids=["a", "b"])
    assert torch.allclose(first, second, atol=1e-5)
    first_worker.close()
    second_worker.close()


def test_fused_router_matches_reference_topk_and_normalization():
    experts = [make_expert(0), make_expert(1), make_expert(2)]
    hidden = torch.randn(2, 4, generator=torch.Generator().manual_seed(71))
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 2.0, 3.0]])
    reference_runtime, reference_registry, reference_worker = runtime_with(experts)
    fused_runtime, fused_registry, fused_worker = runtime_with(experts)
    calls = []

    def fake_fused_topk(**kwargs):
        calls.append(kwargs)
        weights = torch.softmax(kwargs["gating_output"].float(), dim=-1)
        weights, selected = weights.topk(kwargs["topk"], dim=-1)
        if kwargs["renormalize"]:
            weights /= weights.sum(dim=-1, keepdim=True)
        token_expert_indices = torch.zeros_like(selected, dtype=torch.int32)
        return weights, selected, token_expert_indices

    reference = OffloadedExpertExecutor(reference_runtime, reference_registry, top_k=2)
    fused = OffloadedExpertExecutor(
        fused_runtime,
        fused_registry,
        top_k=2,
        fused_topk=fake_fused_topk,
    )

    expected = reference(hidden, logits, layer=0, request_ids=["a", "b"])
    actual = fused(hidden, logits, layer=0, request_ids=["a", "b"])

    assert torch.allclose(actual, expected, atol=1e-5)
    assert len(calls) == 1
    assert calls[0]["renormalize"] is True
    reference_worker.close()
    fused_worker.close()


def test_expert_demand_planning_requires_cpu_route_ids():
    experts = [make_expert(0), make_expert(1), make_expert(2)]
    runtime, registry, worker = runtime_with(experts)
    executor = OffloadedExpertExecutor(runtime, registry, top_k=2)
    selected_cpu = torch.tensor([[0, 1], [2, 1]])

    loaded = executor._load(selected_cpu, 0, ["a", "b"])

    assert set(loaded) == {0, 1, 2}
    with pytest.raises(ValueError, match="CPU route IDs"):
        executor._load(torch.empty(1, device="meta", dtype=torch.long), 0, ["a"])
    for key, _ in loaded.values():
        runtime.release(key)
    worker.close()


def test_registry_reuses_canonical_preloaded_resource_key():
    experts = [make_expert(0)]
    _, registry, worker = runtime_with(experts)

    first = registry.ensure(0, 0)
    second = registry.ensure(0, 0)

    assert first is second
    assert registry._keys[(0, 0)] is first
    worker.close()


def test_grouped_prefill_streams_more_experts_than_cache_capacity():
    from src.runtime.transfer import PackedExpertSlots

    class PackedBackend:
        def __init__(self):
            self.slots = PackedExpertSlots(2, "cpu")

        def copy_to_gpu(self, key, value):
            return self.slots.acquire(key, value)

        def release_gpu(self, key, value):
            del value
            self.slots.release(key)

    experts = [make_expert(0), make_expert(1), make_expert(2)]
    hidden = torch.randn(3, 4, generator=torch.Generator().manual_seed(9))
    logits = torch.tensor([[4.0, 3.0, 0.0], [0.0, 4.0, 3.0], [3.0, 0.0, 4.0]])
    runtime, registry, worker = runtime_with(experts, capacity=2, backend=PackedBackend())
    executor = OffloadedExpertExecutor(runtime, registry, top_k=2, vectorized_token_limit=0)
    actual = executor(hidden, logits, layer=0, request_ids=["a", "b", "c"])

    probabilities = torch.softmax(logits, dim=-1)
    weights, selected = probabilities.topk(2, dim=-1)
    weights /= weights.sum(-1, keepdim=True)
    expected = torch.zeros_like(hidden)
    for token in range(3):
        for route in range(2):
            expert = experts[selected[token, route]]
            value = F.silu(F.linear(hidden[token], expert.gate))
            value *= F.linear(hidden[token], expert.up)
            expected[token] += F.linear(value, expert.down) * weights[token, route]
    assert torch.allclose(actual, expected, atol=1e-5)
    assert worker.metrics.maximum_transfer_batch == 2
    worker.close()


def test_fused_adapter_maps_logical_experts_to_packed_slots():
    from src.runtime.transfer import ExpertSlotMap, PackedExpertSlots

    class PackedBackend:
        def __init__(self):
            self.slots = PackedExpertSlots(3, "cpu")
            self.slot_maps = ExpertSlotMap("cpu")

        def copy_to_gpu(self, key, value):
            result = self.slots.acquire(key, value)
            self.slot_maps.update([(key, self.slots.slot_for(key))])
            return result

        def release_gpu(self, key, value):
            del value
            self.slot_maps.remove(key)
            self.slots.release(key)

        def packed_expert_weights(self):
            return self.slots.fused_weights()

        def expert_map(self, layer, num_experts):
            return self.slot_maps.get(layer, num_experts)

        def expert_slot(self, key):
            return self.slots.slot_for(key)

    def fake_fused_moe(**kwargs):
        hidden = kwargs["hidden_states"]
        w1 = kwargs["w1"]
        w2 = kwargs["w2"]
        weights = kwargs["topk_weights"]
        selected = kwargs["topk_ids"]
        expert_map = kwargs["expert_map"]
        width = w1.shape[1] // 2
        result = torch.zeros_like(hidden)
        for token in range(len(hidden)):
            for route in range(selected.shape[1]):
                expert = expert_map[selected[token, route]]
                gate_up = F.linear(hidden[token], w1[expert])
                activated = F.silu(gate_up[:width]) * gate_up[width:]
                result[token] += F.linear(activated, w2[expert]) * weights[token, route]
        return result

    experts = [make_expert(0), make_expert(1), make_expert(2)]
    backend = PackedBackend()
    runtime, registry, worker = runtime_with(experts, backend=backend)
    executor = OffloadedExpertExecutor(runtime, registry, top_k=2, fused_moe=fake_fused_moe)
    hidden = torch.randn(2, 4, generator=torch.Generator().manual_seed(10))
    logits = torch.tensor([[3.0, 2.0, 0.0], [0.0, 2.0, 3.0]])
    routing = torch.softmax(logits, dim=-1)
    routing, selected = routing.topk(2, dim=-1)
    routing /= routing.sum(dim=-1, keepdim=True)
    loaded = executor._load(selected, 0, ["a", "b"])
    actual = executor._fused(hidden, selected, routing, loaded, 0, global_num_experts=3)
    expected = executor._vectorized(hidden, selected, routing, loaded)
    assert torch.allclose(actual, expected, atol=1e-5)
    with pytest.raises(ValueError, match="expert count changed"):
        executor._fused(hidden, selected, routing, loaded, 0, global_num_experts=2)
    for key, _ in loaded.values():
        runtime.release(key)
    worker.close()


def test_predictions_merge_shared_expert_consumers():
    experts = [make_expert(0), make_expert(1)]
    runtime, registry, worker = runtime_with(experts)
    # Stop the worker so the queue snapshot is deterministic for this policy test.
    worker.close()

    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 2})
    registry = ExpertRegistry(Source(experts), residency)
    registry.ensure = Mock(wraps=registry.ensure)
    dormant_worker = TransferWorker(queue, residency, IdentityBackend())
    runtime = OffloadRuntime(queue, residency, dormant_worker)
    enqueue_expert_predictions(
        torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
        layer=0,
        request_ids=["a", "b"],
        top_k=1,
        deadline=2,
        miss_cost_ms=0.5,
        registry=registry,
        runtime=runtime,
    )
    request = queue.snapshot()[0]
    assert request.key == expert_key(0, 0)
    assert request.consumer_probabilities == {"a": 0.8999999761581421, "b": 0.800000011920929}
    assert registry.ensure.call_count == 1


def test_prediction_logits_apply_sigmoid_only_after_topk():
    experts = [make_expert(0), make_expert(1), make_expert(2)]
    _, registry, worker = runtime_with(experts)
    logits = torch.tensor([[-3.0, 2.0, 1.0], [4.0, -2.0, 0.5]])

    requests, _ = expert_prediction_requests(
        logits,
        layer=0,
        request_ids=["a", "b"],
        top_k=2,
        deadline=3,
        miss_cost_ms=0.5,
        registry=registry,
        logits=True,
    )

    selected = logits.topk(2, dim=1)
    assert [request.key.object_id for request in requests] == selected.indices.flatten().tolist()
    assert [request.probability for request in requests] == torch.sigmoid(
        selected.values
    ).flatten().tolist()
    worker.close()
