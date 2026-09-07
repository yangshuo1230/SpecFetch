import torch
import torch.nn.functional as F

from src.runtime.expert import (
    ExpertRegistry,
    ExpertWeights,
    OffloadedExpertExecutor,
    enqueue_expert_predictions,
    expert_key,
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


def runtime_with(experts):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: len(experts), ResourceKind.KV: 2})
    worker = TransferWorker(queue, residency, IdentityBackend())
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


def test_predictions_merge_shared_expert_consumers():
    experts = [make_expert(0), make_expert(1)]
    runtime, registry, worker = runtime_with(experts)
    # Stop the worker so the queue snapshot is deterministic for this policy test.
    worker.close()

    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 2})
    registry = ExpertRegistry(Source(experts), residency)
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
