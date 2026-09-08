import os

import pytest
import torch
import torch.nn.functional as F

from src.gpu_guard import require_idle_gpus
from src.runtime.expert import (
    ExpertRegistry,
    ExpertWeights,
    OffloadedExpertExecutor,
    optional_vllm_fused_moe,
)
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKey, ResourceKind
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import CudaTransferBackend, OffloadRuntime, TransferWorker

pytestmark = pytest.mark.skipif(
    os.environ.get("SPECFETCH_RUN_CUDA_TESTS") != "1",
    reason="explicit opt-in prevents tests from disturbing a busy GPU",
)


@pytest.fixture(scope="module", autouse=True)
def require_idle_selected_gpu():
    physical = int(os.environ.get("SPECFETCH_PHYSICAL_GPU_INDEX", "0"))
    require_idle_gpus(1000, 10, {physical})


def test_cuda_backend_copies_nested_expert_payload():
    source = ExpertWeights(*(torch.randn(4, 4, pin_memory=True) for _ in range(3)))
    backend = CudaTransferBackend("cuda:0")
    result = backend.copy_to_gpu(ResourceKey(ResourceKind.EXPERT, layer=0, object_id=0), source)
    assert result.gate.is_cuda
    assert torch.equal(result.gate.cpu(), source.gate)


def test_cuda_batched_packed_vllm_fused_moe_matches_torch(monkeypatch):
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setenv("VLLM_MOE_USE_DEEP_GEMM", "0")
    generator = torch.Generator().manual_seed(31)
    experts = [
        ExpertWeights(
            torch.randn(64, 128, dtype=torch.bfloat16, generator=generator).pin_memory(),
            torch.randn(64, 128, dtype=torch.bfloat16, generator=generator).pin_memory(),
            torch.randn(128, 64, dtype=torch.bfloat16, generator=generator).pin_memory(),
        )
        for _ in range(4)
    ]

    class Source:
        def get(self, layer, expert):
            del layer
            return experts[expert]

    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 4, ResourceKind.KV: 1})
    backend = CudaTransferBackend("cuda:0", expert_slots=4)
    worker = TransferWorker(queue, residency, backend)
    runtime = OffloadRuntime(queue, residency, worker)
    registry = ExpertRegistry(Source(), residency)
    fused = optional_vllm_fused_moe("vllm", backend)
    executor = OffloadedExpertExecutor(runtime, registry, top_k=2, fused_moe=fused)
    worker.start()
    hidden = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda:0")
    logits = torch.tensor(
        [[4.0, 3.0, 0.0, -1.0], [0.0, 4.0, 3.0, -1.0], [3.0, 0.0, 4.0, -1.0]],
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    actual = executor(hidden, logits, layer=0, request_ids=["a", "b", "c"])

    routing = torch.softmax(logits.float(), dim=-1)
    routing, selected = routing.topk(2, dim=-1)
    routing = (routing / routing.sum(dim=-1, keepdim=True)).to(hidden.dtype)
    expected = torch.zeros_like(hidden)
    for token in range(len(hidden)):
        for route in range(2):
            weights = experts[int(selected[token, route])]
            gate = F.linear(hidden[token], weights.gate.cuda())
            up = F.linear(hidden[token], weights.up.cuda())
            expected[token] += (
                F.linear(F.silu(gate) * up, weights.down.cuda()) * routing[token, route]
            )
    assert torch.allclose(actual, expected, atol=0.2, rtol=0.02)
    assert worker.metrics.maximum_transfer_batch >= 2
    worker.close()
