import os

import pytest
import torch

from src.runtime.expert import ExpertWeights
from src.runtime.memory_queue import ResourceKey, ResourceKind
from src.runtime.transfer import CudaTransferBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("SPECFETCH_RUN_CUDA_TESTS") != "1",
    reason="explicit opt-in prevents tests from disturbing a busy GPU",
)


def test_cuda_backend_copies_nested_expert_payload():
    source = ExpertWeights(*(torch.randn(4, 4, pin_memory=True) for _ in range(3)))
    backend = CudaTransferBackend("cuda:0")
    result = backend.copy_to_gpu(ResourceKey(ResourceKind.EXPERT, layer=0, object_id=0), source)
    assert result.gate.is_cuda
    assert torch.equal(result.gate.cpu(), source.gate)
