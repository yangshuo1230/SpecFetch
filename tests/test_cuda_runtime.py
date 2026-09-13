import os

import pytest
import torch
import torch.nn.functional as F

from src.gpu_guard import require_idle_gpus
from src.runtime.expert import (
    CpuRouteBuffer,
    ExpertRegistry,
    ExpertWeights,
    OffloadedExpertExecutor,
    optional_flash_kv_attention,
    optional_vllm_fused_add_rms_norm,
    optional_vllm_fused_moe,
    optional_vllm_fused_topk,
    optional_vllm_rms_norm,
    optional_vllm_rotary_embedding,
)
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKey, ResourceKind
from src.runtime.predictor import CpuFeatureBuffer
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


def test_cuda_backend_updates_persistent_expert_map_with_slot_reuse():
    backend = CudaTransferBackend("cuda:0", expert_slots=1)
    weights = ExpertWeights(*(torch.randn(4, 4, pin_memory=True) for _ in range(3)))
    first = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=3)
    second = ResourceKey(ResourceKind.EXPERT, layer=2, object_id=5)

    first_value = backend.copy_to_gpu(first, weights)
    mapping = backend.expert_map(2, 8)
    pointer = mapping.data_ptr()
    assert mapping.cpu().tolist() == [-1, -1, -1, 0, -1, -1, -1, -1]

    backend.release_gpu(first, first_value)
    backend.copy_to_gpu(second, weights)
    assert backend.expert_map(2, 8).data_ptr() == pointer
    assert mapping.cpu().tolist() == [-1, -1, -1, -1, -1, 0, -1, -1]


def test_cuda_batched_packed_vllm_fused_moe_matches_torch(monkeypatch):
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setenv("VLLM_MOE_USE_DEEP_GEMM", "0")
    monkeypatch.setenv("VLLM_MOE_USE_ACEXT", "0")
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
    residency = ResidencyManager({ResourceKind.EXPERT: 2, ResourceKind.KV: 1})
    backend = CudaTransferBackend("cuda:0", expert_slots=2)
    worker = TransferWorker(queue, residency, backend)
    runtime = OffloadRuntime(queue, residency, worker)
    registry = ExpertRegistry(Source(), residency)
    fused = optional_vllm_fused_moe("vllm", backend)
    fused_topk = optional_vllm_fused_topk("vllm", backend)
    executor = OffloadedExpertExecutor(
        runtime,
        registry,
        top_k=2,
        vectorized_token_limit=0,
        fused_moe=fused,
        fused_topk=fused_topk,
    )
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
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float())
    relative_l2 /= torch.linalg.vector_norm(expected.float())
    cosine = F.cosine_similarity(actual.float().reshape(1, -1), expected.float().reshape(1, -1))
    assert relative_l2 < 0.01
    assert cosine > 0.9999
    assert worker.metrics.maximum_transfer_batch == 2
    worker.close()


def test_vllm_rms_norm_matches_reference_for_batched_decode_shape():
    backend = CudaTransferBackend("cuda:0", expert_slots=1)
    fused = optional_vllm_rms_norm("vllm", backend)
    hidden = torch.randn(4, 1, 4, 128, dtype=torch.bfloat16, device="cuda:0")
    weight = torch.randn(128, dtype=torch.bfloat16, device="cuda:0")

    actual = fused(hidden, weight, 1e-6)
    normalized = hidden.float()
    normalized *= torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = weight * normalized.to(hidden.dtype)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_vllm_fused_add_rms_norm_matches_reference():
    backend = CudaTransferBackend("cuda:0", expert_slots=1)
    fused = optional_vllm_fused_add_rms_norm("vllm", backend)
    hidden = torch.randn(4, 1, 128, dtype=torch.bfloat16, device="cuda:0")
    residual = torch.randn_like(hidden)
    weight = torch.randn(128, dtype=torch.bfloat16, device="cuda:0")
    combined = hidden + residual
    normalized = combined.float()
    normalized *= torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = weight * normalized.to(hidden.dtype)

    actual, actual_residual = fused(hidden, residual, weight, 1e-6)

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_residual, combined, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("sequence_length", [1, 512])
def test_vllm_rotary_embedding_matches_native_qwen_layout(sequence_length):
    from types import SimpleNamespace

    backend = CudaTransferBackend("cuda:0", expert_slots=1)
    embedding = torch.nn.Embedding(1, 128, dtype=torch.bfloat16, device="cuda:0")
    model = SimpleNamespace(
        config=SimpleNamespace(
            head_dim=128,
            max_position_embeddings=8192,
            rope_theta=1_000_000.0,
            rope_scaling=None,
        ),
        model=SimpleNamespace(embed_tokens=embedding),
    )
    rotary = optional_vllm_rotary_embedding("vllm", backend, model)
    if sequence_length == 1:
        positions = torch.tensor([[511], [1023], [4095], [4096]], device="cuda:0")
    else:
        positions = torch.arange(
            3584,
            3584 + 4 * sequence_length,
            device="cuda:0",
        ).reshape(4, sequence_length)
    query = torch.randn(
        4,
        sequence_length,
        4,
        128,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    key = torch.randn(
        4,
        sequence_length,
        2,
        128,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    expected_query, expected_key = rotary.forward_native(positions, query.clone(), key.clone())

    actual_query, actual_key = rotary(positions, query, key)

    assert torch.allclose(actual_query, expected_query, atol=2e-2, rtol=2e-2)
    assert torch.allclose(actual_key, expected_key, atol=2e-2, rtol=2e-2)


def test_flash_attention_fuses_resident_kv_append_and_gqa_decode():
    backend = CudaTransferBackend("cuda:0", expert_slots=1)
    flash_attention = optional_flash_kv_attention("vllm", backend)
    generator = torch.Generator(device="cuda:0").manual_seed(41)
    query = torch.randn(4, 1, 8, 128, dtype=torch.bfloat16, device="cuda:0", generator=generator)
    key = torch.randn(4, 1, 2, 128, dtype=torch.bfloat16, device="cuda:0", generator=generator)
    value = torch.randn_like(key)
    key_cache = torch.randn(
        4, 16, 2, 128, dtype=torch.bfloat16, device="cuda:0", generator=generator
    )
    value_cache = torch.randn_like(key_cache)
    reference_key = key_cache.clone()
    reference_value = value_cache.clone()
    reference_key[:, 7:8].copy_(key)
    reference_value[:, 7:8].copy_(value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        reference_key[:, :8].transpose(1, 2),
        reference_value[:, :8].transpose(1, 2),
        enable_gqa=True,
    ).transpose(1, 2)

    actual = flash_attention(
        query,
        key_cache,
        value_cache,
        k=key,
        v=value,
        cache_seqlens=7,
        causal=True,
    )

    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(key_cache[:, 7:8], key)
    assert torch.equal(value_cache[:, 7:8], value)


def test_feature_signal_uses_reusable_pinned_staging():
    buffer = CpuFeatureBuffer()
    first = buffer.copy(torch.randn(4, 28, 1024, dtype=torch.bfloat16, device="cuda:0"))
    compact_pointer = buffer.compact.data_ptr()
    float_pointer = first.data_ptr()
    second = buffer.copy(torch.randn(4, 28, 1024, dtype=torch.bfloat16, device="cuda:0"))

    assert buffer.compact.is_pinned()
    assert buffer.compact.data_ptr() == compact_pointer
    assert second.data_ptr() == float_pointer
    assert second.dtype == torch.float32


def test_actual_routes_use_reusable_pinned_snapshot():
    buffer = CpuRouteBuffer()
    first = buffer.copy(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda:0"))
    pointer = first.data_ptr()
    second = buffer.copy(torch.tensor([[5, 6], [7, 8]], dtype=torch.int32, device="cuda:0"))

    assert second.is_pinned()
    assert second.data_ptr() == pointer
    assert torch.equal(second, torch.tensor([[5, 6], [7, 8]], dtype=torch.int32))
