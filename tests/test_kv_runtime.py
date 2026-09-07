import torch

from src.runtime.config import RuntimeConfig
from src.runtime.hybrid_attention import attention_output
from src.runtime.kv_cache import RequestLayerKV
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKind
from src.runtime.residency import ResidencyManager, ResourceState
from src.runtime.transfer import OffloadRuntime, TransferWorker


class IdentityBackend:
    def copy_to_gpu(self, key, value):
        return value


def build_cache(config):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.KV: config.kv_cache_slots, ResourceKind.EXPERT: 2})
    worker = TransferWorker(queue, residency, IdentityBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    worker.start()
    cache = RequestLayerKV("r0", 0, config, residency, runtime, pin_cpu=False)
    return cache, residency, worker


def test_layout_keeps_sink_and_bounded_recent_window():
    config = RuntimeConfig(sink_tokens=2, recent_tokens=4, kv_chunk_tokens=2)
    cache, residency, worker = build_cache(config)
    key = torch.arange(20.0).reshape(10, 1, 2)
    cache.initialize(key, key + 1)
    assert len(cache.sink[0]) == 2
    assert len(cache.recent[0]) == 4
    assert len(cache.old) == 2
    assert all(residency.state(item) == ResourceState.CPU_ONLY for item in cache.old.values())

    cache.append(torch.ones(2, 1, 2), torch.ones(2, 1, 2))
    assert len(cache.recent[0]) <= config.recent_tokens + config.kv_chunk_tokens - 1
    assert len(cache.old) == 3
    worker.close()


def test_sparse_attention_matches_dense_when_all_old_chunks_selected():
    config = RuntimeConfig(
        sink_tokens=2,
        recent_tokens=2,
        kv_chunk_tokens=2,
        predicted_mass_threshold=0.95,
        marginal_mass_threshold=1.0,
        marginal_patience=1,
        minimum_old_chunks=2,
    )
    cache, _, worker = build_cache(config)
    generator = torch.Generator().manual_seed(3)
    key = torch.randn(8, 1, 2, generator=generator)
    value = torch.randn(8, 1, 2, generator=generator)
    query = torch.randn(2, 2, generator=generator)
    cache.initialize(key, value)
    mass = {0: 0.6, 1: 0.4}
    cache.enqueue(mass, deadline=1, miss_cost_ms=1)
    actual = cache.sparse_attention(query, mass, miss_cost_ms=1)
    expected = attention_output(query, [(key, value)])
    assert actual.selected_old_chunks == [0, 1]
    assert torch.allclose(actual.output, expected, atol=1e-5)
    worker.close()


def test_reconcile_evicts_chunks_not_predicted_for_next_decode():
    config = RuntimeConfig(sink_tokens=2, recent_tokens=2, kv_chunk_tokens=2)
    cache, residency, worker = build_cache(config)
    values = torch.randn(8, 1, 2)
    cache.initialize(values, values)
    cache.enqueue({0: 0.8, 1: 0.2}, deadline=1, miss_cost_ms=1)
    cache.sparse_attention(torch.ones(2, 2), {0: 0.8, 1: 0.2}, miss_cost_ms=1)
    cache.reconcile({1: 1.0}, deadline=2, miss_cost_ms=1)
    assert residency.state(cache.old[0]) == ResourceState.CPU_ONLY
    worker.close()
