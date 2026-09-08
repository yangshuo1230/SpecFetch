import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from src.runtime.config import RuntimeConfig
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKind
from src.runtime.qwen3_engine import Qwen3SparseOffloadEngine, StepPredictions
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import OffloadRuntime, TransferWorker


class IdentityBackend:
    def copy_to_gpu(self, key, value):
        return value


def tiny_model():
    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        norm_topk_prob=True,
        hidden_act="silu",
        attention_dropout=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return Qwen3MoeForCausalLM(config).eval()


def build_engine(model):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 8, ResourceKind.KV: 32})
    worker = TransferWorker(queue, residency, IdentityBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    runtime_config = RuntimeConfig(
        sink_tokens=2,
        recent_tokens=8,
        kv_chunk_tokens=2,
        expert_cache_slots=8,
        kv_cache_slots=32,
    )
    engine = Qwen3SparseOffloadEngine.from_transformers_model(
        model, runtime, residency, runtime_config
    )
    worker.start()
    return engine, worker


def test_prefill_matches_transformers_dense_reference():
    torch.manual_seed(11)
    model = tiny_model()
    reference = tiny_model()
    reference.load_state_dict(model.state_dict())
    engine, worker = build_engine(model)
    assert engine.moe_backend == "torch"
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    expected = reference(tokens, use_cache=False).logits
    actual = engine.prefill(tokens, ["a", "b"])
    assert torch.allclose(actual.logits, expected, atol=2e-5, rtol=2e-5)
    worker.close()


def test_torch_backend_moe_warmup_is_a_noop():
    engine, worker = build_engine(tiny_model())
    assert engine.warmup_moe([8, 2]) is False
    assert not engine.residency.resident_keys()
    worker.close()


def test_decode_matches_full_sequence_when_all_kv_is_resident():
    torch.manual_seed(12)
    model = tiny_model()
    reference = tiny_model()
    reference.load_state_dict(model.state_dict())
    engine, worker = build_engine(model)
    prefix = torch.tensor([[1, 2, 3], [3, 2, 1]])
    state = engine.prefill(prefix, ["a", "b"]).state
    next_tokens = torch.tensor([4, 5])
    actual = engine.decode(next_tokens, state, StepPredictions()).logits[:, -1]
    full = torch.cat((prefix, next_tokens[:, None]), dim=1)
    expected = reference(full, use_cache=False).logits[:, -1]
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)
    worker.close()


def test_admitted_state_can_be_added_and_completed_state_removed():
    torch.manual_seed(13)
    engine, worker = build_engine(tiny_model())
    first = engine.prefill(torch.tensor([[1, 2, 3]]), ["a"]).state
    second = engine.prefill(torch.tensor([[4, 5]]), ["b"]).state
    engine.add_state(first, second)
    assert first.request_ids == ["a", "b"]
    assert first.lengths == [3, 2]
    engine.decode(torch.tensor([6, 7]), first, StepPredictions())
    engine.remove_requests(first, ["a"])
    assert first.request_ids == ["b"]
    assert all(request_id == "b" for request_id, _ in first.kv)
    engine.remove_requests(first, ["b"])
    assert not first.request_ids
    worker.close()
