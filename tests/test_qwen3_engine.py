from unittest.mock import Mock, call, patch

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

from src.runtime.config import RuntimeConfig
from src.runtime.expert import ExpertWeights, expert_prediction_requests
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKind
from src.runtime.qwen3_engine import Qwen3SparseOffloadEngine, StepPredictions
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import OffloadRuntime, TransferWorker


class IdentityBackend:
    def copy_to_gpu(self, key, value):
        return value


class BatchBackend(IdentityBackend):
    def __init__(self):
        self.batches = []

    def copy_many_to_gpu(self, items):
        self.batches.append([key for key, _ in items])
        return [value for _, value in items]


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


def build_engine(
    model,
    *,
    speculative_expert_budget=None,
    speculative_kv_budget=None,
    speculative_layer_lookahead=None,
    backend=None,
    kv_cache_slots=32,
    kv_storage="sparse",
    resident_kv_capacity_tokens=None,
):
    queue = MemoryRequestQueue()
    residency = ResidencyManager({ResourceKind.EXPERT: 8, ResourceKind.KV: kv_cache_slots})
    worker = TransferWorker(queue, residency, backend or IdentityBackend())
    runtime = OffloadRuntime(queue, residency, worker)
    runtime_config = RuntimeConfig(
        sink_tokens=2,
        recent_tokens=8,
        kv_chunk_tokens=2,
        expert_cache_slots=8,
        kv_cache_slots=kv_cache_slots,
        speculative_expert_budget=speculative_expert_budget,
        speculative_kv_budget=speculative_kv_budget,
        speculative_layer_lookahead=speculative_layer_lookahead,
        kv_storage=kv_storage,
        resident_kv_capacity_tokens=resident_kv_capacity_tokens,
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
    actual = engine.prefill(tokens, ["a", "b"], full_logits=True)
    assert torch.allclose(actual.logits, expected, atol=2e-5, rtol=2e-5)
    assert all(
        projection.weight.numel() == 0
        for layer in engine.model.model.layers
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)
    )
    assert [weight.shape for weight in engine._qkv_weights] == [(32, 16), (32, 16)]
    serving = engine.prefill(tokens, ["c", "d"])
    assert serving.logits.shape == expected[:, -1:].shape
    assert torch.allclose(serving.logits, expected[:, -1:], atol=2e-5, rtol=2e-5)
    worker.close()


def test_torch_backend_moe_warmup_is_a_noop():
    engine, worker = build_engine(tiny_model())
    assert engine.warmup_moe([8, 2]) is False
    assert not engine.residency.resident_keys()
    worker.close()


def test_expert_maps_are_preinitialized_for_every_vllm_layer():
    engine, worker = build_engine(tiny_model())
    maps = Mock(return_value=torch.empty(4, dtype=torch.int32))
    engine.moe_backend = "vllm"
    engine.runtime.worker.backend.expert_map = maps

    assert engine.initialize_expert_maps()
    assert maps.call_args_list == [call(0, 4), call(1, 4)]
    worker.close()


def test_engine_preallocates_expert_storage_for_any_compute_backend():
    engine, worker = build_engine(tiny_model())
    initialize = Mock(return_value=True)
    engine.runtime.worker.backend.initialize_expert_storage = initialize

    assert engine.initialize_expert_storage()
    payload = initialize.call_args.args[0]
    assert isinstance(payload, ExpertWeights)
    assert payload.gate.shape == (8, 16)
    worker.close()


def test_resident_flash_attention_warmup_uses_declared_decode_shape():
    engine, worker = build_engine(
        tiny_model(),
        kv_storage="resident",
        resident_kv_capacity_tokens=16,
    )
    calls = []

    def fake_flash(query, key_cache, value_cache, **kwargs):
        calls.append((query.shape, key_cache.shape, value_cache.shape, kwargs))
        return query

    engine._flash_attention = fake_flash
    engine.attention_backend = "flash_kvcache"

    assert engine.warmup_resident_attention(2, 12)
    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (2, 1, 4, 4)
    assert key_shape == value_shape == (2, 16, 2, 4)
    assert kwargs["k"].shape == kwargs["v"].shape == (2, 1, 2, 4)
    assert kwargs["cache_seqlens"] == 12
    assert kwargs["causal"] is True
    worker.close()


def test_engine_routes_every_norm_through_resolved_kernel():
    engine, worker = build_engine(tiny_model())
    calls = []

    def reference_rms_norm(hidden, weight, epsilon):
        calls.append(hidden.shape)
        normalized = hidden.float()
        normalized *= torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + epsilon)
        return weight * normalized.to(hidden.dtype)

    engine._rms_norm = reference_rms_norm
    output = engine.prefill(torch.tensor([[1, 2, 3]]), ["a"])
    engine.decode(output.logits[:, -1].argmax(-1), output.state, StepPredictions())

    assert len(calls) == 18  # Input/post/Q/K/final norms in both 2-layer passes.
    worker.close()


def test_engine_fuses_residual_add_norm_boundaries():
    engine, worker = build_engine(tiny_model())
    standalone_calls = []
    fused_calls = []

    def reference_rms_norm(hidden, weight, epsilon):
        standalone_calls.append(hidden.shape)
        normalized = hidden.float()
        normalized *= torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + epsilon)
        return weight * normalized.to(hidden.dtype)

    def reference_add_rms_norm(hidden, residual, weight, epsilon):
        fused_calls.append(hidden.shape)
        combined = hidden + residual
        return reference_rms_norm(combined, weight, epsilon), combined

    engine._rms_norm = reference_rms_norm
    engine._fused_add_rms_norm = reference_add_rms_norm
    output = engine.prefill(torch.tensor([[1, 2, 3]]), ["a"])
    engine.decode(output.logits[:, -1].argmax(-1), output.state, StepPredictions())

    assert len(fused_calls) == 8  # Four residual boundaries per 2-layer pass.
    assert len(standalone_calls) == 18  # Also includes eight Q/K norms.
    worker.close()


def test_engine_routes_every_layer_through_resolved_rotary_kernel():
    engine, worker = build_engine(tiny_model())
    calls = []

    def reference_rotary(positions, query, key):
        calls.append((positions.shape, query.shape, key.shape))
        cos, sin = engine.model.model.rotary_emb(query, positions)
        rotated_query, rotated_key = apply_rotary_pos_emb(
            query.transpose(1, 2),
            key.transpose(1, 2),
            cos,
            sin,
        )
        return rotated_query.transpose(1, 2), rotated_key.transpose(1, 2)

    engine._rotary_embedding = reference_rotary
    output = engine.prefill(torch.tensor([[1, 2, 3]]), ["a"])
    engine.decode(output.logits[:, -1].argmax(-1), output.state, StepPredictions())

    assert len(calls) == 4
    assert calls[-1] == (torch.Size([1, 1]), torch.Size([1, 1, 4, 4]), torch.Size([1, 1, 2, 4]))
    worker.close()


def test_prediction_window_submits_expert_and_kv_requests_as_one_queue_batch():
    engine, worker = build_engine(tiny_model())
    state = engine.prefill(torch.tensor([[1] * 12]), ["a"]).state
    prediction = StepPredictions(
        kv={
            ("a", layer): {chunk: 1.0 for chunk in state.kv[("a", layer)].old} for layer in range(2)
        },
        experts={layer: torch.tensor([[0.7, 0.2, 0.1, 0.0]]) for layer in range(2)},
    )
    before = worker.metrics.prefetch_batches
    engine.enqueue_predictions(state, [prediction])
    assert worker.metrics.prefetch_batches - before == 1
    assert state.speculative_consumers
    engine.remove_requests(state, ["a"])
    worker.close()


def test_prediction_items_deduplicate_buckets_before_building_requests():
    engine, worker = build_engine(
        tiny_model(),
        kv_storage="resident",
        resident_kv_capacity_tokens=16,
    )
    state = engine.prefill(torch.tensor([[1] * 12]), ["a"]).state
    prediction = StepPredictions(experts={0: torch.tensor([[0.7, 0.2, 0.1, 0.0]])})

    with patch(
        "src.runtime.qwen3_engine.expert_prediction_requests",
        wraps=expert_prediction_requests,
    ) as prediction_requests:
        engine._enqueue_prediction_items(
            state,
            [(1, prediction, 0), (1, prediction, 0)],
        )

    assert prediction_requests.call_count == 1
    assert worker.metrics.prefetch_candidates == 2
    assert worker.metrics.prefetch_requests == 2
    assert len(state.speculative_consumers) == 2
    engine.remove_requests(state, ["a"])
    worker.close()


def test_resident_prediction_window_skips_kv_bookkeeping_but_submits_experts():
    engine, worker = build_engine(
        tiny_model(),
        kv_storage="resident",
        resident_kv_capacity_tokens=16,
    )
    state = engine.prefill(torch.tensor([[1] * 12]), ["a"]).state
    for cache in state.kv.values():
        cache.prefetch_requests = Mock(side_effect=AssertionError("resident KV cannot prefetch"))
        cache.retain_predicted = Mock(side_effect=AssertionError("resident KV cannot evict"))
    prediction = StepPredictions(
        kv={("a", layer): {0: 1.0} for layer in range(2)},
        experts={layer: torch.tensor([[0.7, 0.2, 0.1, 0.0]]) for layer in range(2)},
    )

    engine.enqueue_predictions(state, [prediction])

    assert state.speculative_consumers
    assert all(key.kind == ResourceKind.EXPERT for key, _ in state.speculative_consumers)
    engine.remove_requests(state, ["a"])
    worker.close()


def test_prediction_budget_caps_unique_resources_and_preserves_shared_consumers():
    engine, worker = build_engine(
        tiny_model(), speculative_expert_budget=1, speculative_kv_budget=1
    )
    state = engine.prefill(torch.tensor([[1] * 12, [2] * 12]), ["a", "b"]).state
    prediction = StepPredictions(
        kv={
            (request_id, layer): {chunk: 1.0 for chunk in state.kv[(request_id, layer)].old}
            for request_id in ("a", "b")
            for layer in range(2)
        },
        experts={layer: torch.tensor([[0.7, 0.2, 0.1, 0.0]] * 2) for layer in range(2)},
    )

    engine.enqueue_predictions(state, [prediction])

    admitted = state.speculative_consumers
    admitted_keys = {key for key, _ in admitted}
    assert sum(key.kind == ResourceKind.EXPERT for key in admitted_keys) == 1
    assert sum(key.kind == ResourceKind.KV for key in admitted_keys) == 1
    assert len(admitted) == 3  # one shared expert has both request consumers
    assert worker.metrics.prefetch_candidates == 12
    assert worker.metrics.prefetch_requests == 3
    assert worker.metrics.prefetch_budget_dropped == 9
    assert worker.metrics.maximum_prefetch_candidates == 12
    assert worker.metrics.maximum_prefetch_admitted == 3
    engine.remove_requests(state, ["a", "b"])
    worker.close()


def test_decode_rolls_prediction_admission_forward_by_layer():
    engine, worker = build_engine(tiny_model(), speculative_layer_lookahead=1)
    state = engine.prefill(torch.tensor([[1] * 12]), ["a"]).state
    prediction = StepPredictions(
        kv={
            ("a", layer): {chunk: 1.0 for chunk in state.kv[("a", layer)].old} for layer in range(2)
        },
        experts={layer: torch.tensor([[0.7, 0.2, 0.1, 0.0]]) for layer in range(2)},
    )
    calls = []
    enqueue = engine._enqueue_prediction_items

    def record_items(state, items):
        items = tuple(items)
        calls.append(tuple((horizon, layer) for horizon, _, layer in items))
        enqueue(state, items)

    engine._enqueue_prediction_items = record_items
    engine.decode(torch.tensor([3]), state, prediction)

    assert calls == [((1, 0),), ((1, 1),)]
    assert worker.metrics.prefetch_candidates == 6
    assert worker.metrics.maximum_prefetch_candidates == 3
    assert not state.speculative_consumers
    engine.remove_requests(state, ["a"])
    worker.close()


def test_retiring_layer_cancels_only_current_token_consumers():
    engine, worker = build_engine(tiny_model())
    state = engine.prefill(torch.tensor([[1] * 12]), ["a"]).state
    layer_zero = state.kv[("a", 0)].old[0]
    layer_one = state.kv[("a", 1)].old[0]
    state.speculative_consumers.extend(
        [
            (layer_zero, "a@1"),
            (layer_zero, "a@2"),
            (layer_one, "a@1"),
        ]
    )
    cancel_many = Mock()
    engine.runtime.cancel_many = cancel_many

    engine._retire_prediction_layer(state, 0)

    cancel_many.assert_called_once_with([(layer_zero, "a@1")])
    assert state.speculative_consumers == [
        (layer_zero, "a@2"),
        (layer_one, "a@1"),
    ]
    engine.runtime.cancel_many = Mock()
    engine.remove_requests(state, ["a"])
    worker.close()


def test_decode_retains_next_token_consumers_after_current_layer_retires():
    engine, worker = build_engine(tiny_model(), speculative_layer_lookahead=2)
    state = engine.prefill(torch.tensor([[1] * 12]), ["a"]).state
    prediction = StepPredictions(
        kv={
            ("a", layer): {chunk: 1.0 for chunk in state.kv[("a", layer)].old} for layer in range(2)
        },
        experts={layer: torch.tensor([[0.7, 0.2, 0.1, 0.0]]) for layer in range(2)},
    )

    engine.decode(torch.tensor([3]), state, [prediction, prediction])

    assert state.speculative_consumers
    assert {consumer for _, consumer in state.speculative_consumers} == {"a@2"}
    assert {key.layer for key, _ in state.speculative_consumers} == {0}
    engine.remove_requests(state, ["a"])
    worker.close()


def test_rolling_prediction_window_submits_only_new_tail_horizon():
    engine, worker = build_engine(tiny_model())
    state = engine.prefill(torch.tensor([[1] * 12]), ["a"]).state
    prediction = StepPredictions(
        kv={
            ("a", layer): {chunk: 1.0 for chunk in state.kv[("a", layer)].old} for layer in range(2)
        },
        experts={layer: torch.tensor([[0.7, 0.2, 0.1, 0.0]]) for layer in range(2)},
    )

    engine.decode(
        torch.tensor([3]),
        state,
        [prediction, prediction],
        reuse_prediction_window=True,
    )
    first_candidates = worker.metrics.prefetch_candidates
    assert {consumer for _, consumer in state.speculative_consumers} == {"a@2"}
    with patch(
        "src.runtime.qwen3_engine.expert_prediction_requests",
        wraps=expert_prediction_requests,
    ) as prediction_requests:
        engine.decode(
            torch.tensor([4]),
            state,
            [prediction, prediction],
            reuse_prediction_window=True,
        )

    assert first_candidates == 12
    assert worker.metrics.prefetch_candidates - first_candidates == 6
    assert prediction_requests.call_count == 2
    assert sum(call.args[0].shape[0] for call in prediction_requests.call_args_list) == 2
    assert {consumer for _, consumer in state.speculative_consumers} == {"a@3"}
    engine.remove_requests(state, ["a"])
    worker.close()


def test_decode_batches_guaranteed_kv_across_requests_per_layer():
    backend = BatchBackend()
    engine, worker = build_engine(tiny_model(), backend=backend)
    state = engine.prefill(torch.tensor([[1] * 12, [2] * 12]), ["a", "b"]).state
    prediction = StepPredictions(
        kv={
            (request_id, layer): {chunk: 1.0 for chunk in state.kv[(request_id, layer)].old}
            for request_id in ("a", "b")
            for layer in range(2)
        }
    )
    backend.batches.clear()

    engine.decode(torch.tensor([3, 4]), state, prediction, prefetch=False)

    kv_batches = [batch for batch in backend.batches if batch and batch[0].kind == ResourceKind.KV]
    assert len(kv_batches) == 2
    assert all({key.request_id for key in batch} == {"a", "b"} for batch in kv_batches)
    engine.remove_requests(state, ["a", "b"])
    worker.close()


def test_decode_falls_back_when_cross_request_kv_batch_exceeds_capacity():
    backend = BatchBackend()
    engine, worker = build_engine(tiny_model(), backend=backend, kv_cache_slots=1)
    state = engine.prefill(torch.tensor([[1] * 12, [2] * 12]), ["a", "b"]).state
    prediction = StepPredictions(
        kv={
            (request_id, layer): {chunk: 1.0 for chunk in state.kv[(request_id, layer)].old}
            for request_id in ("a", "b")
            for layer in range(2)
        }
    )
    backend.batches.clear()

    engine.decode(torch.tensor([3, 4]), state, prediction, prefetch=False)

    kv_batches = [batch for batch in backend.batches if batch and batch[0].kind == ResourceKind.KV]
    assert len(kv_batches) == 4
    assert all(len(batch) == 1 for batch in kv_batches)
    engine.remove_requests(state, ["a", "b"])
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


def test_preallocated_resident_kv_decode_matches_full_sequence():
    torch.manual_seed(121)
    model = tiny_model()
    reference = tiny_model()
    reference.load_state_dict(model.state_dict())
    engine, worker = build_engine(
        model,
        kv_storage="resident",
        resident_kv_capacity_tokens=8,
    )
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    state = engine.prefill(tokens, ["a", "b"]).state

    for next_tokens in (torch.tensor([4, 5]), torch.tensor([6, 7])):
        actual = engine.decode(next_tokens, state, StepPredictions()).logits[:, -1]
        tokens = torch.cat((tokens, next_tokens[:, None]), dim=1)
        expected = reference(tokens, use_cache=False).logits[:, -1]
        assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)

    assert not engine.residency.resident_keys(ResourceKind.KV)
    worker.close()


def test_resident_kv_reports_exact_persistent_allocation_bytes():
    engine, worker = build_engine(
        tiny_model(),
        kv_storage="resident",
        resident_kv_capacity_tokens=8,
    )

    # 2 layers * K/V * batch 2 * capacity 8 * KV width 8 * fp32 bytes 4.
    estimated = engine.resident_kv_allocation_bytes(2)
    state = engine.prefill(torch.tensor([[1, 2], [3, 4]]), ["a", "b"]).state
    actual = sum(
        tensor.numel() * tensor.element_size()
        for buffers in state.resident_groups[0].layers.values()
        for tensor in buffers
    )
    assert estimated == actual == 2048
    with pytest.raises(ValueError, match="batch size"):
        engine.resident_kv_allocation_bytes(0)
    worker.close()


def test_engine_reports_exact_packed_expert_slot_payload():
    engine, worker = build_engine(tiny_model())

    # 8 slots * gate/up/down * hidden 16 * expert width 8 * fp32 bytes 4.
    assert engine.expert_slot_allocation_bytes() == 8 * 3 * 16 * 8 * 4
    worker.close()


def test_sparse_kv_reports_no_resident_allocation():
    engine, worker = build_engine(tiny_model())
    assert engine.resident_kv_allocation_bytes(2) == 0
    worker.close()


def test_resident_decode_batches_attention_and_uses_shared_storage():
    engine, worker = build_engine(
        tiny_model(),
        kv_storage="resident",
        resident_kv_capacity_tokens=8,
    )
    state = engine.prefill(torch.tensor([[1, 2, 3], [3, 2, 1]]), ["a", "b"]).state
    group = state.resident_groups[0]
    for layer, buffers in group.layers.items():
        assert buffers[0].shape == (2, 8, 2, 4)
        for row, request_id in enumerate(group.request_ids):
            cache_buffers = state.kv[(request_id, layer)]._resident_buffers
            assert cache_buffers is not None
            assert cache_buffers[0].data_ptr() == buffers[0][row].data_ptr()

    attention = torch.nn.functional.scaled_dot_product_attention
    with patch(
        "torch.nn.functional.scaled_dot_product_attention", wraps=attention
    ) as batched_attention:
        engine.decode(torch.tensor([4, 5]), state, StepPredictions())

    assert batched_attention.call_count == 2
    assert all(call.args[0].shape == (2, 4, 1, 4) for call in batched_attention.call_args_list)
    worker.close()


def test_resident_decode_fuses_kv_append_with_attention():
    engine, worker = build_engine(
        tiny_model(),
        kv_storage="resident",
        resident_kv_capacity_tokens=8,
    )
    state = engine.prefill(torch.tensor([[1, 2, 3], [3, 2, 1]]), ["a", "b"]).state
    calls = []

    def fake_flash_attention(
        query,
        key_cache,
        value_cache,
        *,
        k,
        v,
        cache_seqlens,
        softmax_scale,
        causal,
    ):
        calls.append((query.shape, k.shape, cache_seqlens, causal))
        end = cache_seqlens + k.shape[1]
        key_cache[:, cache_seqlens:end].copy_(k)
        value_cache[:, cache_seqlens:end].copy_(v)
        output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key_cache[:, :end].transpose(1, 2),
            value_cache[:, :end].transpose(1, 2),
            enable_gqa=query.shape[2] != key_cache.shape[2],
            scale=softmax_scale,
        )
        return output.transpose(1, 2)

    engine._flash_attention = fake_flash_attention
    engine.attention_backend = "flash_kvcache"
    engine.decode(torch.tensor([4, 5]), state, StepPredictions())

    assert len(calls) == 2
    assert all(
        call == (torch.Size([2, 1, 4, 4]), torch.Size([2, 1, 2, 4]), 3, True) for call in calls
    )
    assert state.resident_groups[0].length == 4
    worker.close()


def test_resident_groups_survive_admission_and_row_compaction():
    torch.manual_seed(122)
    model = tiny_model()
    reference = tiny_model()
    reference.load_state_dict(model.state_dict())
    engine, worker = build_engine(
        model,
        kv_storage="resident",
        resident_kv_capacity_tokens=8,
    )
    state = engine.prefill(torch.tensor([[1, 2, 3], [3, 2, 1]]), ["a", "b"]).state
    admitted = engine.prefill(torch.tensor([[4, 5]]), ["c"]).state
    engine.add_state(state, admitted)

    first_next = torch.tensor([6, 7, 8])
    actual = engine.decode(first_next, state, StepPredictions()).logits[:, -1]
    sequences = {
        "a": torch.tensor([1, 2, 3, 6]),
        "b": torch.tensor([3, 2, 1, 7]),
        "c": torch.tensor([4, 5, 8]),
    }
    expected = torch.stack(
        [
            reference(sequences[request_id][None], use_cache=False).logits[0, -1]
            for request_id in state.request_ids
        ]
    )
    assert torch.allclose(actual, expected, atol=3e-5, rtol=3e-5)

    first_group = state.resident_groups[0]
    engine.remove_requests(state, ["a"])
    assert first_group.request_ids == ["b"]
    for layer, buffers in first_group.layers.items():
        cache_buffers = state.kv[("b", layer)]._resident_buffers
        assert cache_buffers is not None
        assert cache_buffers[0].data_ptr() == buffers[0][0].data_ptr()

    second_next = torch.tensor([9, 10])
    actual = engine.decode(second_next, state, StepPredictions()).logits[:, -1]
    sequences["b"] = torch.cat((sequences["b"], second_next[:1]))
    sequences["c"] = torch.cat((sequences["c"], second_next[1:]))
    expected = torch.stack(
        [
            reference(sequences[request_id][None], use_cache=False).logits[0, -1]
            for request_id in state.request_ids
        ]
    )
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
    assert first.positions.tolist() == [[3], [2]]
    position_storage = first.positions.data_ptr()
    engine.decode(torch.tensor([6, 7]), first, StepPredictions())
    assert first.positions.data_ptr() == position_storage
    assert first.positions.tolist() == [[4], [3]]
    engine.remove_requests(first, ["a"])
    assert first.request_ids == ["b"]
    assert first.positions.tolist() == [[3]]
    assert all(request_id == "b" for request_id, _ in first.kv)
    engine.remove_requests(first, ["b"])
    assert not first.request_ids
    assert first.positions.shape == (0, 1)
    worker.close()
