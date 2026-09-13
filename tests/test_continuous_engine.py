import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.runtime.batch_scheduler import ServingRequest
from src.runtime.continuous_engine import (
    ContinuousBatchRunner,
    RequestExecution,
    _split_predictions,
    _stack_token_rows,
    merge_predictions,
)
from src.runtime.predictor import DraftSignalProvider
from src.runtime.qwen3_engine import StepPredictions
from tests.test_qwen3_engine import build_engine, tiny_model


class PrefillOnlyEngine:
    def __init__(self):
        self.prefill_request_ids = []

    def prefill(self, input_ids, request_ids):
        self.prefill_request_ids.append(list(request_ids))
        state = type("State", (), {})()
        state.request_ids = list(request_ids)
        state.lengths = [input_ids.shape[1]] * len(request_ids)
        state.kv = {}
        state.speculative_consumers = []
        logits = torch.zeros((len(request_ids), 1, 4))
        logits[..., 2] = 1
        return type("Output", (), {"logits": logits, "state": state})()

    def add_state(self, state, admitted):
        state.request_ids.extend(admitted.request_ids)
        state.lengths.extend(admitted.lengths)

    def remove_requests(self, state, request_ids):
        removing = set(request_ids)
        retained = [
            (request_id, length)
            for request_id, length in zip(state.request_ids, state.lengths)
            if request_id not in removing
        ]
        state.request_ids = [item[0] for item in retained]
        state.lengths = [item[1] for item in retained]


def tiny_draft():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def test_pending_token_rows_use_one_flat_stack(monkeypatch):
    rows = [[torch.tensor(row * 2 + column) for column in range(2)] for row in range(4)]
    stack = torch.stack
    calls = []

    def counted_stack(tensors, *args, **kwargs):
        calls.append(list(tensors))
        return stack(calls[-1], *args, **kwargs)

    monkeypatch.setattr(torch, "stack", counted_stack)

    assert torch.equal(_stack_token_rows(rows), torch.arange(8).reshape(4, 2))
    assert len(calls) == 1
    with pytest.raises(ValueError, match="equal widths"):
        _stack_token_rows([rows[0], rows[1][:1]])


def test_continuous_runner_backfills_and_releases_qwen_states():
    torch.manual_seed(21)
    engine, worker = build_engine(tiny_model())
    draft = tiny_draft()
    providers = []
    draft_forwards = []
    hook = draft.register_forward_hook(lambda *_: draft_forwards.append(1))

    def provider_factory():
        provider = DraftSignalProvider(draft, None, lookahead=2)
        providers.append(provider)
        return provider

    runner = ContinuousBatchRunner(
        engine,
        provider_factory,
        max_batch_size=2,
        prefetch=False,
    )
    result = runner.run(
        [
            ServingRequest("a", [1, 2, 3], 2),
            ServingRequest("b", [4, 5, 6], 3),
            ServingRequest("c", [6, 7, 8, 9], 2),
        ]
    )
    assert {key: len(value) for key, value in result.generated_token_ids.items()} == {
        "a": 2,
        "b": 3,
        "c": 2,
    }
    assert result.admission_events == 2
    assert result.maximum_active_requests == 2
    assert result.prefill_batches == 2
    assert result.maximum_prefill_batch == 2
    assert result.draft_prefill_batches == 2
    assert result.maximum_draft_prefill_batch == 2
    assert result.draft_refresh_batches == 0
    assert result.maximum_draft_refresh_batch == 0
    # 两轮准入各只初始化一次批量 Draft provider。
    assert len(providers) == 2
    # 每轮均为一次前缀前向加两次批量 lookahead，不随组内请求数增加。
    assert len(draft_forwards) == 6
    hook.remove()
    assert result.request_timings["a"]["completion_seconds"] > 0
    assert (
        result.request_timings["c"]["admission_seconds"]
        > result.request_timings["b"]["time_to_first_token_seconds"]
    )
    assert all(
        timing["completion_seconds"] >= timing["time_to_first_token_seconds"]
        for timing in result.request_timings.values()
    )
    for timing in result.request_timings.values():
        assert timing["queue_seconds"] == timing["admission_seconds"]
        assert timing["prefill_seconds"] == (
            timing["time_to_first_token_seconds"] - timing["admission_seconds"]
        )
        assert timing["decode_service_seconds"] == (
            timing["completion_seconds"] - timing["time_to_first_token_seconds"]
        )
        assert timing["active_service_seconds"] == (
            timing["completion_seconds"] - timing["admission_seconds"]
        )
        assert timing["request_latency_seconds"] == timing["completion_seconds"]
    assert not any(key.request_id for key in engine.residency.resident_keys())
    worker.close()


def test_prefill_only_requests_report_peak_admitted_batch_without_draft():
    def unexpected_provider():
        raise AssertionError("单 token 请求不应创建 Draft provider")

    engine = PrefillOnlyEngine()
    runner = ContinuousBatchRunner(
        engine,
        unexpected_provider,
        max_batch_size=2,
        prefetch=False,
    )
    result = runner.run(
        [
            ServingRequest("a", [1], 1),
            ServingRequest("b", [2], 1),
            ServingRequest("c", [3], 1),
        ]
    )
    assert result.generated_token_ids == {"a": [2], "b": [2], "c": [2]}
    assert result.decode_cycles == 0
    assert result.admission_events == 2
    assert result.maximum_active_requests == 2
    assert result.prefill_batches == 2
    assert result.maximum_prefill_batch == 2
    assert result.draft_prefill_batches == 0
    assert result.maximum_draft_prefill_batch == 0
    assert result.draft_refresh_batches == 0
    assert result.maximum_draft_refresh_batch == 0
    assert engine.prefill_request_ids == [["a", "b"], ["c"]]


def test_split_draft_predictions_round_trip_in_request_order():
    prediction = StepPredictions(
        kv={("a", 0): {1: 0.75}, ("b", 0): {2: 0.5}},
        experts={0: torch.tensor([[0.1, 0.9], [0.8, 0.2]])},
        expert_scores={0: torch.tensor([[-2.0, 2.0], [1.0, -1.0]])},
    )
    split = _split_predictions([prediction], ["a", "b"])
    executions = {
        request_id: RequestExecution(None, torch.tensor(0), horizons)  # type: ignore[arg-type]
        for request_id, horizons in split.items()
    }
    merged = merge_predictions(["b", "a"], executions, 0)

    assert merged.kv == prediction.kv
    assert torch.equal(merged.experts[0], prediction.experts[0].flip(0))
    assert torch.equal(merged.expert_scores[0], prediction.expert_scores[0].flip(0))


def test_continuous_runner_batches_compatible_draft_refreshes():
    torch.manual_seed(22)
    engine, worker = build_engine(tiny_model())
    draft = tiny_draft()
    draft_forwards = []
    hook = draft.register_forward_hook(lambda *_: draft_forwards.append(1))
    runner = ContinuousBatchRunner(
        engine,
        lambda: DraftSignalProvider(draft, None, lookahead=2),
        max_batch_size=2,
        prefetch=False,
    )
    result = runner.run(
        [
            ServingRequest("a", [1, 2, 3], 4),
            ServingRequest("b", [4, 5, 6], 4),
        ]
    )

    assert {key: len(value) for key, value in result.generated_token_ids.items()} == {
        "a": 4,
        "b": 4,
    }
    # 首轮：prefix + 2 lookahead；刷新轮：advance + 2 lookahead。
    assert len(draft_forwards) == 6
    assert result.draft_refresh_batches == 1
    assert result.maximum_draft_refresh_batch == 2
    hook.remove()
    worker.close()
