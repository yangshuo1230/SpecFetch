from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.runtime.predictor import (
    DraftSignalProvider,
    ExpertProbeBank,
    ProbeEntry,
    aggregate_old_chunk_mass,
)
from src.runtime.qwen3_engine import BatchState


class StubTargetCache:
    def __init__(self, kv_storage: str, old_ranges=None):
        self.config = SimpleNamespace(kv_storage=kv_storage)
        self.old_ranges = old_ranges or {}


def test_attention_aggregation_uses_target_chunk_ranges():
    attention = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.2, 0.6]])
    result = aggregate_old_chunk_mass(attention, {3: (0, 2), 7: (2, 4)})
    assert result[3] == torch.tensor(0.25)
    assert result[7] == torch.tensor(0.75)


def test_probe_bank_round_trip(tmp_path):
    probe = {
        "x_mean": torch.zeros(1, 2),
        "x_scale": torch.ones(1, 2),
        "y_mean": torch.zeros(1, 2),
        "coef": torch.eye(2),
    }
    bank = ExpertProbeBank({0: ProbeEntry(0, probe)})
    path = tmp_path / "probes.pt"
    bank.save(path)
    loaded = ExpertProbeBank.load(path)
    hidden = (torch.zeros(1, 1, 2), torch.tensor([[[2.0, -2.0]]]))
    prediction = loaded.predict(0, hidden)
    cached_prediction = loaded.predict_features(0, hidden[1][:, -1].float())
    assert prediction[0, 0] > prediction[0, 1]
    assert torch.equal(prediction, cached_prediction)


def test_probe_bank_batched_layers_match_independent_predictions():
    first = {
        "x_mean": torch.tensor([[1.0, 0.0]]),
        "x_scale": torch.tensor([[2.0, 1.0]]),
        "y_mean": torch.tensor([[0.1, -0.2]]),
        "coef": torch.tensor([[1.0, 0.0], [0.5, -1.0]]),
    }
    second = {
        "x_mean": torch.tensor([[0.0, -1.0]]),
        "x_scale": torch.tensor([[1.0, 0.5]]),
        "y_mean": torch.tensor([[-0.3, 0.4]]),
        "coef": torch.tensor([[-0.5, 1.0], [1.0, 0.25]]),
    }
    bank = ExpertProbeBank({0: ProbeEntry(0, first), 1: ProbeEntry(1, second)})
    features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.0], [2.0, 1.0], [4.0, 3.0]],
        ]
    )

    actual = bank.predict_feature_batch([0, 1], features)
    expected = torch.stack(
        [bank.predict_features(0, features[0]), bank.predict_features(1, features[1])]
    )

    assert torch.allclose(actual, expected)
    assert len(bank._parameter_batches) == 1


def test_draft_rollout_restores_prefix_cache():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    provider = DraftSignalProvider(model, probe_bank=None, lookahead=2)
    prefix = torch.tensor([[1, 2, 3]])
    provider.initialize(prefix, ["r0"])
    state = BatchState(["r0"], [3], {})
    plan = provider.predict(state)
    assert plan.token_ids.shape == (1, 2)
    assert plan.token_ids.device.type == "cpu"
    assert len(plan.horizons) == 2
    assert provider.cache.get_seq_length() == 3
    provider.advance(torch.tensor([4]))
    assert provider.cache.get_seq_length() == 4
    provider.advance(torch.tensor([[5, 6]]))
    assert provider.cache.get_seq_length() == 6


def test_resident_target_skips_draft_attentions_but_keeps_expert_features():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    rollout_options = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        if "output_attentions" in kwargs:
            rollout_options.append((kwargs["output_attentions"], kwargs["output_hidden_states"]))
        return original_forward(*args, **kwargs)

    model.forward = recording_forward
    probe = {
        "x_mean": torch.zeros(1, config.hidden_size),
        "x_scale": torch.ones(1, config.hidden_size),
        "y_mean": torch.zeros(1, 2),
        "coef": torch.zeros(config.hidden_size, 2),
    }
    probe_bank = ExpertProbeBank({0: ProbeEntry(0, probe)})
    batched_predict = probe_bank.predict_feature_batch
    probe_bank.predict_feature_batch = Mock(wraps=batched_predict)
    provider = DraftSignalProvider(model, probe_bank, lookahead=2)
    provider.initialize(torch.tensor([[1, 2, 3]]), ["r0"])
    state = BatchState(
        ["r0"],
        [3],
        {("r0", 0): StubTargetCache("resident")},
    )

    plan = provider.predict(state)

    assert rollout_options == [(False, True), (False, True)]
    assert all(not prediction.kv for prediction in plan.horizons)
    assert all(0 in prediction.experts for prediction in plan.horizons)
    assert probe_bank.predict_feature_batch.call_count == 1
    batched_features = probe_bank.predict_feature_batch.call_args.args[1]
    assert batched_features.shape == (1, 2, config.hidden_size)
    expected = batched_predict([0], batched_features)[0]
    for prediction, values in zip(plan.horizons, expected):
        assert torch.equal(prediction.experts[0], values[None])


def test_sparse_target_still_requests_and_aggregates_draft_attention():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    rollout_options = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        if "output_attentions" in kwargs:
            rollout_options.append((kwargs["output_attentions"], kwargs["output_hidden_states"]))
        return original_forward(*args, **kwargs)

    model.forward = recording_forward
    provider = DraftSignalProvider(model, probe_bank=None, lookahead=1)
    provider.initialize(torch.tensor([[1, 2, 3]]), ["r0"])
    state = BatchState(
        ["r0"],
        [3],
        {("r0", 0): StubTargetCache("sparse", {0: (0, 2), 1: (2, 4)})},
    )

    plan = provider.predict(state)

    assert rollout_options == [(True, False)]
    assert set(plan.horizons[0].kv[("r0", 0)]) == {0, 1}


def test_batched_draft_prefix_splits_into_independent_request_caches():
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    provider = DraftSignalProvider(model, probe_bank=None, lookahead=2)
    provider.initialize(torch.tensor([[1, 2, 3], [4, 5, 6]]), ["a", "b"])
    expected_logits = provider.next_logits.clone()

    first, second = provider.split_requests()
    assert first.request_ids == ["a"]
    assert second.request_ids == ["b"]
    assert torch.equal(first.next_logits, expected_logits[:1])
    assert torch.equal(second.next_logits, expected_logits[1:])
    assert first.cache.get_seq_length() == second.cache.get_seq_length() == 3

    first.advance(torch.tensor([7]))
    assert first.cache.get_seq_length() == 4
    assert second.cache.get_seq_length() == 3
    plan = second.predict(BatchState(["b"], [3], {}))
    assert plan.token_ids.shape == (1, 2)
    assert second.cache.get_seq_length() == 3

    with pytest.raises(ValueError, match="缓存长度不兼容"):
        DraftSignalProvider.merge_requests([first, second])
    second.advance(torch.tensor([8]))
    merged = DraftSignalProvider.merge_requests([first, second])
    assert merged.request_ids == ["a", "b"]
    assert merged.next_logits.shape == (2, config.vocab_size)
    assert merged.cache.get_seq_length() == 4
