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
    assert prediction[0, 0] > prediction[0, 1]


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
    assert len(plan.horizons) == 2
    assert provider.cache.get_seq_length() == 3
    provider.advance(torch.tensor([4]))
    assert provider.cache.get_seq_length() == 4
    provider.advance(torch.tensor([[5, 6]]))
    assert provider.cache.get_seq_length() == 6


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
