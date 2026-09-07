import pytest
import torch

from src.metrics import (
    fit_ridge_probe,
    ndcg_at_k,
    overlap_at_k,
    predict_probe,
    recall_at_k,
    top_k,
    weighted_jaccard,
)
from src.trace import (
    aggregate_mass,
    attention_block_mass,
    map_layer,
    normalize_mass,
    router_probabilities,
)


def test_ranking_metrics():
    values = {0: 0.2, 1: 0.8, 2: 0.5}
    assert top_k(values, 2) == [1, 2]
    assert overlap_at_k([0, 1, 2], [1, 2, 3], 3) == pytest.approx(2 / 3)
    assert recall_at_k([0, 1, 2], [1, 2, 3], 3) == pytest.approx(2 / 3)
    assert ndcg_at_k([1, 2], values, 2) == pytest.approx(1.0)


def test_overlap_caps_k_to_available_items():
    assert overlap_at_k([0], [0], 8) == 1.0


def test_weighted_jaccard():
    left = {0: 0.7, 1: 0.3}
    right = {0: 0.3, 1: 0.7}
    assert weighted_jaccard(left, right) == pytest.approx(0.6 / 1.4)


def test_attention_block_mass_preserves_layers_and_queries():
    attention = torch.zeros(1, 2, 3, 3)
    attention[:, :, 1, :2] = torch.tensor([0.25, 0.75])
    attention[:, :, 2, :3] = torch.tensor([0.1, 0.2, 0.7])
    result = attention_block_mass((attention,), block_size=2, token_slice=slice(1, 3))
    assert result[0][0] == pytest.approx({0: 1.0})
    assert result[0][1] == pytest.approx({0: 0.3, 1: 0.7})


def test_attention_block_mass_limits_keys_available_at_prefetch_time():
    attention = torch.zeros(1, 1, 4, 4)
    attention[0, 0, 3] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    fixed = attention_block_mass((attention,), block_size=1, token_slice=slice(3, 4), key_limit=2)
    lagged = attention_block_mass((attention,), block_size=1, token_slice=slice(3, 4), key_lag=2)
    assert fixed[0][0] == pytest.approx({0: 1 / 3, 1: 2 / 3})
    assert lagged == fixed


def test_attention_block_mass_rejects_two_key_window_modes():
    attention = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        attention_block_mass(
            (attention,), block_size=1, token_slice=slice(0, 1), key_limit=1, key_lag=1
        )


def test_mass_helpers_and_layer_mapping():
    assert aggregate_mass([1, 2, 3, 4], 2) == {0: 3.0, 1: 7.0}
    assert normalize_mass({0: 1.0, 1: 3.0}) == {0: 0.25, 1: 0.75}
    assert map_layer(47, 48, 28) == 27
    assert map_layer(0, 48, 28) == 0


def test_router_probabilities_flattens_batch_and_sequence():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    probabilities = router_probabilities(logits, sequence_length=2)
    assert probabilities.shape == (2, 2)
    assert probabilities.sum(1).tolist() == pytest.approx([1.0, 1.0])


def test_ridge_probe_learns_held_out_linear_pattern():
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    labels = torch.cat([-features, features], dim=1)
    probe = fit_ridge_probe(features, labels, alpha=0.01)
    prediction = predict_probe(probe, torch.tensor([[3.0]]))
    assert prediction.argmax(1).item() == 1
