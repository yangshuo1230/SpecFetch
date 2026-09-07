import pytest
import torch

from src.runtime.hybrid_attention import (
    HybridStopController,
    chunk_logsumexp,
    empty_partition,
    marginal_partition_mass,
    update_partition,
)


def test_hybrid_stop_requires_both_signals_and_patience():
    controller = HybridStopController(
        predicted_mass_threshold=0.9,
        marginal_mass_threshold=0.05,
        marginal_patience=2,
        minimum_chunks=2,
    )
    assert not controller.observe(0.6, 0.4)
    assert not controller.observe(0.3, 0.04)
    assert controller.observe(0.05, 0.03)


def test_large_marginal_resets_streak():
    controller = HybridStopController(0.5, 0.1, 2, 1)
    assert not controller.observe(0.5, 0.05)
    assert not controller.observe(0.2, 0.2)
    assert not controller.observe(0.2, 0.05)
    assert controller.observe(0.1, 0.05)


def test_gqa_chunk_partition_and_marginal():
    query = torch.ones(4, 2)
    key = torch.ones(3, 2, 2)
    chunk_lse = chunk_logsumexp(query, key, scale=1.0)
    assert chunk_lse.shape == (4,)
    expected = 2.0 + torch.log(torch.tensor(3.0))
    assert chunk_lse == pytest.approx(torch.full((4,), expected))

    partition = empty_partition(4)
    partition = update_partition(partition, chunk_lse)
    second = chunk_lse - torch.log(torch.tensor(3.0))
    marginal = marginal_partition_mass(partition, second)
    assert marginal == pytest.approx(torch.full((4,), 0.25))
