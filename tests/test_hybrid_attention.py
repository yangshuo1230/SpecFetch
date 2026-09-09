import pytest
import torch

from src.runtime.hybrid_attention import (
    HybridStopController,
    attention_output,
    chunk_logsumexp,
    empty_partition,
    marginal_partition_mass,
    mean_target_marginal,
    sequence_target_marginals,
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


def test_sequence_marginals_match_ordered_scalar_updates():
    partition = torch.tensor([0.1, -0.2])
    chunks = [torch.tensor([0.4, 0.3]), torch.tensor([-0.1, 0.8])]
    expected_partition = partition
    expected_marginals = []
    for chunk in chunks:
        expected_marginals.append(mean_target_marginal(expected_partition, chunk))
        expected_partition = update_partition(expected_partition, chunk)

    actual_partition, actual_marginals = sequence_target_marginals(partition, chunks)

    assert actual_partition == pytest.approx(expected_partition)
    assert actual_marginals == pytest.approx(expected_marginals)


def test_grouped_attention_matches_explicit_kv_head_repetition():
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(6, 4, generator=generator)
    key = torch.randn(5, 2, 4, generator=generator)
    value = torch.randn(5, 2, 4, generator=generator)
    repeated_key = key.repeat_interleave(3, dim=1)
    repeated_value = value.repeat_interleave(3, dim=1)
    logits = torch.einsum("hd,thd->ht", query.float(), repeated_key.float()) * 0.5
    weights = torch.softmax(logits, dim=-1)
    expected = torch.einsum("ht,thd->hd", weights, repeated_value)

    actual = attention_output(query, [(key, value)], scale=0.5)

    assert actual == pytest.approx(expected, abs=1e-6)
    assert chunk_logsumexp(query, key, scale=0.5) == pytest.approx(
        torch.logsumexp(logits, dim=-1), abs=1e-6
    )
