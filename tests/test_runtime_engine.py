import pytest

from scripts.run_runtime_engine import draft_signals_required, maximize_expert_cache_slots


def test_sparse_decode_keeps_draft_kv_ranking_without_prefetch():
    assert draft_signals_required(
        "sparse",
        prefetch_enabled=False,
        expert_probes_available=False,
    )


def test_resident_decode_only_runs_draft_for_expert_prefetch():
    assert draft_signals_required(
        "resident",
        prefetch_enabled=True,
        expert_probes_available=True,
    )
    assert not draft_signals_required(
        "resident",
        prefetch_enabled=False,
        expert_probes_available=True,
    )
    assert not draft_signals_required(
        "resident",
        prefetch_enabled=True,
        expert_probes_available=False,
    )


def test_expert_cache_maximizer_fills_only_persistent_headroom():
    slots = maximize_expert_cache_slots(
        memory_limit_gib=10.0,
        base_allocated_gib=3.0,
        resident_kv_bytes=2 * 2**30,
        expert_slot_bytes=8 * 2**20,
        workspace_reserve_gib=1.0,
        maximum_slots=10_000,
    )
    assert slots == 512
    assert (
        maximize_expert_cache_slots(
            memory_limit_gib=10.0,
            base_allocated_gib=3.0,
            resident_kv_bytes=0,
            expert_slot_bytes=8 * 2**20,
            workspace_reserve_gib=1.0,
            maximum_slots=100,
        )
        == 100
    )
    with pytest.raises(ValueError, match="no expert cache slot"):
        maximize_expert_cache_slots(
            memory_limit_gib=4.0,
            base_allocated_gib=3.0,
            resident_kv_bytes=0,
            expert_slot_bytes=8 * 2**20,
            workspace_reserve_gib=1.0,
            maximum_slots=100,
        )
