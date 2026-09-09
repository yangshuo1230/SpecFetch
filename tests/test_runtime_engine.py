from scripts.run_runtime_engine import draft_signals_required


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
