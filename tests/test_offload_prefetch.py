import torch

from scripts.run_offload_prefetch import (
    DraftRollout,
    PromptTrace,
    TargetTrace,
    evaluate_kv,
    token_matches,
)
from src.trace import ModelTrace


def rollout(tokens, masses):
    trace = ModelTrace(kv_mass={0: masses}, hidden_states={}, router_probabilities={})
    return DraftRollout(torch.tensor(tokens), trace)


def test_independent_rollout_scores_each_prefetch_horizon():
    target = TargetTrace(
        sequence=torch.tensor([1, 2, 10, 11, 12]),
        prompt_tokens=2,
        kv_by_horizon={
            1: {0: [{0: 0.9, 1: 0.1}, {0: 0.1, 1: 0.9}, {0: 0.8, 1: 0.2}]},
            2: {0: [{0: 0.2, 1: 0.8}, {0: 0.7, 1: 0.3}]},
        },
        routers={},
    )
    prompt = PromptTrace(
        target,
        [
            rollout([10, 99], [{0: 0.8, 1: 0.2}, {0: 0.1, 1: 0.9}]),
            rollout([11, 12], [{0: 0.2, 1: 0.8}, {0: 0.8, 1: 0.2}]),
            rollout([12], [{0: 0.7, 1: 0.3}]),
        ],
    )

    result = evaluate_kv([prompt], budget=1, seed=0)["by_horizon"]
    assert result["1"]["layer_token_observations"] == 3
    assert result["1"]["summary"]["draft_recall@k"] == 1.0
    assert result["2"]["layer_token_observations"] == 2
    assert result["2"]["summary"]["token_match"] == 0.5
    assert result["2"]["summary"]["draft_recall@k"] == 1.0
    assert result["2"]["summary"]["draft_recall@k_when_token_differs"] == 1.0


def test_token_match_distinguishes_one_token_from_whole_rollout():
    target = TargetTrace(torch.tensor([1, 10, 11]), 1, {}, {})
    prompt = PromptTrace(target, [rollout([99, 11], []), rollout([11], [])])
    token_match, prefix_match = token_matches(prompt, issue=0, horizon=2)
    assert token_match
    assert not prefix_match
