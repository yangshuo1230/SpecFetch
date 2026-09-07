import torch

from scripts.run_experiment import evaluate_experts, evaluate_kv
from src.trace import ModelTrace, PairedTrace


def model_trace(kv_mass, hidden_states=None, routers=None):
    return ModelTrace(kv_mass, hidden_states or {}, routers or {})


def test_evaluate_kv_uses_aligned_query_distributions():
    target = model_trace(
        {
            0: [{0: 0.8, 1: 0.2}],
            1: [{0: 0.1, 1: 0.9}],
        }
    )
    draft = model_trace({0: [{0: 0.8, 1: 0.2}]})
    trace = PairedTrace(target, draft, prompt_tokens=4, evaluated_tokens=1)
    result = evaluate_kv([trace], max_rank=1, seed=0)
    assert result["observations"] == 2
    assert result["summary"]["draft_recall@k"] == 0.5


def test_evaluate_experts_runs_probe_only_on_evaluation_trace():
    experts = 16
    train_router = torch.zeros(2, experts)
    train_router[0, :8] = 1
    train_router[1, 8:] = 1
    evaluation_router = train_router.clone()
    hidden = torch.tensor([[-1.0], [1.0]])
    train = PairedTrace(
        model_trace({}, routers={0: train_router}),
        model_trace({}, hidden_states={0: hidden}),
        prompt_tokens=4,
        evaluated_tokens=2,
    )
    evaluation = PairedTrace(
        model_trace({}, routers={0: evaluation_router}),
        model_trace({}, hidden_states={0: hidden}),
        prompt_tokens=4,
        evaluated_tokens=2,
    )
    result = evaluate_experts([train], [evaluation], alpha=0.01, seed=0)
    assert result["status"] == "ok"
    assert result["observations"] == 2
    assert result["summary"]["probe_recall@8"] == 1.0
