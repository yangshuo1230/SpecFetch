from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gpu_guard import require_idle_gpus
from src.metrics import fit_ridge_probe, ndcg_at_k, predict_probe, recall_at_k, top_k
from src.trace import Mass, ModelTrace, attention_block_mass, map_layer, router_probabilities


@dataclass
class TargetTrace:
    sequence: torch.Tensor
    prompt_tokens: int
    kv_by_horizon: dict[int, dict[int, list[Mass]]]
    routers: dict[int, torch.Tensor]

    @property
    def generated_tokens(self) -> int:
        return len(self.sequence) - self.prompt_tokens


@dataclass
class DraftRollout:
    token_ids: torch.Tensor
    trace: ModelTrace


@dataclass
class PromptTrace:
    target: TargetTrace
    rollouts: list[DraftRollout]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate independent draft rollouts as CPU-offload prefetch signals."
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/offload-prefetch.json"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--kv-block-size", type=int, default=8)
    parser.add_argument("--kv-prefetch-blocks", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--require-idle-gpus", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-existing-memory-mib", type=int, default=1000)
    parser.add_argument("--max-existing-utilization", type=int, default=10)
    return parser.parse_args()


def load_prompts(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") if isinstance(data, dict) else data
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("prompts must be a JSON array of strings")
    if len(prompts) < 2:
        raise ValueError("at least two prompts are required")
    return prompts


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_model(name: str, args: argparse.Namespace):
    return AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager",
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()


def generate(model, input_ids: torch.Tensor, new_tokens: int, eos_token_id: int) -> torch.Tensor:
    ids = input_ids.unsqueeze(0).to(input_device(model))
    with torch.inference_mode():
        sequence = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=eos_token_id,
        )[0]
    return sequence.detach().cpu()


def capture_forward(model, sequence: torch.Tensor, hidden_states: bool):
    router_logits: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is not None:

            def capture(_module, _inputs, output, index=layer_index):
                router_logits[index] = output[0] if isinstance(output, tuple) else output

            handles.append(gate.register_forward_hook(capture))
    ids = sequence.unsqueeze(0).to(input_device(model))
    try:
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                output_attentions=True,
                output_hidden_states=hidden_states,
                return_dict=True,
            )
        if output.attentions is None:
            raise RuntimeError("eager attention tracing returned no attention tensors")
        routers = {
            layer: router_probabilities(logits, len(sequence))
            for layer, logits in router_logits.items()
        }
        return output, routers
    finally:
        for handle in handles:
            handle.remove()
        del ids


def trace_target(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    lookahead: int,
    block_size: int,
    eos_token_id: int,
) -> TargetTrace:
    sequence = generate(model, prompt_ids, max_new_tokens, eos_token_id)
    prompt_tokens = len(prompt_ids)
    output, routers = capture_forward(model, sequence, hidden_states=False)
    generated = len(sequence) - prompt_tokens
    kv_by_horizon = {}
    for horizon in range(1, min(lookahead, generated) + 1):
        # For target query q, q + 1 - horizon is exactly the prefix length at
        # which an h-token-ahead prefetch request would have been issued.
        token_slice = slice(prompt_tokens + horizon - 1, len(sequence))
        kv_by_horizon[horizon] = attention_block_mass(
            output.attentions, block_size, token_slice, key_lag=horizon
        )
    generated_routers = {layer: values[prompt_tokens:] for layer, values in routers.items()}
    del output
    return TargetTrace(sequence, prompt_tokens, kv_by_horizon, generated_routers)


def trace_draft_rollout(
    model,
    prefix: torch.Tensor,
    lookahead: int,
    block_size: int,
    eos_token_id: int,
) -> DraftRollout:
    sequence = generate(model, prefix, lookahead, eos_token_id)
    prefix_tokens = len(prefix)
    output, routers = capture_forward(model, sequence, hidden_states=True)
    token_slice = slice(prefix_tokens, len(sequence))
    kv_mass = attention_block_mass(
        output.attentions, block_size, token_slice, key_limit=prefix_tokens
    )
    hidden = {
        layer: state[0, token_slice].detach().float().cpu()
        for layer, state in enumerate(output.hidden_states[1:])
    }
    generated_routers = {layer: values[token_slice] for layer, values in routers.items()}
    del output
    return DraftRollout(sequence[prefix_tokens:], ModelTrace(kv_mass, hidden, generated_routers))


def collect_prompt(model, target: TargetTrace, args: argparse.Namespace, eos: int) -> PromptTrace:
    rollouts = []
    for issue in range(target.generated_tokens):
        prefix_end = target.prompt_tokens + issue
        remaining = min(args.lookahead, target.generated_tokens - issue)
        rollouts.append(
            trace_draft_rollout(
                model,
                target.sequence[:prefix_end],
                remaining,
                args.kv_block_size,
                eos,
            )
        )
    return PromptTrace(target, rollouts)


def mean_rows(rows: dict[str, list[float]]) -> dict[str, float]:
    return {name: sum(values) / len(values) for name, values in rows.items() if values}


def add_conditioned(rows: dict[str, list[float]], name: str, value: float, match: bool) -> None:
    rows[name].append(value)
    rows[f"{name}_when_token_{'matches' if match else 'differs'}"].append(value)


def token_matches(prompt: PromptTrace, issue: int, horizon: int) -> tuple[bool, bool]:
    rollout = prompt.rollouts[issue].token_ids
    actual = prompt.target.sequence[
        prompt.target.prompt_tokens + issue : prompt.target.prompt_tokens + issue + horizon
    ]
    if len(rollout) < horizon:
        return False, False
    return bool(rollout[horizon - 1] == actual[horizon - 1]), bool(
        torch.equal(rollout[:horizon], actual)
    )


def evaluate_kv(prompts: list[PromptTrace], budget: int, seed: int) -> dict[str, Any]:
    by_horizon = {}
    rng = random.Random(seed)
    max_horizon = max(len(prompt.target.kv_by_horizon) for prompt in prompts)
    for horizon in range(1, max_horizon + 1):
        rows: dict[str, list[float]] = defaultdict(list)
        observations = 0
        for prompt in prompts:
            target_by_layer = prompt.target.kv_by_horizon.get(horizon, {})
            for issue in range(max(0, prompt.target.generated_tokens - horizon + 1)):
                rollout = prompt.rollouts[issue]
                if len(rollout.token_ids) < horizon:
                    continue
                match, prefix_match = token_matches(prompt, issue, horizon)
                rows["token_match"].append(float(match))
                rows["rollout_prefix_match"].append(float(prefix_match))
                for target_layer, target_rows in target_by_layer.items():
                    draft_layer = map_layer(
                        target_layer, len(target_by_layer), len(rollout.trace.kv_mass)
                    )
                    target_mass = target_rows[issue]
                    draft_mass = rollout.trace.kv_mass[draft_layer][horizon - 1]
                    k = min(budget, len(target_mass), len(draft_mass))
                    if not k:
                        continue
                    actual = top_k(target_mass, k)
                    predicted = top_k(draft_mass, k)
                    candidates = list(target_mass)
                    random_prediction = rng.sample(candidates, k)
                    recency_prediction = sorted(candidates, reverse=True)[:k]
                    add_conditioned(
                        rows,
                        "draft_recall@k",
                        recall_at_k(predicted, actual, k),
                        match,
                    )
                    rows["draft_target_mass@k"].append(
                        sum(target_mass[block] for block in predicted)
                    )
                    rows["draft_ndcg@k"].append(ndcg_at_k(predicted, target_mass, k))
                    rows["random_recall@k"].append(recall_at_k(random_prediction, actual, k))
                    rows["recency_recall@k"].append(recall_at_k(recency_prediction, actual, k))
                    observations += 1
        by_horizon[str(horizon)] = {
            "summary": mean_rows(rows),
            "layer_token_observations": observations,
        }
    return {"by_horizon": by_horizon, "signal": "independent draft attention"}


def routed_labels(probabilities: torch.Tensor, experts: int) -> torch.Tensor:
    labels = torch.zeros_like(probabilities)
    indices = probabilities.topk(min(experts, probabilities.shape[1]), dim=1).indices
    return labels.scatter(1, indices, 1.0)


def probe_examples(
    prompts: list[PromptTrace], target_layer: int, draft_layer: int, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    features, probabilities = [], []
    for prompt in prompts:
        for issue in range(max(0, prompt.target.generated_tokens - horizon + 1)):
            rollout = prompt.rollouts[issue]
            if len(rollout.token_ids) < horizon:
                continue
            features.append(rollout.trace.hidden_states[draft_layer][horizon - 1])
            probabilities.append(prompt.target.routers[target_layer][issue + horizon - 1])
    return torch.stack(features), torch.stack(probabilities)


def evaluate_experts(
    train: list[PromptTrace], evaluation: list[PromptTrace], alpha: float, seed: int
) -> dict[str, Any]:
    if not train[0].target.routers:
        return {"status": "not_applicable", "reason": "target has no MoE router"}
    by_horizon = {}
    rng = random.Random(seed)
    target_layers = len(train[0].target.routers)
    draft_layers = len(train[0].rollouts[0].trace.hidden_states)
    max_horizon = max(len(prompt.target.kv_by_horizon) for prompt in evaluation)
    for horizon in range(1, max_horizon + 1):
        rows: dict[str, list[float]] = defaultdict(list)
        observations = 0
        for target_layer in sorted(train[0].target.routers):
            draft_layer = map_layer(target_layer, target_layers, draft_layers)
            train_x, train_probabilities = probe_examples(train, target_layer, draft_layer, horizon)
            experts = min(8, train_probabilities.shape[1])
            labels = routed_labels(train_probabilities, experts)
            probe = fit_ridge_probe(train_x, labels, alpha)
            frequency = labels.mean(0).topk(experts).indices.tolist()
            for prompt in evaluation:
                for issue in range(max(0, prompt.target.generated_tokens - horizon + 1)):
                    rollout = prompt.rollouts[issue]
                    if len(rollout.token_ids) < horizon:
                        continue
                    feature = rollout.trace.hidden_states[draft_layer][horizon - 1].unsqueeze(0)
                    predicted_scores = predict_probe(probe, feature)[0]
                    actual_scores = prompt.target.routers[target_layer][issue + horizon - 1]
                    actual = actual_scores.topk(experts).indices.tolist()
                    predicted = predicted_scores.topk(experts).indices.tolist()
                    random_prediction = rng.sample(range(len(actual_scores)), experts)
                    match, _ = token_matches(prompt, issue, horizon)
                    add_conditioned(
                        rows,
                        "probe_recall@8",
                        recall_at_k(predicted, actual, experts),
                        match,
                    )
                    rows["probe_ndcg@8"].append(
                        ndcg_at_k(predicted, dict(enumerate(actual_scores.tolist())), experts)
                    )
                    rows["frequency_recall@8"].append(recall_at_k(frequency, actual, experts))
                    rows["random_recall@8"].append(recall_at_k(random_prediction, actual, experts))
                    observations += 1
        by_horizon[str(horizon)] = {
            "summary": mean_rows(rows),
            "layer_token_observations": observations,
        }
    return {
        "status": "ok",
        "by_horizon": by_horizon,
        "signal": "ridge probe on independent draft hidden states",
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.lookahead <= 0 or args.max_new_tokens <= 0:
        raise ValueError("lookahead and max-new-tokens must be positive")
    if not 0 < args.train_fraction < 1:
        raise ValueError("train-fraction must be between zero and one")
    if args.require_idle_gpus:
        require_idle_gpus(args.max_existing_memory_mib, args.max_existing_utilization)


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    prompts = load_prompts(args.prompts)
    random.Random(args.seed).shuffle(prompts)
    split = max(1, min(len(prompts) - 1, round(len(prompts) * args.train_fraction)))
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft, trust_remote_code=True)
    encoded = [tokenizer(prompt, add_special_tokens=False).input_ids for prompt in prompts]
    draft_encoded = [
        draft_tokenizer(prompt, add_special_tokens=False).input_ids for prompt in prompts
    ]
    if encoded != draft_encoded:
        raise ValueError("target and draft tokenizers produce different prompt token IDs")

    target_model = load_model(args.target, args)
    targets = []
    for index, ids in enumerate(encoded, 1):
        print(f"[target {index}/{len(encoded)}] generate ground truth and trace", flush=True)
        targets.append(
            trace_target(
                target_model,
                torch.tensor(ids),
                args.max_new_tokens,
                args.lookahead,
                args.kv_block_size,
                tokenizer.eos_token_id,
            )
        )
    del target_model
    torch.cuda.empty_cache()

    # Check again between the two model phases, while this process owns no model memory.
    if args.require_idle_gpus:
        require_idle_gpus(args.max_existing_memory_mib, args.max_existing_utilization)
    draft_model = load_model(args.draft, args)
    traces = []
    for index, target in enumerate(targets, 1):
        print(f"[draft {index}/{len(targets)}] independent per-step rollouts", flush=True)
        traces.append(collect_prompt(draft_model, target, args, tokenizer.eos_token_id))
    del draft_model
    torch.cuda.empty_cache()

    train, evaluation = traces[:split], traces[split:]
    result = {
        "models": {"target": args.target, "draft": args.draft},
        "semantics": {
            "draft": "independent greedy rollout from the currently known target prefix",
            "target": "normal decoding; never verifies or accepts draft tokens",
            "ground_truth": "target access traces are used only for offline scoring",
            "offload": "KV chunks and experts are treated as CPU-resident prefetch objects",
        },
        "configuration": {
            "dtype": args.dtype,
            "prompts": len(prompts),
            "train_prompts": len(train),
            "evaluation_prompts": len(evaluation),
            "max_new_tokens": args.max_new_tokens,
            "lookahead": args.lookahead,
            "kv_block_size": args.kv_block_size,
            "kv_prefetch_blocks": args.kv_prefetch_blocks,
            "expert_prefetch_count": 8,
            "ridge_alpha": args.ridge_alpha,
            "seed": args.seed,
            "layer_alignment": "nearest relative depth",
        },
        "kv_prefetch": evaluate_kv(evaluation, args.kv_prefetch_blocks, args.seed),
        "expert_prefetch": evaluate_experts(train, evaluation, args.ridge_alpha, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
