from __future__ import annotations

import argparse
import gc
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    ModelTrace,
    PairedTrace,
    attention_block_mass,
    map_layer,
    router_probabilities,
)

DEFAULT_PROMPTS = [
    "<|im_start|>user\nExplain speculative decoding in two sentences.\n<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhy do mixture-of-experts models save compute?\n<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a Python function that checks whether a number is prime.\n<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nTranslate 'memory bandwidth is limited' into Chinese.\n<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is the difference between RAM and a CPU cache?\n<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nGive three practical uses of matrix multiplication.\n<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nContinue the sequence: 1, 1, 2, 3, 5, and explain the rule.\n<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nSummarize why reproducible experiments need held-out data.\n<|im_end|>\n<|im_start|>assistant\n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/experiment.json"))
    parser.add_argument("--prompts", help="JSON file, array, or object containing prompts")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--kv-block-size", type=int, default=8)
    parser.add_argument("--max-kv-rank", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser.parse_args()


def load_prompts(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_PROMPTS
    path = Path(value)
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(value)
    prompts = data.get("prompts") if isinstance(data, dict) else data
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a JSON array of strings")
    if len(prompts) < 2:
        raise ValueError("at least two prompts are required for a train/evaluation split")
    return prompts


def model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def load_model(name: str, args: argparse.Namespace):
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    return AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()


def generate_sequence(
    model, tokenizer, prompt: str, max_new_tokens: int
) -> tuple[torch.Tensor, int]:
    encoded = tokenizer(prompt, return_tensors="pt")
    prompt_tokens = encoded.input_ids.shape[1]
    encoded = {key: value.to(model_input_device(model)) for key, value in encoded.items()}
    sequence = (
        model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )[0]
        .detach()
        .cpu()
    )
    return sequence, prompt_tokens


def trace_model(
    model,
    input_ids: torch.Tensor,
    prompt_tokens: int,
    block_size: int,
    capture_hidden_states: bool,
) -> ModelTrace:
    sequence_length = len(input_ids)
    token_slice = slice(prompt_tokens, sequence_length)
    router_logits: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(model.model.layers):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is not None:

            def capture(_module, _inputs, output, index=layer_index):
                router_logits[index] = output[0] if isinstance(output, tuple) else output

            handles.append(gate.register_forward_hook(capture))
    ids = input_ids.unsqueeze(0).to(model_input_device(model))
    try:
        with torch.inference_mode():
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                output_attentions=True,
                output_hidden_states=capture_hidden_states,
                return_dict=True,
            )
        if output.attentions is None:
            raise RuntimeError("model returned no attentions; eager attention is required")
        kv_mass = attention_block_mass(output.attentions, block_size, token_slice)
        hidden = {}
        if capture_hidden_states:
            hidden = {
                layer: state[0, token_slice].detach().float().cpu()
                for layer, state in enumerate(output.hidden_states[1:])
            }
        routers = {
            layer: router_probabilities(logits, sequence_length)[token_slice]
            for layer, logits in router_logits.items()
        }
        return ModelTrace(kv_mass, hidden, routers)
    finally:
        for handle in handles:
            handle.remove()
        del ids


def release_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean_rows(rows: dict[str, list[float]]) -> dict[str, float]:
    return {key: sum(values) / len(values) for key, values in rows.items() if values}


def evaluate_kv(traces: list[PairedTrace], max_rank: int, seed: int) -> dict[str, Any]:
    rows: dict[str, list[float]] = defaultdict(list)
    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rng = random.Random(seed)
    observations = 0
    for trace in traces:
        target_layers = len(trace.target.kv_mass)
        draft_layers = len(trace.draft.kv_mass)
        for target_layer, target_queries in trace.target.kv_mass.items():
            draft_layer = map_layer(target_layer, target_layers, draft_layers)
            for target, draft in zip(target_queries, trace.draft.kv_mass[draft_layer]):
                k = min(max_rank, len(target), len(draft))
                if not k:
                    continue
                actual = top_k(target, k)
                predicted = top_k(draft, k)
                candidates = list(target)
                random_prediction = rng.sample(candidates, k)
                recency_prediction = sorted(candidates, reverse=True)[:k]
                values = {
                    "draft_overlap@k": overlap_at_k(predicted, actual, k),
                    "draft_recall@k": recall_at_k(predicted, actual, k),
                    "draft_target_mass@k": sum(target[block] for block in predicted),
                    "draft_weighted_jaccard": weighted_jaccard(draft, target),
                    "draft_ndcg@k": ndcg_at_k(predicted, target, k),
                    "random_recall@k": recall_at_k(random_prediction, actual, k),
                    "random_target_mass@k": sum(target[block] for block in random_prediction),
                    "recency_recall@k": recall_at_k(recency_prediction, actual, k),
                    "recency_target_mass@k": sum(target[block] for block in recency_prediction),
                }
                for name, value in values.items():
                    rows[name].append(value)
                    per_layer[target_layer][name].append(value)
                observations += 1
    return {
        "summary": mean_rows(rows),
        "per_target_layer": {str(layer): mean_rows(values) for layer, values in per_layer.items()},
        "observations": observations,
    }


def routed_labels(probabilities: torch.Tensor, top_experts: int) -> torch.Tensor:
    labels = torch.zeros_like(probabilities)
    indices = probabilities.topk(min(top_experts, probabilities.shape[1]), dim=1).indices
    return labels.scatter(1, indices, 1.0)


def evaluate_experts(
    train: list[PairedTrace],
    evaluation: list[PairedTrace],
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    if not train or not train[0].target.router_probabilities:
        return {"status": "not_applicable", "reason": "target model has no captured MoE router"}
    rows: dict[str, list[float]] = defaultdict(list)
    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rng = random.Random(seed)
    target_layers = len(train[0].target.router_probabilities)
    draft_layers = len(train[0].draft.hidden_states)
    observations = 0
    for target_layer in sorted(train[0].target.router_probabilities):
        draft_layer = map_layer(target_layer, target_layers, draft_layers)
        train_x = torch.cat([trace.draft.hidden_states[draft_layer] for trace in train])
        train_probabilities = torch.cat(
            [trace.target.router_probabilities[target_layer] for trace in train]
        )
        top_experts = min(8, train_probabilities.shape[1])
        labels = routed_labels(train_probabilities, top_experts)
        probe = fit_ridge_probe(train_x, labels, alpha)
        frequency_prediction = labels.mean(0).topk(top_experts).indices.tolist()
        for trace in evaluation:
            actual_probabilities = trace.target.router_probabilities[target_layer]
            predictions = predict_probe(probe, trace.draft.hidden_states[draft_layer])
            for predicted_scores, actual_scores in zip(predictions, actual_probabilities):
                actual_mass = dict(enumerate(actual_scores.tolist()))
                actual = actual_scores.topk(top_experts).indices.tolist()
                predicted = predicted_scores.topk(top_experts).indices.tolist()
                random_prediction = rng.sample(range(len(actual_scores)), top_experts)
                values = {
                    "probe_overlap@8": overlap_at_k(predicted, actual, top_experts),
                    "probe_recall@8": recall_at_k(predicted, actual, top_experts),
                    "probe_ndcg@8": ndcg_at_k(predicted, actual_mass, top_experts),
                    "frequency_recall@8": recall_at_k(frequency_prediction, actual, top_experts),
                    "random_recall@8": recall_at_k(random_prediction, actual, top_experts),
                }
                for name, value in values.items():
                    rows[name].append(value)
                    per_layer[target_layer][name].append(value)
                observations += 1
    return {
        "status": "ok",
        "summary": mean_rows(rows),
        "per_target_layer": {str(layer): mean_rows(values) for layer, values in per_layer.items()},
        "observations": observations,
        "probe": "multi-output ridge on aligned draft hidden states",
    }


def validate_tokenizers(target_tokenizer, draft_name: str, prompts: list[str]) -> None:
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_name, trust_remote_code=True)
    for index, prompt in enumerate(prompts):
        target_ids = target_tokenizer(prompt, add_special_tokens=False).input_ids
        draft_ids = draft_tokenizer(prompt, add_special_tokens=False).input_ids
        if target_ids != draft_ids:
            raise ValueError(f"target and draft tokenizers differ for prompt {index}")


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between zero and one")
    torch.manual_seed(args.seed)
    prompts = list(load_prompts(args.prompts))
    random.Random(args.seed).shuffle(prompts)
    split = max(1, min(len(prompts) - 1, round(len(prompts) * args.train_fraction)))
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    validate_tokenizers(tokenizer, args.draft, prompts)

    target = load_model(args.target, args)
    sequences: list[tuple[torch.Tensor, int]] = []
    target_traces: list[ModelTrace] = []
    for index, prompt in enumerate(prompts, 1):
        print(f"[target {index}/{len(prompts)}] generate and trace", flush=True)
        sequence, prompt_tokens = generate_sequence(target, tokenizer, prompt, args.max_new_tokens)
        sequences.append((sequence, prompt_tokens))
        target_traces.append(
            trace_model(target, sequence, prompt_tokens, args.kv_block_size, False)
        )
    del target
    release_accelerator_cache()

    draft = load_model(args.draft, args)
    paired: list[PairedTrace] = []
    for index, ((sequence, prompt_tokens), target_trace) in enumerate(
        zip(sequences, target_traces), 1
    ):
        print(f"[draft {index}/{len(prompts)}] trace aligned sequence", flush=True)
        draft_trace = trace_model(draft, sequence, prompt_tokens, args.kv_block_size, True)
        paired.append(
            PairedTrace(target_trace, draft_trace, prompt_tokens, len(sequence) - prompt_tokens)
        )
    del draft
    release_accelerator_cache()

    train, evaluation = paired[:split], paired[split:]
    result = {
        "models": {"target": args.target, "draft": args.draft},
        "configuration": {
            "seed": args.seed,
            "prompts": len(prompts),
            "train_prompts": len(train),
            "evaluation_prompts": len(evaluation),
            "train_tokens": sum(trace.evaluated_tokens for trace in train),
            "evaluation_tokens": sum(trace.evaluated_tokens for trace in evaluation),
            "max_new_tokens": args.max_new_tokens,
            "kv_block_size": args.kv_block_size,
            "max_kv_rank": args.max_kv_rank,
            "ridge_alpha": args.ridge_alpha,
            "dtype": args.dtype,
            "layer_alignment": "nearest relative depth",
            "split": "seeded prompt-level shuffle, then train/evaluation",
        },
        "kv_prefetch": evaluate_kv(evaluation, args.max_kv_rank, args.seed),
        "expert_prefetch": evaluate_experts(train, evaluation, args.ridge_alpha, args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
