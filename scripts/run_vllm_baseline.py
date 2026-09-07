from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from src.gpu_guard import require_idle_gpus
from src.workload import fixed_contexts, load_prompt_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible vLLM production baseline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--kv-offloading-size", type=float)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_idle_gpus(1000, 10, {args.physical_gpu_index})
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = fixed_contexts(
        tokenizer,
        load_prompt_texts(args.prompts),
        args.batch_size,
        args.context_tokens,
    )
    engine_options = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": True,
    }
    if args.kv_offloading_size is not None:
        engine_options.update(
            kv_offloading_size=args.kv_offloading_size,
            kv_offloading_backend="native",
        )
    initialization_start = time.perf_counter()
    llm = LLM(**engine_options)
    initialization_seconds = time.perf_counter() - initialization_start
    prompts = [{"prompt_token_ids": row.tolist()} for row in input_ids]
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.perf_counter() - start
    generated = [list(item.outputs[0].token_ids) for item in outputs]
    result = {
        "configuration": {**vars(args), "prompts": str(args.prompts), "output": str(args.output)},
        "engine_options": engine_options,
        "performance": {
            "initialization_seconds": initialization_seconds,
            "request_seconds": elapsed,
            "throughput_tokens_per_second": sum(map(len, generated)) / elapsed,
            "mean_tpot_ms_including_prefill": elapsed / sum(map(len, generated)) * 1000,
        },
        "generated_token_ids": generated,
        "generated_text": [tokenizer.decode(tokens) for tokens in generated],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
