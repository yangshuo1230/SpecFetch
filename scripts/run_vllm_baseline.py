from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--gpu-memory-limit-gib",
        type=float,
        help="Strict post-initialization GPU reserved-memory cap for matched runs",
    )
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=65,
        help="Total generated tokens, including the untimed first token (default: 65)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="vLLM memory fraction; defaults to 0.9 or is derived from --gpu-memory-limit-gib",
    )
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--kv-offloading-size", type=float)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


@dataclass
class DecodeOnlyRun:
    outputs: list[Any]
    prefill_seconds: float
    decode_wall_seconds: float


def run_decode_only(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> DecodeOnlyRun:
    """Run the offline engine and time only work after every first token.

    vLLM's public offline ``generate`` API returns only after the whole batch.
    Stepping its engine is necessary to observe the shared first-token boundary.
    This adapter intentionally fails if one engine step emits multiple initial
    tokens because that would make the requested timing boundary unobservable.
    """
    request_ids = [llm._add_request(prompt, sampling) for prompt in prompts]
    latest: dict[str, Any] = {}
    started_at = clock()
    decode_started_at: float | None = None

    while llm.llm_engine.has_unfinished_requests():
        for output in llm.llm_engine.step():
            latest[output.request_id] = output
        if decode_started_at is None and all(request_id in latest for request_id in request_ids):
            initial_lengths = [
                len(latest[request_id].outputs[0].token_ids) for request_id in request_ids
            ]
            if all(length >= 1 for length in initial_lengths):
                if any(length != 1 for length in initial_lengths):
                    raise RuntimeError(
                        "vLLM emitted multiple tokens before the first-token timing boundary"
                    )
                decode_started_at = clock()

    finished_at = clock()
    if decode_started_at is None:
        raise RuntimeError("vLLM completed without exposing a first token for every request")
    outputs = [latest[request_id] for request_id in request_ids]
    if not all(output.finished for output in outputs):
        raise RuntimeError("vLLM stopped with unfinished request outputs")
    return DecodeOnlyRun(
        outputs=outputs,
        prefill_seconds=decode_started_at - started_at,
        decode_wall_seconds=finished_at - decode_started_at,
    )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 2:
        raise ValueError("max-new-tokens must be at least 2 for decode-only timing")
    if args.gpu_memory_limit_gib is not None and (
        not math.isfinite(args.gpu_memory_limit_gib) or args.gpu_memory_limit_gib <= 0
    ):
        raise ValueError("gpu-memory-limit-gib must be positive")
    require_idle_gpus(1000, 10, {args.physical_gpu_index})
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = fixed_contexts(
        tokenizer,
        load_prompt_texts(args.prompts),
        args.batch_size,
        args.context_tokens,
    )
    cuda_device = torch.device("cuda:0")
    total_gpu_gib = torch.cuda.get_device_properties(cuda_device).total_memory / 2**30
    if args.gpu_memory_limit_gib is not None:
        if args.gpu_memory_limit_gib > total_gpu_gib:
            raise ValueError(
                f"gpu-memory-limit-gib ({args.gpu_memory_limit_gib}) exceeds device capacity "
                f"({total_gpu_gib:.3f} GiB)"
            )
        derived_utilization = args.gpu_memory_limit_gib / total_gpu_gib
        if (
            args.gpu_memory_utilization is not None
            and not abs(args.gpu_memory_utilization - derived_utilization) < 1e-6
        ):
            raise ValueError(
                "gpu-memory-utilization conflicts with the fraction implied by gpu-memory-limit-gib"
            )
        gpu_memory_utilization = derived_utilization
    else:
        gpu_memory_utilization = (
            0.9 if args.gpu_memory_utilization is None else args.gpu_memory_utilization
        )
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    engine_options = {
        "model": args.model,
        "dtype": "bfloat16",
        "gpu_memory_utilization": gpu_memory_utilization,
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
    torch.cuda.reset_peak_memory_stats(cuda_device)
    prompts = [{"prompt_token_ids": row.tolist()} for row in input_ids]
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    run = run_decode_only(llm, prompts, sampling)
    torch.cuda.synchronize(cuda_device)
    peak_gpu_allocated_gib = torch.cuda.max_memory_allocated(cuda_device) / 2**30
    peak_gpu_reserved_gib = torch.cuda.max_memory_reserved(cuda_device) / 2**30
    memory_limit_satisfied = (
        args.gpu_memory_limit_gib is None or peak_gpu_reserved_gib <= args.gpu_memory_limit_gib
    )
    outputs = run.outputs
    generated = [list(item.outputs[0].token_ids) for item in outputs]
    lengths = [len(tokens) for tokens in generated]
    if any(length != args.max_new_tokens for length in lengths):
        raise RuntimeError(
            f"vLLM returned generated lengths {lengths}, expected {args.max_new_tokens}"
        )
    decode_tokens_per_request = args.max_new_tokens - 1
    decode_tokens = args.batch_size * decode_tokens_per_request
    result = {
        "configuration": {**vars(args), "prompts": str(args.prompts), "output": str(args.output)},
        "timing_protocol": "batch_decode_after_all_first_tokens_v1",
        "engine_options": engine_options,
        "performance": {
            "initialization_seconds": initialization_seconds,
            "prefill_seconds": run.prefill_seconds,
            "decode_wall_seconds": run.decode_wall_seconds,
            "decode_tokens_per_request": decode_tokens_per_request,
            "decode_tokens": decode_tokens,
            "request_seconds": run.prefill_seconds + run.decode_wall_seconds,
            "mean_tpot_ms": run.decode_wall_seconds / decode_tokens_per_request * 1000,
            "decode_throughput_tokens_per_second": decode_tokens / run.decode_wall_seconds,
        },
        "gpu_memory": {
            "measurement_protocol": "post_initialization_peak_reserved_v1",
            "limit_gib": args.gpu_memory_limit_gib,
            "total_device_gib": total_gpu_gib,
            "peak_allocated_gib": peak_gpu_allocated_gib,
            "peak_reserved_gib": peak_gpu_reserved_gib,
            "limit_satisfied": memory_limit_satisfied,
        },
        "generated_token_ids": generated,
        "generated_text": [tokenizer.decode(tokens) for tokens in generated],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not memory_limit_satisfied:
        raise RuntimeError(
            f"peak reserved GPU memory {peak_gpu_reserved_gib:.3f} GiB exceeds "
            f"the {args.gpu_memory_limit_gib:.3f} GiB limit"
        )


if __name__ == "__main__":
    main()
