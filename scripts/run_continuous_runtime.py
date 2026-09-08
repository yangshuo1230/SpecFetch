from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gpu_guard import require_idle_gpus
from src.runtime.batch_scheduler import ServingRequest
from src.runtime.config import RuntimeConfig
from src.runtime.continuous_engine import ContinuousBatchRunner
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKind
from src.runtime.model_loader import load_qwen3_non_expert
from src.runtime.predictor import DraftSignalProvider, ExpertProbeBank
from src.runtime.qwen3_engine import Qwen3SparseOffloadEngine
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import CudaTransferBackend, OffloadRuntime, TransferWorker
from src.workload import fixed_contexts, load_prompt_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous-batch sparse offload inference.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument(
        "--output-lengths",
        default="4,8,12,16",
        help="Comma-separated limits repeated across submitted requests",
    )
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument("--prefetch-horizons", type=int, default=1)
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--recent-tokens", type=int, default=256)
    parser.add_argument("--kv-chunk-tokens", type=int, default=64)
    parser.add_argument("--mass-threshold", type=float, default=0.95)
    parser.add_argument("--marginal-threshold", type=float, default=0.01)
    parser.add_argument("--marginal-patience", type=int, default=2)
    parser.add_argument("--minimum-old-chunks", type=int, default=2)
    parser.add_argument("--expert-cache-slots", type=int, default=64)
    parser.add_argument("--kv-cache-slots", type=int, default=512)
    parser.add_argument("--disable-prefetch", action="store_true")
    parser.add_argument("--no-pin-experts", action="store_true")
    parser.add_argument("--lazy-expert-store", action="store_true")
    return parser.parse_args()


def output_lengths(value: str, request_count: int) -> list[int]:
    pattern = [int(item) for item in value.split(",") if item.strip()]
    if not pattern or any(item <= 0 for item in pattern):
        raise ValueError("output lengths must be positive integers")
    return [pattern[index % len(pattern)] for index in range(request_count)]


def main() -> None:
    args = parse_args()
    require_idle_gpus(1000, 10, {args.physical_gpu_index})
    if args.request_count <= 0:
        raise ValueError("request-count must be positive")
    if args.prefetch_horizons > args.lookahead:
        raise ValueError("prefetch-horizons cannot exceed lookahead")
    config = RuntimeConfig(
        sink_tokens=args.sink_tokens,
        recent_tokens=args.recent_tokens,
        kv_chunk_tokens=args.kv_chunk_tokens,
        predicted_mass_threshold=args.mass_threshold,
        marginal_mass_threshold=args.marginal_threshold,
        marginal_patience=args.marginal_patience,
        minimum_old_chunks=args.minimum_old_chunks,
        expert_cache_slots=args.expert_cache_slots,
        kv_cache_slots=args.kv_cache_slots,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    input_ids = fixed_contexts(
        tokenizer,
        load_prompt_texts(args.prompts),
        args.request_count,
        args.context_tokens,
    )
    lengths = output_lengths(args.output_lengths, args.request_count)

    initialization_start = time.perf_counter()
    target, expert_source = load_qwen3_non_expert(
        args.target,
        device=args.device,
        dtype=torch.bfloat16,
        pin_experts=not args.no_pin_experts,
    )
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device)
    probes = ExpertProbeBank.load(args.probes) if args.probes else None
    queue = MemoryRequestQueue()
    residency = ResidencyManager(
        {
            ResourceKind.EXPERT: config.expert_cache_slots,
            ResourceKind.KV: config.kv_cache_slots,
        }
    )
    backend = CudaTransferBackend(args.device, expert_slots=config.expert_cache_slots)
    worker = TransferWorker(queue, residency, backend)
    runtime = OffloadRuntime(queue, residency, worker)
    engine = Qwen3SparseOffloadEngine(target, expert_source, runtime, residency, config)
    if not args.lazy_expert_store:
        engine.expert_registry.preload(
            range(len(target.model.layers)), range(target.config.num_experts)
        )
    initialization_seconds = time.perf_counter() - initialization_start
    requests = [
        ServingRequest(f"request-{index}", row.tolist(), lengths[index])
        for index, row in enumerate(input_ids)
    ]
    runner = ContinuousBatchRunner(
        engine,
        lambda: DraftSignalProvider(draft, probes, args.lookahead),
        max_batch_size=args.max_batch_size,
        prefetch_horizons=args.prefetch_horizons,
        prefetch=not args.disable_prefetch,
    )

    worker.start()
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    start = time.perf_counter()
    try:
        run_result = runner.run(requests)
        torch.cuda.synchronize(torch.device(args.device))
        request_seconds = time.perf_counter() - start
    finally:
        worker.close()

    generated = run_result.generated_token_ids
    token_count = sum(map(len, generated.values()))
    result = {
        "configuration": {
            **vars(args),
            "prompts": str(args.prompts),
            "output": str(args.output),
            "probes": str(args.probes) if args.probes else None,
            "resolved_output_lengths": lengths,
            "policy": "demand_only" if args.disable_prefetch else "speculative",
        },
        "performance": {
            "initialization_seconds": initialization_seconds,
            "request_seconds": request_seconds,
            "throughput_tokens_per_second": token_count / request_seconds,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**30,
            "decode_cycles": run_result.decode_cycles,
            "admission_events": run_result.admission_events,
            "maximum_active_requests": run_result.maximum_active_requests,
        },
        "transfer": vars(worker.metrics_snapshot()),
        "residency": {
            "evictions": residency.evictions,
            "wasted_prefetches": residency.wasted_prefetches,
        },
        "generated_token_ids": generated,
        "generated_text": {
            request_id: tokenizer.decode(tokens) for request_id, tokens in generated.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
