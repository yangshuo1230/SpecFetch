from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gpu_guard import require_idle_gpus
from src.runtime.config import RuntimeConfig
from src.runtime.memory_queue import MemoryRequestQueue, ResourceKind
from src.runtime.model_loader import load_qwen3_non_expert
from src.runtime.predictor import DraftSignalProvider, ExpertProbeBank
from src.runtime.qwen3_engine import Qwen3SparseOffloadEngine
from src.runtime.residency import ResidencyManager
from src.runtime.transfer import CudaTransferBackend, OffloadRuntime, TransferWorker
from src.workload import fixed_contexts, load_prompt_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3 sparse offload reference engine.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument(
        "--gpu-memory-limit-gib",
        type=float,
        help="Strict post-initialization GPU reserved-memory cap recorded for matched runs",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=65,
        help="Total generated tokens, including the untimed first token (default: 65)",
    )
    parser.add_argument("--lookahead", type=int, default=4)
    parser.add_argument(
        "--draft-refresh-tokens",
        type=int,
        default=0,
        help="Target tokens consumed per draft rollout; 0 uses lookahead",
    )
    parser.add_argument(
        "--prefetch-horizons",
        type=int,
        default=1,
        help="Number of currently available horizons admitted to the memory queue",
    )
    parser.add_argument("--sink-tokens", type=int, default=4)
    parser.add_argument("--recent-tokens", type=int, default=256)
    parser.add_argument("--kv-chunk-tokens", type=int, default=64)
    parser.add_argument("--mass-threshold", type=float, default=0.95)
    parser.add_argument("--marginal-threshold", type=float, default=0.01)
    parser.add_argument("--marginal-patience", type=int, default=2)
    parser.add_argument("--minimum-old-chunks", type=int, default=2)
    parser.add_argument("--expert-cache-slots", type=int, default=64)
    parser.add_argument("--kv-cache-slots", type=int, default=512)
    parser.add_argument(
        "--kv-storage",
        choices=("sparse", "resident"),
        default="sparse",
        help="Offloaded sparse KV or preallocated full GPU-resident KV for decode",
    )
    parser.add_argument("--transfer-batch-size", type=int, default=32)
    parser.add_argument(
        "--speculative-expert-budget",
        type=int,
        help="Maximum unique expert candidates admitted by one prediction update",
    )
    parser.add_argument(
        "--speculative-kv-budget",
        type=int,
        help="Maximum unique KV candidates admitted by one prediction update",
    )
    parser.add_argument(
        "--speculative-layer-lookahead",
        type=int,
        help="Queue only this many logical layer deadlines ahead; unset queues all horizons",
    )
    parser.add_argument("--disable-prefetch", action="store_true")
    parser.add_argument(
        "--shadow-attention",
        action="store_true",
        help="Compute a CPU full-attention shadow for quality metrics; invalidates latency",
    )
    parser.add_argument(
        "--shadow-thresholds",
        default="",
        help="Comma-separated counterfactual mass thresholds evaluated by the CPU shadow",
    )
    parser.add_argument("--no-pin-experts", action="store_true")
    parser.add_argument("--lazy-expert-store", action="store_true")
    parser.add_argument("--moe-backend", choices=("auto", "torch", "vllm"), default="torch")
    parser.add_argument("--disable-moe-warmup", action="store_true")
    return parser.parse_args()


def synchronize(device: str) -> None:
    torch.cuda.synchronize(torch.device(device))


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item) for item in value.split(",") if item.strip())
    if any(not 0 < item <= 1 for item in thresholds):
        raise ValueError("shadow thresholds must be in (0, 1]")
    return thresholds


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 2:
        raise ValueError("max-new-tokens must be at least 2 for decode-only timing")
    if args.gpu_memory_limit_gib is not None and (
        not math.isfinite(args.gpu_memory_limit_gib) or args.gpu_memory_limit_gib <= 0
    ):
        raise ValueError("gpu-memory-limit-gib must be positive")
    require_idle_gpus(1000, 10, {args.physical_gpu_index})
    refresh_tokens = args.draft_refresh_tokens or args.lookahead
    if not 0 < refresh_tokens <= args.lookahead:
        raise ValueError("draft-refresh-tokens must be in [1, lookahead]")
    if not 0 < args.prefetch_horizons <= args.lookahead:
        raise ValueError("prefetch-horizons must be in [1, lookahead]")
    shadow_thresholds = parse_thresholds(args.shadow_thresholds)
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
        speculative_expert_budget=args.speculative_expert_budget,
        speculative_kv_budget=args.speculative_kv_budget,
        speculative_layer_lookahead=args.speculative_layer_lookahead,
        kv_storage=args.kv_storage,
        resident_kv_capacity_tokens=(
            args.context_tokens + args.max_new_tokens if args.kv_storage == "resident" else None
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    input_ids = fixed_contexts(
        tokenizer,
        load_prompt_texts(args.prompts),
        args.batch_size,
        args.context_tokens,
    )

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
    worker = TransferWorker(queue, residency, backend, max_batch_size=args.transfer_batch_size)
    runtime = OffloadRuntime(queue, residency, worker)
    engine = Qwen3SparseOffloadEngine(
        target,
        expert_source,
        runtime,
        residency,
        config,
        moe_backend=args.moe_backend,
    )
    provider = DraftSignalProvider(draft, probes, args.lookahead)
    request_ids = [f"request-{index}" for index in range(args.batch_size)]
    expert_preload_start = time.perf_counter()
    if not args.lazy_expert_store:
        engine.expert_registry.preload(
            range(len(target.model.layers)), range(target.config.num_experts)
        )
    expert_preload_seconds = time.perf_counter() - expert_preload_start
    initialization_seconds = time.perf_counter() - initialization_start
    worker.start()
    try:
        moe_warmup_start = time.perf_counter()
        if not (args.disable_moe_warmup or args.lazy_expert_store):
            engine.warmup_moe([args.batch_size * args.context_tokens, args.batch_size])
        moe_warmup_seconds = time.perf_counter() - moe_warmup_start
    except BaseException:
        worker.close()
        raise
    transfer_start, _ = worker.phase_metrics_since(worker.metrics_snapshot())
    residency_start = (residency.evictions, residency.wasted_prefetches)
    cuda_device = torch.device(args.device)
    total_gpu_gib = torch.cuda.get_device_properties(cuda_device).total_memory / 2**30
    if args.gpu_memory_limit_gib is not None and args.gpu_memory_limit_gib > total_gpu_gib:
        worker.close()
        raise ValueError(
            f"gpu-memory-limit-gib ({args.gpu_memory_limit_gib}) exceeds device capacity "
            f"({total_gpu_gib:.3f} GiB)"
        )
    initial_gpu_allocated_gib = torch.cuda.memory_allocated(cuda_device) / 2**30
    initial_gpu_reserved_gib = torch.cuda.memory_reserved(cuda_device) / 2**30
    resident_kv_allocation_gib = engine.resident_kv_allocation_bytes(args.batch_size) / 2**30
    resident_kv_lower_bound_gib = initial_gpu_allocated_gib + resident_kv_allocation_gib
    if args.gpu_memory_limit_gib is not None:
        if initial_gpu_reserved_gib > args.gpu_memory_limit_gib:
            worker.close()
            raise RuntimeError(
                f"post-initialization reserved GPU memory {initial_gpu_reserved_gib:.3f} GiB "
                f"already exceeds the {args.gpu_memory_limit_gib:.3f} GiB limit"
            )
        if resident_kv_lower_bound_gib > args.gpu_memory_limit_gib:
            worker.close()
            raise RuntimeError(
                f"resident KV requires {resident_kv_allocation_gib:.3f} GiB in addition to "
                f"{initial_gpu_allocated_gib:.3f} GiB of persistent allocations, exceeding "
                f"the {args.gpu_memory_limit_gib:.3f} GiB limit before temporary workspace"
            )
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_limit_gib / total_gpu_gib,
            cuda_device,
        )
    torch.cuda.reset_peak_memory_stats(cuda_device)
    try:
        start = time.perf_counter()
        output = engine.prefill(input_ids, request_ids)
        synchronize(args.device)
        prefill_seconds = time.perf_counter() - start
        transfer_after_prefill, prefill_transfer = worker.phase_metrics_since(transfer_start)
        residency_after_prefill = (residency.evictions, residency.wasted_prefetches)
        start = time.perf_counter()
        provider.initialize(input_ids, request_ids)
        synchronize(args.device)
        draft_prefill_seconds = time.perf_counter() - start
        transfer_after_draft_prefill, draft_prefill_transfer = worker.phase_metrics_since(
            transfer_after_prefill
        )
        residency_after_draft_prefill = (residency.evictions, residency.wasted_prefetches)

        token_ids = output.logits[:, -1].argmax(dim=-1).cpu()
        generated = [token_ids]
        target_decode_seconds = []
        draft_seconds = []
        step_seconds = []
        selected_chunks = []
        predicted_mass = []
        target_mass_coverage = []
        relative_l2_error = []
        cosine_similarity = []
        shadow_seconds = 0.0
        threshold_sweep: dict[str, dict[str, list[float]]] = {}
        pending_actual: list[torch.Tensor] = []
        decode_start = time.perf_counter()
        first_rollout_start = decode_start
        plan = provider.predict(output.state)
        synchronize(args.device)
        first_rollout_seconds = time.perf_counter() - first_rollout_start
        remaining_horizons = plan.horizons[:refresh_tokens]
        for step_index in range(args.max_new_tokens - 1):
            # The first measured step starts before the initial rollout. Later
            # steps start at their own refresh/Target boundary.
            step_start = decode_start if step_index == 0 else time.perf_counter()
            draft_step_seconds = 0.0
            if not remaining_horizons:
                draft_start = time.perf_counter()
                provider.advance(torch.stack(pending_actual, dim=1))
                plan = provider.predict(output.state)
                synchronize(args.device)
                draft_step_seconds = time.perf_counter() - draft_start
                pending_actual.clear()
                remaining_horizons = plan.horizons[:refresh_tokens]
            target_start = time.perf_counter()
            output = engine.decode(
                token_ids,
                output.state,
                remaining_horizons[: args.prefetch_horizons],
                prefetch=not args.disable_prefetch,
                shadow_attention=args.shadow_attention,
                shadow_thresholds=shadow_thresholds,
            )
            synchronize(args.device)
            target_decode_seconds.append(time.perf_counter() - target_start)
            draft_seconds.append(draft_step_seconds)
            pending_actual.append(token_ids)
            remaining_horizons = remaining_horizons[1:]
            token_ids = output.logits[:, -1].argmax(dim=-1).cpu()
            generated.append(token_ids)
            step_seconds.append(time.perf_counter() - step_start)
            selected_chunks.extend(
                len(trace.selected_old_chunks) for trace in output.sparse_attention.values()
            )
            predicted_mass.extend(
                trace.predicted_mass for trace in output.sparse_attention.values()
            )
            target_mass_coverage.extend(
                trace.target_mass_coverage
                for trace in output.sparse_attention.values()
                if trace.target_mass_coverage is not None
            )
            relative_l2_error.extend(
                trace.relative_l2_error
                for trace in output.sparse_attention.values()
                if trace.relative_l2_error is not None
            )
            cosine_similarity.extend(
                trace.cosine_similarity
                for trace in output.sparse_attention.values()
                if trace.cosine_similarity is not None
            )
            shadow_seconds += sum(
                trace.shadow_seconds for trace in output.sparse_attention.values()
            )
            for trace in output.sparse_attention.values():
                for threshold, metrics in (trace.threshold_sweep or {}).items():
                    aggregate = threshold_sweep.setdefault(
                        threshold,
                        {
                            "selected_old_chunks": [],
                            "target_mass_coverage": [],
                            "relative_l2_error": [],
                            "cosine_similarity": [],
                        },
                    )
                    for name, value in metrics.items():
                        aggregate[name].append(value)
        decode_wall_seconds = time.perf_counter() - decode_start
    finally:
        worker.close()

    transfer_end, decode_transfer = worker.phase_metrics_since(transfer_after_draft_prefill)
    residency_end = (residency.evictions, residency.wasted_prefetches)
    peak_gpu_allocated_gib = torch.cuda.max_memory_allocated(cuda_device) / 2**30
    peak_gpu_reserved_gib = torch.cuda.max_memory_reserved(cuda_device) / 2**30
    memory_limit_satisfied = (
        args.gpu_memory_limit_gib is None or peak_gpu_reserved_gib <= args.gpu_memory_limit_gib
    )

    tokens = torch.stack(generated, dim=1)
    total_steps = decode_wall_seconds
    total_request = prefill_seconds + draft_prefill_seconds + decode_wall_seconds
    decode_count = args.max_new_tokens - 1
    result = {
        "configuration": {
            **vars(args),
            "prompts": str(args.prompts),
            "output": str(args.output),
            "probes": str(args.probes) if args.probes else None,
            "policy": "demand_only" if args.disable_prefetch else "speculative",
            "resolved_moe_backend": engine.moe_backend,
        },
        "timing_protocol": "batch_decode_after_all_prefix_caches_v1",
        "performance": {
            "prefill_seconds": prefill_seconds,
            "draft_prefill_seconds": draft_prefill_seconds,
            "first_rollout_seconds": first_rollout_seconds,
            "target_decode_seconds": sum(target_decode_seconds),
            "draft_refresh_seconds": sum(draft_seconds),
            "draft_decode_seconds": first_rollout_seconds + sum(draft_seconds),
            "decode_wall_seconds": decode_wall_seconds,
            "decode_step_seconds": decode_wall_seconds,
            "decode_tokens_per_request": decode_count,
            "decode_tokens": args.batch_size * decode_count,
            "request_seconds": total_request,
            "mean_tpot_ms": decode_wall_seconds / decode_count * 1000,
            "p50_tpot_ms": statistics.median(step_seconds) * 1000 if step_seconds else 0,
            "decode_throughput_tokens_per_second": args.batch_size * decode_count / total_steps
            if total_steps
            else 0,
            "end_to_end_tokens_per_second": args.batch_size * args.max_new_tokens / total_request,
            "peak_gpu_gib": peak_gpu_allocated_gib,
            "initialization_seconds": initialization_seconds,
            "expert_preload_seconds": expert_preload_seconds,
            "moe_warmup_seconds": moe_warmup_seconds,
            "shadow_attention_seconds": shadow_seconds,
            "latency_valid": not (args.shadow_attention or shadow_thresholds),
        },
        "gpu_memory": {
            "measurement_protocol": "post_initialization_peak_reserved_v1",
            "limit_gib": args.gpu_memory_limit_gib,
            "total_device_gib": total_gpu_gib,
            "peak_allocated_gib": peak_gpu_allocated_gib,
            "peak_reserved_gib": peak_gpu_reserved_gib,
            "initial_allocated_gib": initial_gpu_allocated_gib,
            "initial_reserved_gib": initial_gpu_reserved_gib,
            "resident_kv_allocation_gib": resident_kv_allocation_gib,
            "resident_kv_allocated_lower_bound_gib": resident_kv_lower_bound_gib,
            "allocator_limit_enforced": args.gpu_memory_limit_gib is not None,
            "limit_satisfied": memory_limit_satisfied,
        },
        "sparse_kv": {
            "mean_selected_old_chunks_per_layer_request": statistics.mean(selected_chunks)
            if selected_chunks
            else 0,
            "mean_predicted_mass": statistics.mean(predicted_mass) if predicted_mass else 0,
            "mean_target_mass_coverage": statistics.mean(target_mass_coverage)
            if target_mass_coverage
            else None,
            "mean_relative_l2_error": statistics.mean(relative_l2_error)
            if relative_l2_error
            else None,
            "mean_cosine_similarity": statistics.mean(cosine_similarity)
            if cosine_similarity
            else None,
            "threshold_sweep": {
                threshold: {
                    f"mean_{name}": statistics.mean(values) for name, values in metrics.items()
                }
                for threshold, metrics in threshold_sweep.items()
            },
        },
        "transfer": vars(transfer_end),
        "transfer_by_phase": {
            "prefill": vars(prefill_transfer),
            "draft_prefill": vars(draft_prefill_transfer),
            "decode": vars(decode_transfer),
        },
        "residency": {
            "evictions": residency_end[0] - residency_start[0],
            "wasted_prefetches": residency_end[1] - residency_start[1],
        },
        "residency_by_phase": {
            "prefill": {
                "evictions": residency_after_prefill[0] - residency_start[0],
                "wasted_prefetches": residency_after_prefill[1] - residency_start[1],
            },
            "draft_prefill": {
                "evictions": residency_after_draft_prefill[0] - residency_after_prefill[0],
                "wasted_prefetches": residency_after_draft_prefill[1] - residency_after_prefill[1],
            },
            "decode": {
                "evictions": residency_end[0] - residency_after_draft_prefill[0],
                "wasted_prefetches": residency_end[1] - residency_after_draft_prefill[1],
            },
        },
        "generated_token_ids": tokens.tolist(),
        "generated_text": [tokenizer.decode(row) for row in tokens],
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
