from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3 sparse offload reference engine.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=4)
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
    return parser.parse_args()


def load_prompt_texts(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    prompts = value.get("prompts") if isinstance(value, dict) else value
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("prompts must contain a string array")
    return prompts


def fixed_contexts(tokenizer, prompts: list[str], batch: int, tokens: int) -> torch.Tensor:
    rows = []
    for prompt in prompts[:batch]:
        encoded = tokenizer(prompt, add_special_tokens=False).input_ids
        if not encoded:
            raise ValueError("an empty prompt cannot form a benchmark context")
        repeats = (tokens + len(encoded) - 1) // len(encoded)
        rows.append((encoded * repeats)[:tokens])
    if len(rows) != batch:
        raise ValueError("not enough prompts for requested batch size")
    return torch.tensor(rows, dtype=torch.long)


def synchronize(device: str) -> None:
    torch.cuda.synchronize(torch.device(device))


def main() -> None:
    args = parse_args()
    require_idle_gpus(1000, 10)
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
        args.batch_size,
        args.context_tokens,
    )

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
    backend = CudaTransferBackend(args.device)
    worker = TransferWorker(queue, residency, backend)
    runtime = OffloadRuntime(queue, residency, worker)
    engine = Qwen3SparseOffloadEngine(target, expert_source, runtime, residency, config)
    provider = DraftSignalProvider(draft, probes, args.lookahead)
    request_ids = [f"request-{index}" for index in range(args.batch_size)]
    worker.start()
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    try:
        start = time.perf_counter()
        output = engine.prefill(input_ids, request_ids)
        synchronize(args.device)
        prefill_seconds = time.perf_counter() - start
        provider.initialize(input_ids, request_ids)

        generated = []
        decode_seconds = []
        selected_chunks = []
        predicted_mass = []
        logits = output.logits[:, -1]
        for _ in range(args.max_new_tokens):
            token_ids = logits.argmax(dim=-1).cpu()
            plan = provider.predict(output.state)
            start = time.perf_counter()
            output = engine.decode(
                token_ids,
                output.state,
                plan.horizons,
                prefetch=not args.disable_prefetch,
            )
            synchronize(args.device)
            decode_seconds.append(time.perf_counter() - start)
            provider.advance(token_ids)
            logits = output.logits[:, -1]
            generated.append(token_ids)
            selected_chunks.extend(
                len(trace.selected_old_chunks) for trace in output.sparse_attention.values()
            )
            predicted_mass.extend(
                trace.predicted_mass for trace in output.sparse_attention.values()
            )
    finally:
        worker.close()

    tokens = torch.stack(generated, dim=1)
    total_decode = sum(decode_seconds)
    result = {
        "configuration": {
            **vars(args),
            "prompts": str(args.prompts),
            "output": str(args.output),
            "probes": str(args.probes) if args.probes else None,
            "policy": "demand_only" if args.disable_prefetch else "speculative",
        },
        "performance": {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": total_decode,
            "mean_tpot_ms": statistics.mean(decode_seconds) * 1000,
            "p50_tpot_ms": statistics.median(decode_seconds) * 1000,
            "throughput_tokens_per_second": args.batch_size * args.max_new_tokens / total_decode,
            "peak_gpu_gib": torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**30,
        },
        "sparse_kv": {
            "mean_selected_old_chunks_per_layer_request": statistics.mean(selected_chunks),
            "mean_predicted_mass": statistics.mean(predicted_mass),
        },
        "transfer": vars(worker.metrics),
        "residency": {"evictions": residency.evictions},
        "generated_token_ids": tokens.tolist(),
        "generated_text": [tokenizer.decode(row) for row in tokens],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
