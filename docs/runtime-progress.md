# Sparse offload runtime progress

Updated: 2026-09-07 21:32 UTC  
Branch: `feature/sparse-offload-runtime`

## Objective and semantics

The runtime keeps sink KV and a bounded recent window on GPU. Old KV chunks and all
routed experts have authoritative pinned-CPU copies. A single logical queue orders KV
and expert H2D requests. Draft signals enter with probability and deadline; an actual
miss is promoted above every speculative request. Duplicate requests merge consumers
and are reprioritized.

Sparse KV uses the agreed hybrid stopping rule: cumulative draft mass, target marginal
partition contribution, a consecutive-low-marginal requirement, and a minimum chunk
count. Unseen Target mass is never used online.

## Implemented

- Thread-safe mutable priority queue with stale-entry versions, cancellation, batch
  reuse, demand promotion, and logical-time urgency updates.
- CPU/queued/in-flight/GPU-resident state machine with finite caches and priority-aware
  eviction leases.
- One transfer worker that owns a dedicated CUDA H2D stream; compute only submits or
  waits for dependencies.
- Sink/recent/old KV layout, pinned-CPU old chunks, sparse retrieval, next-step retention
  and eviction.
- Original Qwen3 safetensors expert source, including experts whose three matrices cross
  shard boundaries.
- Exact Qwen3 Top-8 routing and MoE formula, with grouped prefill and vectorized decode.
- Meta-device Qwen3 loader that places only non-expert weights on GPU; 54 GiB of experts
  never transit through GPU during initialization.
- Stateful Qwen3-0.6B predictor whose KV cache is rolled back after a four-token rollout.
  One rollout is consumed over four Target steps before refresh.
- Fixed-batch Qwen3 adapter and a framework-neutral continuous-batch admission core.
- Isolated vLLM 0.11.1 baseline harness and generation-quality comparison tool.

## Correctness evidence

- 46 CPU tests pass; one CUDA test is opt-in to avoid touching a busy GPU.
- On a random miniature Qwen3-MoE, custom dense prefill and incremental decode logits
  match the Transformers reference when all KV is selected.
- The vectorized and grouped expert executors match numerically.
- A real Qwen3-30B-A3B run completed with original checkpoint experts, real pinned CPU
  storage, real H2D, two old KV chunks per layer, and no transfer failures.
- Demand-only and speculative modes produced the same nine tokens in the current exact
  two-old-chunk workload.

## Current measured state

Workload: batch 1, context 16, output 9, sink 4, recent 4, KV chunk 4, expert cache 64.
The short context deliberately forces two old chunks while making full selection possible.

| System | GPU weight/residency | Request seconds | Throughput |
| --- | ---: | ---: | ---: |
| vLLM full resident, eager Triton MoE | 56.88 GiB model | 2.126 | 4.233 tok/s |
| vLLM `cpu_offload_gb=54` | 2.706 GiB model, 9.8 GiB budget | 2.675 | 3.365 tok/s |
| SpecFetch demand-only, grouped decode | 4.57 GiB peak | 13.556 | 0.664 tok/s |
| SpecFetch speculative, four admitted horizons | 4.57 GiB peak | 14.319 | 0.629 tok/s |
| SpecFetch speculative, one admitted horizon | 4.58 GiB peak | 14.075 | 0.639 tok/s |
| SpecFetch demand-only, vectorized decode | 4.64 GiB peak | 13.134 | 0.685 tok/s |

The production baseline was run with `VLLM_USE_DEEP_GEMM=0`,
`VLLM_MOE_USE_DEEP_GEMM=0`, and `--enforce-eager`. The PPU build otherwise JIT-warms
hundreds of DeepGemm shapes for minutes before a tiny request; disabling that path uses
vLLM's Triton FusedMoE fallback and gives a reproducible baseline.

The first speculative implementation is not yet a speedup. Relative to grouped
demand-only, it removes roughly 434 demand misses but adds queue work and DMA/GEMM
contention. Limiting admission to one horizon helps but does not reverse the result.
Vectorizing decode experts improves Target decode time from 4.370 to 4.098 seconds
(6.2%) and end-to-end throughput from 0.664 to 0.685 tok/s.

## Known gaps

1. The reference engine remains about 4.9x slower than vLLM weight offload on this tiny
   workload. Per-object allocation/copy and Python queue upserts remain expensive.
2. Expert tensors need fixed GPU slots and a grouped/fused MoE kernel; route-wise
   `torch.stack` is only an intermediate implementation.
3. Transfer metrics currently include prefill and decode together; phase deltas should be
   reported separately.
4. The continuous-batch state machine is tested but not yet wired into variable-length
   Qwen execution.
5. Batch-4, 512/4K contexts and threshold quality sweeps have not run yet.
6. vLLM and the custom adapter use different BF16 attention/MoE kernel orders. Their
   first six generated tokens match in the short test, after which rounding changes the
   greedy path. Quality must be assessed statistically, not by requiring bit identity to
   vLLM.

## Next actions

1. Introduce fixed packed GPU expert slots and coalesced queue admission.
2. Separate prefill/decode transfer counters and quantify cache hit utility.
3. Re-run demand/speculative batch-1 smoke, then batch 4 at context 512.
4. Run quality sweeps for mass thresholds 0.90/0.95/0.99 against a full-attention shadow.
5. Run vLLM full and 54-GiB weight-offload baselines at matching batch/context lengths.
6. Integrate request removal/backfill into the Qwen adapter, then measure continuous batch.

## Safety and versioning

All GPU entrypoints refuse to start on a selected busy physical GPU. CUDA tests require
`SPECFETCH_RUN_CUDA_TESTS=1`. Work was split into small pushed commits; no external GPU
jobs were killed or modified. Two interrupted vLLM processes created by this task were
explicitly terminated after their EngineCore children became zombies.
