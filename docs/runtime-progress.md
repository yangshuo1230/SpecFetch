# Sparse offload runtime progress

Updated: 2026-09-08 07:55 UTC
Branch: `feature/sparse-offload-runtime`

## 协作约定

- 后续进度更新、问题说明、文档新增内容和最终交付说明全部使用中文。
- GPU 被其他任务占用时不停止整体推进：优先完成 CPU 可验证的实现、测试、审计与文档；
  GPU 空闲后按门禁顺序补跑 CUDA 正确性和性能实验，再继续后续任务。

## Objective and semantics

The runtime keeps sink KV and a bounded recent window on GPU. Old KV chunks and all
routed experts have authoritative pinned-CPU copies. A single logical queue orders KV
and expert H2D requests. Draft signals enter with probability and deadline; an actual
miss is promoted above every speculative request. Duplicate requests merge consumers
and are reprioritized.

Sparse KV uses the agreed hybrid stopping rule: cumulative draft mass, target marginal
partition contribution, a consecutive-low-marginal requirement, and a minimum chunk
count. Unseen Target mass is never used online.

## Work breakdown and completion gates

| Phase | Deliverable / gate | Status |
| --- | --- | --- |
| 1. Semantics | Unified mutable queue; demand dominates speculation; duplicate reprioritization | Complete |
| 2. Physical memory | Real pinned CPU expert/KV sources, finite GPU residency, dedicated H2D stream | Complete |
| 3. Qwen execution | Meta non-expert load, exact Top-8 MoE, sparse KV decode, fixed-batch correctness | Complete |
| 4. Prediction | Stateful four-token Draft rollout, KV ranking, expert probes, hybrid online stop | Complete |
| 5. Serving | Variable-length admission, request removal/backfill, request-level TTFT/latency | Implemented; full-model run pending |
| 6. Optimization | Packed slots, coalesced queue, batched H2D, fused slot-mapped MoE | In progress |
| 7. Evaluation | Demand/spec causality match, shadow quality, batch 1/4, 512/4K, vLLM baselines | 512 complete; 4K pending |
| 8. Release | Final regression, progress/results update, merge to `main`, push | Pending |

## Implemented

- Thread-safe mutable priority queue with stale-entry versions, cancellation, batch
  reuse, demand promotion, and logical-time urgency updates.
- CPU/queued/in-flight/GPU-resident state machine with finite caches and priority-aware
  eviction leases.
- One transfer worker that owns a dedicated CUDA H2D stream; compute only submits or
  waits for dependencies.
- Coalesced speculative admission and same-class transfer batches. A decode layer
  promotes all missing experts before waiting, then issues their copies with one CUDA
  stream synchronization. Demand and speculation are never mixed in one transfer batch.
- Fixed packed expert slots: gate/up share a fused weight bank and evicted slots are
  overwritten in place rather than allocated again. Prefill batches experts within the
  physical-slot bound; compute-stream CUDA events prevent H2D from overwriting a slot
  while a kernel still reads it.
- 融合 MoE 的 logical-to-physical `expert_map` 先在 CPU 完整构造，再一次性复制到计算
  设备，取代最多 64 次逐 expert 的 GPU 标量写入；越界逻辑 expert 会在进入 kernel 前
  明确报错。实际性能收益与 CUDA 数值仍按空闲后的门禁实验确认。
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
- Runnable variable-length continuous batching with independent per-request Draft caches,
  separate prefill admission, completion removal, backfill, and request-private KV cleanup.
- 连续批处理结果现已分别记录排队时延、Target 预填充、解码服务、活跃服务、TTFT
  和端到端请求时延。单 token 请求会在预填充 token 可用时立即完成，不再初始化无用的
  Draft provider；峰值活跃请求数在完成移除前采样，因此纯单-token 工作负载也不会误报
  为零。
- 同一轮准入中，prompt 长度相同的请求会合并为一次 Target 预填充，并继续保留独立的
  Draft provider。测试覆盖批量预填充后部分请求立即完成、其余请求继续解码及私有 KV
  清理；结果同时报告预填充批次数和最大预填充批大小。
- 同组继续生成的请求现在只执行一次批量 Draft 前缀预填充，再通过 Transformers 的
  DynamicCache batch split 拆成可独立推进的请求私有 KV；结果另行报告 Draft 预填充
  批次数和最大批大小。首轮 lookahead 也在拆分前按组执行，随后按请求拆分 KV/专家信号；
  测试验证拆分后再按任意请求顺序合并可恢复原信号。
- lookahead 耗尽后的 Draft 刷新会按模型、probe、lookahead、缓存长度和待推进 token 数分组；
  兼容请求临时合并 KV，批量 advance/rollout 后再拆回私有状态。不同长度或不同进度的
  请求不会强行合并；结果报告刷新批次数和最大刷新批大小。
- 每个 horizon 的 Draft attention 和 hidden feature 现在按唯一 Draft 层一次性搬到 CPU
  并复用，避免对每个 Target 层、请求和旧 KV 块反复触发标量同步；lookahead token 也改为
  在 rollout 末尾一次性同步。实际 GPU 时延收益待空闲后复测，不以 CPU 测试代替性能证据。
- Optional vLLM Triton FusedMoE adapter over physical expert slot IDs, with the readable
  PyTorch executor retained as the default until the CUDA path is benchmarked.
- Phase-separated transfer/residency counters and an opt-in CPU full-attention shadow.
  One shadow run evaluates 0.90/0.95/0.99 thresholds counterfactually and reports Target
  mass coverage, relative L2 error, cosine similarity, and selected chunk count.
- Isolated vLLM 0.11.1 baseline harness and generation-quality comparison tool.

## Correctness evidence

- 当前 66 项 CPU 测试全部通过；两项 CUDA 测试为显式启用，以免触碰繁忙 GPU。本检查点
  再次通过完整 CPU 测试、Ruff lint/format 与 `git diff --check`。
- On a random miniature Qwen3-MoE, custom dense prefill and incremental decode logits
  match the Transformers reference when all KV is selected.
- The vectorized and grouped expert executors match numerically.
- A real Qwen3-30B-A3B run completed with original checkpoint experts, real pinned CPU
  storage, real H2D, two old KV chunks per layer, and no transfer failures.
- Demand-only and speculative modes produced the same nine tokens in the current exact
  two-old-chunk workload.
- Tiny Qwen3-MoE tests exercise variable prompt/output lengths, completion backfill,
  packed-slot reuse, a cache capacity smaller than the routed expert set, logical-to-
  physical fused routing, batched demand copies, and full-attention shadow metrics.
- On physical GPU 1, packed/batched H2D and the portable vLLM Triton FusedMoE path
  passed against a PyTorch reference (relative L2 below 1%, cosine above 0.9999).
  The platform AC-MoE shortcut was rejected after producing incorrect output; the
  adapter now forces portable Triton and disables unbounded DeepGemm warmup.

## Current measured state

### Post-optimization batch 1 smoke

Workload: batch 1, context 16, output 9, sink 4, recent 4, KV chunk 4.

| Policy / MoE | Request seconds | Target decode | Throughput | Generated tokens |
| --- | ---: | ---: | ---: | --- |
| Demand-only / PyTorch | 12.267 | 3.903 | 0.734 tok/s | reference |
| Speculative H1 / PyTorch | 12.513 | 4.233 | 0.719 tok/s | identical |
| Demand-only / vLLM Triton | 13.694 | 5.375 | 0.657 tok/s | identical |

Batching/fixed slots improved demand-only throughput from 0.685 to 0.734 tok/s (7.1%).
At batch 1 speculation is still 2.0% slower: it saves about 248 ms of cumulative demand
wait but adds 199 ms of queue admission plus DMA/compute contention. The vLLM functional
kernel has comparable median TPOT but a large first-shape JIT cost, so PyTorch remains the
default for this reference engine.

### Batch 4, context 512 result

Workload: batch 4, context 512, output 5, sink 4, recent 256, KV chunk 64.

| System | Request seconds | Target decode | Throughput |
| --- | ---: | ---: | ---: |
| SpecFetch demand-only | 26.452 | 9.205 | 0.756 tok/s |
| SpecFetch speculative H1 | **23.923** | **6.753** | **0.836 tok/s** |
| vLLM `cpu_offload_gb=54` | 6.524 | n/a | 3.066 tok/s |
| vLLM full resident | 0.457 | n/a | 43.717 tok/s |

At the target batch size speculation is a real speedup: 10.6% end-to-end throughput,
26.6% Target-decode latency reduction, and 48.0% less cumulative decode demand wait.
It turns 2,144 of 7,083 decode demands into hits while increasing total H2D bytes by only
0.9%. Demand and speculative runs generate the same 20 tokens. The remaining gap to the
fair production weight-offload baseline is 3.67x; this does not yet satisfy the desired
"roughly baseline" performance gate.

### Full-attention shadow

Batch 1/context 512/output 5 evaluates each sparse layer output against all CPU old-KV
chunks. Thresholds 0.90, 0.95, and 0.99 all select three old chunks on average because
the Target marginal-patience rule dominates. They obtain 99.69% mean Target attention
mass, 0.389% CPU counterfactual relative L2 error, and 0.999965 cosine similarity. Shadow
work takes 87.6 seconds and is correctly excluded from performance claims.

### Historical pre-optimization smoke

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
| SpecFetch speculative, vectorized decode, horizon 1 | 4.64 GiB peak | 13.524 | 0.665 tok/s |

These measurements predate fixed slots, coalesced admission, batched expert demand,
scoped residency leases, and the optional fused kernel. They are retained as the
pre-optimization baseline and must not be presented as current optimized performance.

The production baseline was run with `VLLM_USE_DEEP_GEMM=0`,
`VLLM_MOE_USE_DEEP_GEMM=0`, and `--enforce-eager`. The PPU build otherwise JIT-warms
hundreds of DeepGemm shapes for minutes before a tiny request; disabling that path uses
vLLM's Triton FusedMoE fallback and gives a reproducible baseline.

The first speculative implementation was not a speedup at batch 1. This remains a useful
low-concurrency counterexample; the batch-4 result above is the current primary evidence.

## Known gaps

1. The reference runtime is still 3.67x slower than vLLM 54-GiB weight offload at batch
   4/context 512. SpecFetch prefill alone takes 15.3 seconds because it uses per-expert
   PyTorch GEMMs; slot-mapped fused prefill is the current optimization task.
2. Slot-mapped fused prefill has CPU mapping tests, but its newest multi-chunk CUDA test
   was refused when an unrelated eight-process image-generation job reoccupied all four
   GPUs (about 45.5 GiB each at 70--100% utilization). It must pass before a real-model
   fused-prefill benchmark is trusted; no external process was disturbed.
3. Context 4K, full-model continuous batching, and longer-output steady-state runs remain.
4. 连续 runner 已合并同长度准入请求；不同 prompt 长度仍需分组串行执行。分块预填充和
   预填充/解码的 kernel 级交错仍属于后续生产集成工作。
5. vLLM and the custom adapter use different BF16 attention/MoE kernel orders. Their
   first six generated tokens match in the short test, after which rounding changes the
   greedy path. Quality must be assessed statistically, not by requiring bit identity to
   vLLM.

## Next actions

1. When a GPU is idle, pass the multi-chunk slot-mapped fused CUDA test, then benchmark
   fused prefill on batch 4/context 512. Keep it only if outputs pass and the production
   baseline gap materially narrows.
2. Run batch 4/context 4K demand/speculative and matching vLLM baselines; use a longer
   output window to separate prefill from steady-state decode.
3. Run the variable-output continuous-batch CLI and report throughput plus request-level
   latency; then add chunked prefill only if profiling shows admission stalls dominate.
4. Update this document after each evidence-producing step, merge the feature branch to
   `main`, and push only after every correctness/performance gate is accounted for.

## Safety and versioning

All GPU entrypoints refuse to start on a selected busy physical GPU. CUDA tests require
`SPECFETCH_RUN_CUDA_TESTS=1`. Work was split into small pushed commits; no external GPU
jobs were killed or modified. Two interrupted vLLM processes created by this task were
explicitly terminated after their EngineCore children became zombies.
