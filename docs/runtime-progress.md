# Sparse offload runtime progress

Updated: 2026-09-09 06:07 UTC
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

当前阶段的优化和发布门禁只针对 steady-state decode。模型加载、kernel warmup、Target/
Draft prefix prefill 与 prefix cache 构造不计入主性能指标；prefill 可以采用与本项目解耦的
直接高效实现，不要求走 speculative offload 路径。计时边界设在两侧 prefix cache 准备完成
之后、首次 speculative rollout 之前，因此首次 rollout、后续 refresh、queue/residency
调度、H2D、demand wait 和 Target decode 均计入 decode wall time，不能把必要工作藏入
prefill。

主指标改为固定长度 steady-state decode 的 batch wall time、TPOT 和 decode tokens/s；
冷 residency 与稳态热缓存分别报告。vLLM 对照必须使用相同模型、精度、batch、context、
输出长度和 GPU 显存约束，并从首 token 后的匹配区间计算 decode-only 指标。发布硬门禁是
SpecFetch decode tokens/s 至少为 vLLM CPU-weight-offload 的 1.50 倍；固定 token 数下等价为
decode wall time 和 TPOT 至多是 vLLM 的 0.667 倍。full-resident vLLM 只作为硬件上界。
Python 路径继续承担语义参考和回归测试，profile 证明处于关键路径的调度、同步和 kernel
必须允许下沉至 C++/CUDA、融合 kernel、CUDA Graph 或设备侧调度。

预先声明的主发布 workload 为 batch 4、context 512 和 batch 4、context 4096；两者均在
首 token 之后计时 64 个固定 decode token，并分别用一致的冷 residency 起点比较 SpecFetch
与 vLLM。batch 1 用于定位低并发开销，128-token 运行用于确认稳态趋势，但不得替代两个
主 workload。kernel/JIT warmup 可以在计时外完成，不过必须在开始计时前恢复声明的
residency/cache 起点，避免把资源预取伪装成免费 prefill。

## Work breakdown and completion gates

| Phase | Deliverable / gate | Status |
| --- | --- | --- |
| 1. Semantics | Unified mutable queue; demand dominates speculation; duplicate reprioritization | Complete |
| 2. Physical memory | Real pinned CPU expert/KV sources, finite GPU residency, dedicated H2D stream | Complete |
| 3. Qwen execution | Meta non-expert load, exact Top-8 MoE, sparse KV decode, fixed-batch correctness | Complete |
| 4. Prediction | Stateful four-token Draft rollout, KV ranking, expert probes, hybrid online stop | Complete |
| 5. Serving | Variable-length admission, request removal/backfill, request-level TTFT/latency | Complete; full-model run passed |
| 6. Optimization | Decode-only pipeline: packed slots, coalesced queue/H2D, fused kernels, C++/CUDA hot paths | In progress |
| 7. Evaluation | Causality/quality plus matched 512/4K decode; SpecFetch throughput >=1.50x vLLM offload | In progress; decode-only runners complete, matched GPU measurements pending |
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
- 每个预测窗口的全部 expert 与 KV speculative 请求现在先跨 horizon、层和请求汇总，再以
  一次队列事务提交；deadline 仍由统一堆排序。batch-4/context-512 的调用结构由每步 240
  次提交降为每步 1 次；后续 c512 fixed/continuous 实模结果已包含该路径。
- 上一预测窗口的 consumer 也通过批量事务撤销：队列按资源聚合 consumer、每个资源最多
  重建一次堆项，驻留 lease 在一次锁内统一刷新；耗尽 consumer 的 QUEUED 资源会原子地
  回到 CPU_ONLY。测试覆盖共享请求保留、重复取消去重和完全撤销。
- QUEUED residency 现在也保存待处理 consumer lease；worker 在 `pop` 后开始传输前会在
  同一 residency 锁内重读有效 lease。若该窗口已被撤销则原子回到 CPU_ONLY 并跳过 H2D，
  封住 `pop -> cancel -> begin_transfer` 竞态，确定性测试覆盖该时序。
- speculative admission 对驻留状态的分类、CPU_ONLY 到 QUEUED 的状态迁移，以及
  resident/in-flight consumer lease 写入现在合并到一次驻留锁；同一资源每批只刷新一次
  优先级。8,448 请求合成基准的 enqueue 中位数由约 96.5 ms 降至 73.5 ms（-23.8%），
  cancel 维持约 34.1 ms。
- Sparse KV 会先按 Draft 累计质量确定必然要取的候选前缀，并通过一次 `demand_many`
  提交；达到预测质量阈值后仍逐块计算 Target marginal 并在线停止，因而不改变选块或
  输出语义。batch-4/context-512 demand 实测 transfer batch 从 2,688 降至 1,171，
  请求时延由 24.578 s 降至 24.078 s。4K demand 的 decode transfer batch 从
  150,412 降至 7,006（-95.3%），decode wait 从 281.6 s 降至 258.7 s（-8.1%），
  请求时延从 466.436 s 降至 451.895 s（-3.1%），68 个 token 保持完全一致。
- Fixed packed expert slots: gate/up share a fused weight bank and evicted slots are
  overwritten in place rather than allocated again. Prefill batches experts within the
  physical-slot bound; compute-stream CUDA events prevent H2D from overwriting a slot
  while a kernel still reads it.
- 融合 decode 的 logical-to-physical `expert_map` 现在按层持久保存在设备上，由 transfer
  stream 仅在 packed slot 分配或释放时批量更新；decode 主线程不再为 48 个 MoE 层逐层
  创建 CPU map 并 H2D。超过 slot 容量的分组 prefill 仍使用精确的单组 map，避免把其他
  驻留 expert 重复累加。实际性能收益与 CUDA 数值仍按空闲后的门禁实验确认。
- 可选 vLLM MoE 在正式请求计时前按实际预填充/解码 token 数执行纯 MoE 形状预热；预热
  后驱逐全部专家驻留，并在快照边界清除预热产生的 transfer/residency 计数，使正式请求
  仍从冷专家驻留开始。预热耗时单独报告，可用 `--disable-moe-warmup` 关闭；lazy expert
  store 模式自动跳过，避免提前加载权重。CUDA 数值与 c512/4K 实模均已验证。
- 固定输出长度的 vLLM baseline 现在显式设置 `ignore_eos=True`，避免某个请求提前遇到
  EOS 后少生成 token，使自定义运行时和生产基线保持完全相同的 token 数。
- 两侧 release runner 现在接受同一个绝对 `gpu_memory_limit_gib`；vLLM 从物理设备容量
  反推实际 `gpu_memory_utilization`，两侧均在初始化结束后重置峰值统计并记录整个请求的
  allocated/reserved 峰值。release gate 会拒绝缺失、不同、超过上限或未实际传入 vLLM
  engine 的显存约束，旧的无显存证据 JSON 不再能进入发布结算。
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

- 当前 90 项 CPU 测试全部通过；三项 opt-in CUDA 测试中，前两项曾在此前检查点于物理
  GPU 2 通过；新增 persistent expert-map CUDA 集成测试尚待 GPU 空闲后执行。既有测试
  包括 multi-chunk slot-mapped fused MoE、容量 2/3、稀疏/全局 expert map 与事件槽复用。
  最新滚动准入/GQA 改动已通过完整 CPU 测试、Ruff lint/format 与 `git diff --check`；
  CUDA 和实模门禁因四张 GPU 均被外部任务占用而待跑，不能沿用上一提交替代。
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
| SpecFetch demand-only / vLLM MoE warmup | 24.578 | 8.626 | 0.814 tok/s |
| SpecFetch speculative H1 / vLLM MoE warmup | **22.023** | **6.600** | **0.908 tok/s** |
| vLLM `cpu_offload_gb=54` | 6.524 | n/a | 3.066 tok/s |
| vLLM full resident | 0.457 | n/a | 43.717 tok/s |

该表是调整目标前的历史端到端实验，`Request seconds` 与端到端吞吐不再作为当前发布门禁。
其中 SpecFetch 的 `Target decode` 也不是完整的新口径：首次 Draft rollout 被旧 runner 计入
`draft_prefill_seconds`，而 vLLM 又没有独立 decode 计时，因此不能据此声称已经接近或超过
vLLM。后续统一使用下文定义的长输出 decode-only protocol。

At the target batch size speculation is a real speedup: 10.6% end-to-end throughput,
26.6% Target-decode latency reduction, and 48.0% less cumulative decode demand wait.
It turns 2,144 of 7,083 decode demands into hits while increasing total H2D bytes by only
0.9%. Demand and speculative runs generate the same 20 tokens. 真实形状 warmup 后的 vLLM
融合 MoE 路径保持 20 个 token 完全一致，并把 speculative 请求时延再降低 8.6%；剩余
生产权重卸载基线差距由 3.67x 缩至 3.38x，但仍不满足期望的
当时使用的端到端 "roughly baseline" gate；该旧门禁现已被上文的 decode-only 1.50x
硬目标取代。

KV guaranteed-prefix 批量 demand 的首轮 c512 demand 复测为 24.078 s、0.831 tok/s；
它比同后端批处理前快约 2.0%，但尚未重跑 speculative，所以上表保留成对可比结果。

### Batch 4, context 4K, longer decode

Workload: batch 4, context 4096, output 17, sink 4, recent 256, KV chunk 64.
所有系统在 `ignore_eos=True` 的固定长度口径下生成相同 68 个 token。

| System | Request seconds | Throughput | Relative to vLLM offload |
| --- | ---: | ---: | ---: |
| SpecFetch demand-only / vLLM MoE | 466.436 | 0.146 tok/s | 15.26x slower |
| SpecFetch demand-only / batched KV demand | 451.895 | 0.150 tok/s | 14.79x slower |
| SpecFetch speculative H1 / vLLM MoE | 610.714 | 0.111 tok/s | 19.98x slower |
| vLLM `cpu_offload_gb=54` | 30.561 | 2.225 tok/s | reference |
| vLLM full resident | 2.382 | 28.549 tok/s | 12.83x faster |

该表同样保留为历史诊断。端到端相对倍数不再是当前门禁，但 SpecFetch 已记录的 Target
decode/demand wait 仍表明 4K decode 存在数量级瓶颈；必须用匹配的 vLLM decode-only
结果重新定量，不能因排除 prefill 而忽略该问题。

混合停止规则在该 4K 工作负载平均选择 48.71 个旧块。旧的逐块 blocking demand
实现使 demand decode 产生 169,118 次 miss、150,412 个 transfer batch，累计 wait
281.6 s。H1 虽得到 7,982 个 decode hit，却提交 208,128 个预测请求、丢弃 186,514 个，
并因竞争把 decode demand wait 增至 296.9 s，所以比 demand-only 慢 30.9%。该结果是
重要的长上下文反例；它早于 guaranteed-prefix KV 批量 demand，不能直接代表当前路径。

批量 KV demand 复测保持相同的选块统计和 68 个 token，将 decode transfer batch 降低
95.3%，但端到端仅改善 3.1%，说明长上下文瓶颈不只是 CUDA 同步次数。针对 H1 的下一
实验改为按层滚动准入：只提交当前层附近的 deadline，完成一层后再补入后续层；窗口内
还可按同一队列优先级公式限制 expert/KV 唯一资源数。该实现已通过 CPU 因果与队列测试，
并会在当前层最后一次可能消费后撤销过期 consumer，同时保留下个 token 的近端 consumer。
待测分支还把同层整个 batch 的 guaranteed KV 合成一次 demand（超过容量时自动回退），
将必需前缀的 Target marginal 合并为一次 CPU 同步，并用 grouped-query contraction 避免
物理复制 KV head。真实 GPU 数值与时延尚未验证，因此当前表中仍保留旧 H1 反例，不作
收益声明。

### Full-model continuous batching

Workload: 8 requests, maximum batch 4, context 512, output lengths 4/8/12/16 repeated;
两种策略逐请求生成相同 80 个 token，均发生 5 次 admission/backfill，峰值活跃请求为 4。

| Policy | Request seconds | Throughput | Mean TTFT | Mean request latency | Mean queue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Demand-only / vLLM MoE | 97.347 | 0.822 tok/s | 30.078 s | 65.328 s | 21.307 s |
| Speculative H1 / vLLM MoE | **91.363** | **0.876 tok/s** | **29.456 s** | **61.112 s** | **20.207 s** |

speculative 提升吞吐 6.55%，平均请求时延降低 6.45%。平均 active service 为
40.9--44.0 s，明显高于 queue；admission stall 不占主导，因此当前证据不支持立即实现
chunked prefill。

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

1. decode-only runner 已实现但尚未在空闲 GPU 上实测：SpecFetch 在 Target/Draft prefix
   cache 就绪后、首次 rollout 前启动 batch wall timer；vLLM 逐步驱动 offline engine，在
   整批每个请求恰好返回首 token 后启动 timer。两侧默认总输出 65 token，即计时后续
   64 token/request，并直接报告相同的 wall、TPOT、token 数和 tokens/s 字段。
2. 4K H1 预测覆盖远低于提交规模并增加 demand wait；需要依据批量 KV 复测重新决定
   长上下文的 admission budget/背压，不能沿用 c512 的无界候选提交。
3. Python queue/residency、逐层控制流和标量同步仍可能主导 TPOT；最新 GPU profile 后需要
   明确 C++/CUDA 下沉边界，不能把 Python reference 当作最终性能实现。
4. 尚无任何配置满足 `SpecFetch decode tokens/s / vLLM decode tokens/s >= 1.50`；在匹配
   decode-only 基线建立前，不得用 Target-kernel 子计时或端到端旧结果代替该门禁。
5. vLLM and the custom adapter use different BF16 attention/MoE kernel orders. Their
   first six generated tokens match in the short test, after which rounding changes the
   greedy path. Quality must be assessed statistically, not by requiring bit identity to
   vLLM.
6. 新增的 opt-in `--kv-storage resident` 已在 CPU 小模型上验证两步 decode、连续准入/
   完成以及完整序列 reference 一致。它按 uniform-length prefill group/层分配 batch-
   contiguous KV，以批量原位 append 和每组/层一次原生 GQA SDPA 绕过 KV queue、旧块
   拼接与逐块 marginal 同步；完成请求的行会压紧到更小分配，不同长度的新准入保持独立
   group。但它尚未在真实 GPU 上验证 10 GiB 上限、速度或数值，因此不改变默认 sparse
   配置，也不构成性能结论。
7. release runner 现在会在 prefill 前精确计算 resident KV 的持久 buffer payload；若它与
   初始化后 persistent allocation 的下界已超过匹配显存上限则直接拒绝，并将该绝对上限
   施加到 PyTorch CUDA caching allocator。输出同时记录初始化 allocated/reserved、resident
   payload、下界与实测峰值，避免以先 OOM 或事后超限作为唯一反馈。
8. resident KV 不再要求 Draft 输出各层 attention，也不再把 attention 搬到 CPU 后执行无用
   的 chunk 聚合；专家 probe 所需 hidden states 独立保留。这消除了随 Draft 层数、head 数和
   context 长度增长的纯预测开销。预测 admission 也跳过逐层逐请求的 KV retain/prefetch
   调用，同时继续提交 expert 请求；真实 GPU TPOT 收益仍需空闲设备验证。
9. Target router 的实际 Top-K 现在每层只生成一次很小的 CPU route-ID snapshot；unique expert
   与 demand consumer 映射都在该 snapshot 上完成，避免此前每个 expert 各自执行 GPU
   `where` 并逐标量同步回 Python。GPU selected/routing 仍原样进入 fused MoE，exact Top-K
   语义不变。
10. Draft rollout 的 host signal 提取已从逐 horizon、逐 draft layer 的小张量 `.cpu()` 改为
    attention 每 horizon 一次、expert hidden features 每次完整 rollout 一次；expert probe
    同时把全部 horizon 合成一个 CPU batch。对 28 层、lookahead 4 的 release Draft，单次
    refresh 的 expert-feature D2H 同步次数由最多 112 降为 1，输出语义保持不变。
11. Expert registry 的 preload 命中新增只读无锁快路径，且同一 layer/horizon 的跨请求重复
    expert 只解析一次。以 batch 4、48 Target layers、4 horizons、Top-8 的窗口为例，避免了
    最多 6,144 次没有状态变化的 registry lock 进入。
12. `resident + demand-only`（以及 resident 下没有 expert probes）的 decode 不再加载、prefill
    或 rollout Draft：完整 KV 已驻留且没有 expert speculative consumer，此时所有 Draft 输出
    都是死数据。JSON 新增 resolved `draft_signals_enabled` 和
    `prediction_prefetch_enabled`，避免把该策略误记为 speculative。
13. vLLM MoE backend 现同时接入 vLLM fused router kernel，将每层的 FP32 softmax、Top-K、
    Top-K sum/divide 归一化合并；torch backend 继续保留显式 reference 路径。CPU profile 中
    独立 softmax 是 resident 小模型最重 router 算子，正式 GPU 收益仍由后续 matched run 验证。
14. Expert eviction 在相同 priority/deadline 下改为 layer-balanced LRU：先从驻留数最多的层
    选择 victim，再按 LRU。确定性两层循环 trace 在 capacity 3 / working-set 4 下，普通全局
    LRU 第二轮 0 hit，而新策略保留每层代表并得到 2 hit；KV eviction 不受影响。
15. `demand_many` 的 residency hit/state/promotion/get 和完成等待改成两个批量临界区（等待前、
    等待后），替代每个 expert/KV 各自执行 state/record/mark/get/wait。每层最多约 32 个实际
    expert 的 queue upsert 与传输批次语义保持不变，但 Python 锁往返不再随资源数线性增加。
16. 同一次 MoE 调用使用的全部 experts 现在通过一个共享 compute-stream CUDA event 保护 slot
    复用，并用一次 `release_many` 清除 demand scope；此前每个 expert 都分别创建/record event
    并获取 residency 锁。分容量执行时仍按实际 fused kernel batch 各记录一个 event。
17. CPU expert store 将同形状 gate/up 预打包成共享 storage 的相邻 views，与 GPU fused
    `gate_up` slot 布局一致；每个 expert 的 H2D copy 提交由 gate/up/down 三次降为 gate_up/down
    两次，且最终 CPU payload 字节数不增加。独立 tensor 的兼容路径仍保留三次 copy。
18. Layer-balanced expert eviction 在层内新增 actual-demand frequency tie-break，再以 LRU
    决胜；稳定高频 route 不会因一次冷 expert 到达而被逐出，且纯 speculative arrival 不增加
    frequency。定向 trace 验证了高频但 LRU 更旧的 expert 被保留。
19. Draft expert probes 从 48 个逐层小 GEMM 合并为一次 batched GEMM，probe 参数 stack 首次
    构建后缓存；每层 predicted expert route 也从逐 request Top-K 改为整个 batch 一次二维
    Top-K/tolist。所有 layer/horizon/request 输出布局均有逐层 reference 对照测试。
    Release 维度合成微基准（48 layers、16 samples、1024→128）中位数由 526.96 ms 降至
    45.43 ms，即 CPU probe 阶段约 11.6x。
20. Release runner 新增 packed expert-slot 精确 payload 与 model+expert+resident-KV 持久下界；
    allocator cap 提前到模型加载前生效，lazy slots 在 post-init 下界中也不会漏算。输出记录
    base model、expert slots、resident KV 及两阶段 lower bound，为安全扩大 expert cache 提供依据。
21. Target serving prefill 的 lm_head 只投影最后一个 prefix hidden row；full-sequence logits 改为
    显式 reference 选项。batch 4/context 4096/151,936 vocab 的 BF16 全 logits 约 4.64 GiB，
    旧路径生成后只读取末行；新路径消除该临时分配，同时末 token 与 full reference 对齐。
22. 新增 opt-in `--maximize-expert-cache`：在 matched cap 中扣除实测 base allocation、精确
    resident KV payload 和显式 `--gpu-workspace-reserve-gib` 后，以完整 expert slot 为单位填满
    剩余空间，并记录 resolved slots。默认仍为 64，待两项真实工作负载确定共同安全 reserve。
23. Draft initialize、multi-token actual advance 和 rollout 均显式使用 Transformers 原生
    `logits_to_keep=1`。其中 4K/batch4 initialize 旧路径同样会产生约 4.64 GiB 全 vocab
    prefix logits；新路径只保留下一 token 所需末行，并有所有 Draft forward 参数回归断言。
24. Packed expert eviction 的 device map `=-1` 不再逐 expert 发射标量 kernel；host ownership
    立即删除，device removals 与下一 transfer batch assignments 合成每 layer 一次 indexed
    update。同一 MoE batch 共享的 completion event 在 transfer stream 也只 wait 一次。
25. 32-expert 全命中层的 CPU 微基准中，batched demand+release 为 128.8 us（48 层约
    6.2 ms/token），其中重复 lease `sum/min` 是最大 Python 项。现改为仅在 lease 变化时缓存
    aggregate priority/deadline；demand enter/release 只切换 infinity 与缓存值。相同微基准降至
    77.7 us/layer、3.73 ms/48 layers，约减少 39.7%。
26. 同一 profile 的最大剩余 Python 项是 immutable `ResourceKey` 的重复组合 hash；现于 key
    构造时缓存，queue/residency/transfer 的所有 dict/set lookup 复用该值，并保留完整相等性。
    相同 32-expert hit 微基准进一步由 77.7 降至 46.9 us/layer（48 层 3.73→2.25 ms），
    再减少约 39.6%。
27. 每层 `queue.set_step` 的动态 urgency 重排从清空后逐资源 `heappush`（O(N log N)）改为
    一次构造 heap array 后 `heapify`（O(N)）；priority/deadline/sequence 排序语义保持，且测试
    禁止 step rebuild 调用 incremental `_push`。1,000 个 queued resources 的 200 轮 CPU
    微基准为 1.001→0.816 ms/rebuild，约 1.23x，并避免随 N 增长的 logarithmic factor。
28. Queue request 的 `expected_uses=sum(consumer probabilities)` 也改为仅在 upsert/cancel 时
    缓存；逐层仅 urgency 变化的 heap rebuild 不再扫描 consumer dict，并以禁止 `.values()`
    的回归对象验证重建路径只读取缓存。相同 1,000-request 基准进一步从 0.816 降至
    0.681 ms/rebuild，约减少 16.5%。
29. Speculative consumer state 从扁平 list 改为 `(layer, absolute-token consumer)` buckets；
    `_retire_prediction_layer` 直接 pop 当前 layer/request buckets，不再每层过滤全部未来
    layers/horizons，消除 48 层下的二次扫描。请求完成也按 owner bucket 汇总后一次 cancel。
    48 layers、2 horizons、batch 4、Top-8 的完整 enqueue/retire/reset CPU 微基准为
    10.314→0.831 ms/window，约 12.4x。
30. Target 的 bias-free Q/K/V weights 在 engine 初始化时沿输出维打包，原三个 parameters 替换
    为空占位并释放；decode 改为一次 linear 后 split，持久权重字节不增加，每个 48-layer token
    减少 96 次 GEMM launch。CPU full-prefill 与多步 decode reference 继续覆盖数值一致性。
31. vLLM backend 的 Target input/post/final norms 全部接入 vLLM fused RMSNorm，替代 HF 每次
    norm 的 cast/pow/mean/rsqrt/multiply/cast kernel chain；torch reference 保留。CPU fake
    resolver 验证两层 prefill+decode 共 10 次均走统一入口，CUDA opt-in 覆盖 3-D decode shape。
32. Target hidden state 重排为等价 residual stream：attention output 在 post norm 合并 residual，
    MoE output 在下一 input norm（末层为 final norm）合并。vLLM backend 因此把每个 48-layer
    token 的 96 次 residual add 与对应 RMSNorm 融为 96 kernels；torch 路径显式执行相同数据流。
33. vLLM backend 的 Q/K 改为 token-major 下每层一次原位 fused RoPE，再 transpose 到 attention
    layout；参数严格沿用 Qwen full head dim、NeoX rotate、theta/scaling/max position，cache 在
    engine 初始化时上设备。torch 保留 HF 路径，CUDA opt-in 对 511/1023/4095/4096 位置与
    vLLM native reference 对照。
34. Attention `q_norm`/`k_norm` 也统一接入 fused RMSNorm，四维 `[B,T,H,D]` 保持 last-dim
    归一化语义；每层再消除两组 HF cast/pow/mean/rsqrt/multiply/cast 链，CUDA opt-in RMSNorm
    测试相应覆盖四维 Q layout。
35. Resident KV 在 vLLM backend 下接入 FlashAttention `flash_attn_with_kvcache`：一次 kernel
    原位 append 当前 K/V 并完成 native GQA decode attention，替代两个 `copy_` kernels + SDPA；
    torch fallback 保持原路径，JSON 记录 resolved `flash_kvcache`/`sdpa`，CUDA opt-in 对照 SDPA。
36. Transfer worker 将 demand batch 与 speculative microbatch 分离：真实 miss 继续使用最多 32
    objects 的批量 H2D，speculative 使用独立 72 MiB byte cap。Qwen expert 为 9 MiB，因此单个
    不可抢占预测批约 8 experts；128 KiB KV chunks 仍可填满 32 objects，不再被统一 8-object
    限制压低吞吐。减少 head-of-line wait 的同时保留小对象批量效率。
37. `queue.set_step` 从 Target 线程同步 heapify 改为 O(1) 发布 step/dirty flag；transfer worker
    下一次 pop 前才按最新 step 线性重建。DMA 期间跨过多个 layers 时只重建一次，priority
    仍以实际 pop 时的 current step 精确计算，heap 工作移出 Target compute 关键路径。
    1,000 queued resources 下 100,000 次连续 step 发布为 0.648 us/call；旧同步 heapify 基准
    为 0.681 ms/call，Target 侧约三个数量级下降，惰性 rebuild 由 worker 承担并可合并。
38. Release 形状 expert prediction admission 的 CPU profile 为 23.10 ms/refresh，其中 preload
    命中仍每窗口构造约 1,384 个等值 `ResourceKey`。ExpertRegistry 现缓存并返回每个
    `(layer, expert)` 的 canonical key，消除刷新期身份对象构造和初始组合 hash。相同完整
    admission/cancel 微基准降至 19.45 ms/refresh，约减少 15.8%。
39. Residency speculative lease aggregate 改为增量维护：add/overwrite 对 priority sum 应用
    delta，cancel 直接减去贡献；仅覆盖/删除当前最早 lease 时重算 min deadline。常见单-consumer
    expert 不再在 admission 与 cancellation 两侧各扫描一次 lease dict。相同 release-shape
    admission/cancel 微基准由 19.45 降至 17.25 ms/refresh，再减少约 11.3%。
40. Prediction cancellation 的 depleted queued resources 改由一次 `unqueue_many` 完成全部
    QUEUED→CPU_ONLY、lease aggregate 清零与单次 notify；旧路径对约 1,384 resources 分别
    获取 residency 锁并重算空 lease。Worker discard-close 同样复用批量转换。固定预构造
    requests、仅切换 scalar/batch unqueue 的受控微基准为 14.744→13.850 ms/window，约降 6.1%。
41. `prefetch_many` 不再把已有 PrefetchRequest window 复制成同规模五元 tuple list；residency
    通过只读 Protocol 直接消费原 intents，再仅为确需 queue 的资源构造 QueueUpdate。
42. Release/continuous runner 显式标记 causally shifted prediction window；无 budget 且全层
    admission 时保留上一 token H2/H3/... 与新 H1/H2/... 完全相同的 resource+absolute-consumer
    leases，只 enqueue 新 tail horizon。Tiny 2-horizon trace 首步 12 candidates，次步由重复 12
    降为仅新增 6；budget/layer-window/generic 调用仍保守 reset。
43. Shifted lease reuse 现在在 request 构造前按 layer/consumer bucket 过滤：已有 H1 rows 不再
    重做 expert Top-K、registry ensure、KV lookup 或 PrefetchRequest 构造；continuous 新准入时
    只处理新增 rows。Tiny 两层第二步 registry ensure 由两 horizons 的 8 次降为 tail 的 4 次。
44. Expert probe 的 `(x-x_mean)/x_scale @ W + y_mean` 在参数缓存时折叠为 `x@W'+b'`，其中
    `W'=W/x_scale`、`b'=y_mean-(x_mean/x_scale)@W`。每次 rollout 不再生成 standardized
    feature tensor，仅保留 batched GEMM、bias 与 sigmoid，并与逐层原公式做 1e-6 对照。
    相同 48 layers、16 samples、1024→128 微基准由 45.43 降至 27.69 ms，约减少 39.0%。
45. Draft attention/features 的 signal snapshot 改为先按模型 dtype D2H，再在 CPU promote FP32；
    release BF16 下 PCIe payload 减半，aggregation/probe 数值仍与原 BF16→FP32 完全一致。
46. Expert probe 改为传递 raw scores，admission 先 Top-K 再只对入选值 sigmoid；sigmoid 单调性
    保证 IDs 与先全量 sigmoid 完全一致。48×4 horizons×batch4×128 的纯张量微基准为
    10.99→2.43 ms，且 selected probabilities 逐值完全相等。
47. Engine 的 expert intent builder 显式关闭未使用的 `(key,consumer)` compatibility 返回值，
    每个完整 H1 window 避免为 1,536 routes 额外构造 tuple/list entries；公共默认保持兼容。
    固定 raw-score route builder 微基准为 2.586→2.511 ms/window，约减少 2.9%。
48. 每个 Draft provider 新增固定形状 expert-feature buffer：复用 pinned BF16 D2H staging 与
    CPU FP32 probe workspace，避免每个 refresh 重分配；CUDA copy 使用 non-blocking 后单次
    stream sync。随 context 变化的 attention 不缓存，防止多 refresh staging 内存累积。
49. vLLM backend 在 request timing 前预创建全部 48 层 logical-to-slot expert maps，并将其计入
    base allocator measurement；decode 不再每层首次 `_fused` 时 lazy `torch.full(128)`。JSON
    记录初始化耗时与 before-timing 状态，torch backend 明确返回未启用。
50. Resident FlashAttention KV-cache kernel 新增独立 pre-timing warmup：dummy cache 使用正式
    batch/context/capacity、Q/KV heads、dtype 和单 decode-token shape，不修改请求 cache；结果
    记录 warmup seconds/enabled，避免首 decode token 承担 backend 初始化或 JIT。
51. Packed expert slot tensors 从 vLLM MoE warmup 中解耦，任何 backend/lazy/warmup 配置都在
    base memory measurement 前独立分配完整 cache；默认约 576 MiB（maximize 可更大）不再落入
    首次 timed demand。Preflight 对已分配 storage 不重复加 payload，JSON 记录阶段与状态。
52. Actual router Top-K IDs 的逐层 `.to(cpu)` 改为 executor 内复用 pinned host buffer；每层
    non-blocking D2H 后一次 compute-stream sync，只有 batch/Top-K/dtype 改变才重分配，steady
    batch 每 token 避免 48 次 route snapshot host allocation。
53. Expert logical-to-slot map 更新新增按槽容量预分配的 pinned host/device staging；同一
    transfer batch 的跨层更新使用不重叠切片，避免异步 H2D 时 host buffer 被提前覆盖，也避免
    steady-state 每批为 indices/slots 各创建一次临时 CUDA tensor。动态 fallback 只在实际更新数
    超过预备容量时扩容；release 配置按 `2 * expert_slots` 在计时前覆盖同批移除与新增上界。
54. Expert compute-use CUDA event 改为安全回收池：共享 event 以引用计数追踪所有关联 resident
    experts，只有引用归零、`Event.query()` 已完成且不再位于 transfer-stream waited 集合时才能
    重录。compute/worker 两线程以独立锁保护映射和池，transfer batch 同步后再解除 waited 标记；
    steady decode 不再无条件每层、每 token 构造新 event，同时保留淘汰前等待最新 compute use
    的正确性。
55. Actual Top-K route 的 demand consumer 发现从“每层一次 `unique` 加每个唯一 expert 一次
    `torch.where`”改为单次 `.tolist()` 后一遍构建 expert→request rows，并在普通与 capacity-split
    加载间复用。batch4/top8/128 experts 的 48 层 CPU 隔离微基准为 10.58→0.84 ms/token，约
    12.6x；排序后的 expert 加载顺序及重复 route 的 consumer 语义保持不变。
56. TransferWorker 对 accepted batch 的 CPU payload 读取和 GPU resident 发布分别合并为一次
    residency 临界区；完成前先验证整批均为 IN_FLIGHT，再原子更新 resident/LRU/layer counts 并
    只 notify 一次。32 resources×100 batches 的真实 ResidencyManager CPU 微基准由逐项路径
    8.19 ms 降至 2.46 ms，约 3.33x；标量 API 继续委托给批量实现。
57. Prediction admission 在 Top-K 与 PrefetchRequest 构造前，以本次调用和现有状态的
    `(layer, absolute consumer)` bucket 排除重复 items；预算后的 requests 因而天然唯一，不再
    对完整窗口约 1,536 intents 做第二轮 tuple/set/bucket-list membership 过滤。该后置过滤的
    release-shape CPU 隔离微基准为 0.587 ms/window，移除后候选数与 budget-dropped 指标仍按
    真正的新请求结算。
58. TransferWorker 不再为 demand MemoryRequest 构造随后必被 `begin_transfer(demand=True)`
    丢弃的 speculative consumer-leases dict；只有 speculative 请求才计算 probability、urgency
    与 MiB priority。32-resource demand batch 的隔离构造基准为 20.24→1.02 us，约 19.8x；
    48 层全 miss 的上界约减少 0.92 ms/token Python bookkeeping。
59. `demand_many` 复用一次构造的 key list，并以一次 membership pass 同时统计 hit/miss；仅当
    确有 CPU_ONLY/QUEUED miss 时才读取 queue step 和构造 demand QueueUpdate。全 resident batch
    在 worker error check 后直接返回，不再进入空 queue upsert 与 condition wait。32-expert
    `demand_many+release_many` 实测由 43.40 降至 36.39 us/layer，约减少 16.2%；半命中纯扫描
    隔离基准由 34.25 降至 30.05 us/batch，约减少 12.3%。
60. 仅作为不可变属性载体的 `PrefetchRequest`/`DemandRequest` 从 frozen dataclass 改为
    NamedTuple；所有字段名、位置构造、相等性和不可变性保持，residency 的 structural Protocol
    接口不变。完整 prediction window 的 1,536 个五字段 intent 构造微基准为 1.689→0.862 ms，
    约减少 49.0%，同时避免每个普通 dataclass 实例的属性 dict。
61. Raw-score Top-K 返回的新 values tensor 直接原位 sigmoid，不再为每层分配另一结果 tensor；
    48 层后处理微基准由 0.513 降至 0.478 ms/window，约减少 6.8%。尝试把 values/IDs 合并后只
    调一次 `.tolist()` 反而增至 0.681 ms（dtype conversion 与 cat 开销更大），因此保持两次
    独立转换，不采用该方案。
62. Demand miss 使用三字段 `DemandQueueUpdate` 和专用原子 `upsert_demands`：队列锁内采用最新
    logical step，不再读取 step 锁、创建无用 consumer probability/deadline dict 或聚合 expected
    uses；已有 speculative 项原位晋升。consumer cancellation 显式保留已晋升 demand，避免预测
    lease 撤销把同步等待的请求删掉。32 demands×100 batches 的构造+真实 queue 入队微基准由
    11.997 降至 6.551 ms，约减少 45.4%。
63. Residency 补入 `size_bytes` 后生成的七字段 `QueueUpdate` 也从 frozen dataclass 改为
    NamedTuple；字段名、默认 `demand=False`、位置构造和不可变性保持，queue merge/validation
    接口不变。完整 prediction window 的 1,536 个 QueueUpdate 构造微基准为 2.395→1.117 ms，
    约减少 53.4%，并移除普通 dataclass 属性 dict。
64. TransferWorker 的 residency admission 从每个 request 单独 `begin_transfer` 改为一次
    `begin_transfers` 临界区；锁内仍按队列顺序逐项执行相同 lease refresh、priority rejection、
    layer-balanced eviction 与 reserve 转换，但整批最多 notify 一次。32 resources×100 batches
    的真实 demand admission CPU 微基准由 8.787 降至 6.872 ms，约减少 21.8%；标量 API 复用
    同一个锁内实现。
65. Queue 新增 `pop_many_with_step`，在完成 heap rebuild/pop 的同一 condition 临界区返回对应
    logical-step 快照；TransferWorker 不再刚释放 queue lock 就再次获取它读取 step。公共
    `pop_many` 仍返回原 list API。32 demands×100 batches 的 pop 基准由独立 step 读取的
    4.115 降至 4.046 ms，约减少 1.7%；绝对 CPU 收益较小，但消除 compute/worker 争用窗口。
66. 满容量 demand batch 在单次 LRU 扫描中按 `(priority,-deadline)` 分组，并以 per-layer heap
    模拟动态 layer-count/demand-count/LRU tie-break，一次规划全部 victims；不足或异常输入回退
    原逐项路径。原型在 100 个随机状态上与逐次 victim 序列完全一致，32 victims 规划为
    785→100 us（约 7.84x）。500 次交替 32-expert 全 miss 的方法级实测中，admission
    403.96→192.01 us/batch，总 orchestration 690.98→508.34 us/batch，分别减少约 52.5% 和
    26.4%。
67. Demand admission 清空 consumer leases 后直接写入最终 aggregate（lease priority/deadline
    为零、resource priority 为 inf、deadline 为零），不再调用面向 speculative dict 的
    `_refresh_priority`；`protected` set 也只在实际需要逐项 victim scan 时构造。无淘汰的
    32-resource×100 batches admission 由 6.872 降至 5.178 ms，约减少 24.7%；交替满缓存
    全 miss orchestration 由 508.34 降至 447.87 us/batch，再减少约 11.9%。
68. Worker 利用 `pop_many` 不混合 demand/speculative 的不变量，让 demand batch 直接调用
    key-only `begin_demand_transfers`，不再为每个资源构造五字段通用 admission tuple、调用恒返
    inf 的 `MemoryRequest.priority` 或扫描 `all(demand)`；victim 预规划与标量 fallback 复用同一
    helper。32-resource×100 batches 无淘汰 admission 对照为 5.121→4.521 ms，约减少 11.7%。
    跨线程总 wall time 受共享主机调度抖动较大，本项不据此声明额外端到端比例。
69. Expert victim planner 的最大 layer-count 筛选改用生成器传给 `min`，不再为每个 victim
    分配临时 candidate-layer dict；KV 仍遍历全部 layer buckets。100 个随机状态逐项等价，
    96 residents/48 victims 的规划中位数由 160.35 降至 144.34 us，约减少 10.0%。
70. 每个 primary `(priority,-deadline)` group 进一步建立 layer heap；expert entry 以动态
    `(-layer_count, demand_count, LRU)` 排序，KV 的 balance 固定为零。每选一个 victim 只更新
    该层 entry，不再对所有层重复 max/min 扫描。200 个随机 expert 状态逐项等价；32/12/32、
    64/24/32、96/24/48、160/48/32（resident/layers/victims）四组规划分别快约
    2.13x、1.92x、1.98x、1.62x，96/24/48 为 158.20→80.00 us。
71. Demand victim plan 成功后，专用 admission 直接执行已保证容量的 QUEUED/CPU_ONLY→
    IN_FLIGHT 状态转换，不再逐项回到包含容量、speculative lease 和 rejection 分支的通用 helper；
    consumer-leases dict 改为原位 `clear`，避免每次全 miss 分配新空 dict。32 resources×100
    batches 无淘汰 admission 由 4.521 降至 3.378 ms，约减少 25.3%；plan 不完整时仍走原
    scalar fallback。
72. Victim planner 按 layer 收集候选时改为线性 `append`，每个完整 bucket 只做一次
    `heapify`，不再为每个 resident 执行 `heappush`。候选 tuple 含唯一 LRU 序号，因而最终
    弹出次序不变；400 个随机 expert/KV 状态与逐次 victim selection 完全一致。96 residents/
    48 victims 的中位规划时间由改前 82.16/81.45 us（统一/混合 primary rank）降至
    76.54/78.40 us，分别约减少 6.8%/3.7%。
73. Prefetch residency-to-queue handoff 复用原 `PrefetchRequest` 引用，并另传平行的
    `size_bytes` 列表；不再为每个已筛选 intent 分配字段几乎重复的 `QueueUpdate`。队列专用
    入口仍先完成整批概率、大小与已有资源一致性验证，再原子合并，因此错误不会留下部分新
    request；公共 `QueueUpdate/upsert_many` API 保持不变。1,536 intents 的 residency prepare
    由 2.705 降至 2.021 ms，约减少 25.3%；同进程旧/新完整 handoff A/B 在 32、256、1,536
    intents 三档分别为 0.1316→0.1168、1.0179→0.8989、6.0697→5.2451 ms，约减少
    11.3%、11.7%、13.6%。曾测试统一 primary-rank victim 特判；统一场景只再快约 2.7%，
    混合场景却由 78.40 恶化到 83.33 us（约 6.3%），已撤销且不应重试。
74. 专用 prefetch queue handoff 在批内 resource key 全唯一时，用增量 probability/deadline
    aggregate 更新替代每资源一次 `sum`/`min`；若任何 key 重复则自动保留批后统一重算路径，
    避免 shared-consumer 工作负载回退。测试覆盖已有多个 consumer 时最早 deadline 后移、
    新 consumer deadline 前移和 expected uses 等价。相对第 73 项版本，32、256、1,536 intents
    完整 handoff 又由 0.1168→0.1103、0.8989→0.8483、5.2451→4.9777 ms，再减少约
    5.6%、5.6%、5.1%。
75. Demand runtime 同样复用原 `DemandRequest` 引用与平行 size 列表，不再为每个 CPU_ONLY
    miss 分配 `DemandQueueUpdate`；单 miss 走无批量 dict/去重成本的标量入口，两个及以上 miss
    走批量入口。两条入口都保持整批 size 一致性检查、demand promotion、最大 miss cost 和已有
    speculative consumer map。32 个请求中 1/2/3/4/8/16/32 个 miss 的旧→新 handoff 分别为
    5.501→4.364、7.238→7.169、8.940→8.449、10.591→9.749、17.267→14.998、
    30.175→24.963、55.862→44.834 us；所有测点均不回退，单 miss 与全 miss 分别约减少
    20.7% 和 19.7%。256/1,536 个全 miss 的初始原型分别约减少 20.4%/21.5%。
76. Transfer worker 对全接收 demand batch 复用 queue 弹出的 request list 以及 admission 已构造
    的 key list，不再生成 accepted list 后又二次提取 keys；若罕见地部分拒绝，仍按 flags 同步
    筛选 request/key。Speculative 路径在同一循环内生成 accepted 与 keys 并保持 dropped 计数。
    隔离的 admission-result bookkeeping 在 1/8/32/256 个全接收 demand 上由
    0.467→0.175、0.739→0.311、1.690→0.813、10.709→5.644 us，约减少 47–63%；部分
    demand 拒绝路径因额外 fast-path 检查较慢，但它不是正常单 worker demand queue 路径，且
    仍保留正确回退语义。Speculative 全接收/部分拒绝均未见实质回退。
77. `prepare_demands` 保存原本即需构造的去重 key dict，并在常见的全唯一输入中直接以 resident
    value 数量返回 hit count；runtime 不再对全部 key 做第二次哈希 membership 扫描。重复 key
    输入仍回退原逐 request 计数，测试锁定两个相同 resident demand 记为两个 hits、但资源
    demand-count 只增加一次。1/8/32/128 个全 resident 唯一 key 的 prepare+hit-count 中位数由
    1.904→1.618、6.955→6.056、24.190→21.637、94.852→84.330 us，约减少 10–15%。
78. Transfer worker 让 residency 在一次锁内直接生成 backend 所需 `(key,cpu_value)` items，
    不再先分配 values list 再 zip；GPU copy 返回后以平行 keys/values 直接原子发布，不再为
    completion 生成第二个 tuple list。旧 `cpu_values`/`complete_transfers` API 保留兼容包装，
    新发布入口先验证批长且仍在任何状态修改前验证全部 records。1/8/32/256 项读取容器成本由
    1.016→0.620、1.759→1.321、4.280→3.759、31.431→28.662 us（约减少 9–39%）；同规模
    completion 输入容器成本由 0.395→0.259、0.612→0.335、1.522→0.652、10.677→3.502 us
    （约减少 34–67%）。这些是 worker bookkeeping 隔离值，不代表 GPU copy 时间。
79. Expert dependencies、guaranteed KV chunks 及 batch 跨请求 KV keys 都由上游 dict/唯一 chunk
    结构保证唯一；这些内部调用现在显式传递 `keys_are_unique=True`，使 demand prepare 和 expert
    release 跳过各自的 `dict.fromkeys`。公共 runtime/residency API 默认仍去重，重复请求的 hit
    metrics 与 demand-count 语义不变。1/8/32/128 个全 resident demand+release 的生产形态 A/B
    由 3.634→3.108、10.761→8.947、34.737→28.146、128.764→108.765 us，约减少
    14.5%、16.9%、19.0%、15.5%。
80. `wait_resident_many` 将“任一 CPU_ONLY 失败”和“全部 GPU_RESIDENT 成功”合并到一次终态
    predicate 扫描，成功唤醒后不再第三次验证全部 records；runtime 传入 prepare 已去重的
    pending keys 时也跳过重复 list/dict 构造。默认公共路径仍去重，测试覆盖成功、重复输入、
    CPU_ONLY 立即失败和 IN_FLIGHT 超时。1/8/32/128 项已完成等待由 2.767→1.361、
    8.235→4.477、25.916→14.343、100.729→54.411 us，约减少 45–51%。
81. Queue 新增兼容的 metadata pop，将形成 batch 时本来就为 byte cap 累加的 `batch_bytes`
    传给 worker；同类批全部接收时 demand/speculative 都复用原 requests、keys 和 bytes，部分
    拒绝则在已有筛选循环同步累加 accepted bytes。完成后 `completed/bytes/transfer-kind` metrics
    各按批更新一次，测试锁定 demand 与 speculative 的 completed、bytes 和分类计数。隔离的
    admission-filter+completion-metrics 在 1/8/32/256 项全接收 demand/speculative 上分别约由
    0.47/1.31/4.04/29.5 us 降至 0.16/0.22/0.41/2.16 us；8/32/256 项部分拒绝 speculative
    也约减少 26%/44%/51%。单项全拒绝只多约 0.02 us，且不会进入 transfer/completion 路径。
82. Batch pop 已通过 `_peek_valid_locked` 得到并验证下一个合法 heap head 后，直接用专用 helper
    移除该项，不再调用会重复 peek 的 `_pop_valid_locked`；queue condition 锁在整个操作期间持有，
    因此中间不会有并发状态变化。测试锁定两项 batch 只调用两次 peek。32,000 requests/
    1,000 个 32-item batches 的同进程旧/新 drain 中位数由 47.37 降至 42.21 ms，约减少
    10.9%；profile 中合法 head 检查次数由 63,001 降至 32,001。
83. 已验证 head 的 `heappop`/request 删除进一步在两个锁内调用点直接内联，移除每个 batch
    后续元素一次 Python helper 调用。相同 32,000 requests/1,000 batches 的当前 helper 与
    inline A/B 由 41.19 降至 39.54 ms，再减少约 4.0%。另测试过以普通 dict+单调 epoch 替代
    OrderedDict LRU：尽管 scan 较快，32/128 项触碰比 `move_to_end` 慢约 20–25%，已否决，
    不应牺牲常见 resident-hit 路径。
84. GitHub Actions 从首次加入 Qwen3 adapter 后持续失败的根因是依赖无上限：CI 已解析到
    Transformers 5.17/Torch 2.14，而 runtime 依赖 Transformers 4.51 的逐 expert module 与
    `DynamicCache.batch_split` API。隔离复现中 5.17 为 31 failed，最新 4.x 4.57.6 仍有 3 个
    cache API failures；4.51.3 + Torch 2.8.0 + pytest 9.1.1 为 162 passed、9 skipped。因此
    `pyproject.toml` 固定兼容窗口为 Transformers `>=4.51,<4.52`、Torch `>=2.4,<2.9`，
    并将数值栈约束到已验证的 NumPy `>=1.26,<2`。
    Workflow 同时升级到 Node 24 的 checkout/setup-python v7，并只在 main push 或 PR 运行，
    避免 feature 分支同一提交产生 push/PR 两个重复任务；concurrency 会取消同 ref 的旧任务。
85. Demand worker 从 queue 的唯一资源映射弹出 batch，因此其 key 已由结构保证唯一；该不变量
    现在显式传给 residency victim planner，使 planner 按 kind 直接累计 eligible 数量，不再为
    每批构造一组相同 key 的 set。公共 `begin_demand_transfers` 默认仍去重，重复 key 的顺序返回
    语义保持 `[True, False]`。32-key 两组交替填满 32-slot expert cache、3,000 batches、9 轮的
    同进程旧/新完整 admission+publish+release 中位数为 118.783→113.412 us/batch，约减少 4.5%；
    这是 CPU orchestration 隔离值，不代表 GPU H2D 或 decode throughput。
86. Prefetch queue handoff 已在首轮 size/atomicity 检查中得到批内唯一 key 数；唯一-key 常见路径
    现在于 merge 后直接 push，不再把每个 request 再插入临时 `unique` dict 后二次遍历。共享-key
    batch 仍按每个唯一资源统一重算 aggregate 并只 push 一次，已有多 consumer 与原子失败测试
    保持覆盖。预先创建空 queue、仅计 `upsert_prefetches` 的同进程旧/新 A/B 在 32、256、1,536
    intents 分别为 142.118→122.201、1136.781→951.378、6771.422→6032.590 us，约减少
    14.0%、16.3%、10.9%；完整 fresh admission 的独立采样也全部改善。
87. Worker speculative admission 对 QUEUED record 合并 residency 与 queue consumer leases 后，
    已拥有私有 lease dict 及其 priority/deadline 聚合值；状态转为 IN_FLIGHT 时现在直接发布这组
    结果，不再复制 dict 并通过 `_refresh_priority` 第二次执行相同 sum/min。CPU_ONLY 兼容入口
    仍防御性复制调用方 dict。每项 4 consumers 的 1/8/32-resource admission 中位数由
    5.756/36.985/149.923 降至 4.307/25.280/102.409 us，约减少 25.2%/31.6%/31.7%；
    回归测试同时锁定 aggregate 数值、dict 非别名以及 QUEUED 路径不再 refresh。
88. Speculative worker 仍在 residency 锁外重建 consumer lease urgency，但以平行的 requests 与
    lease-dict lists 调用专用 batch 入口，不再为每个资源构造五字段 admission tuple、调用常见
    QUEUED 路径不会使用的 `MemoryRequest.priority()`，或在通用入口扫描 `all(demand)`。专用入口
    对罕见 CPU_ONLY 状态仍按原 queue priority 公式做 eviction 决策；长度、fallback 及 worker
    路由均有回归覆盖。含 key/lease 构造与 admission 的 1/8/32-resource 中位数由
    6.581/42.134/163.662 降至 6.199/39.219/151.473 us，约减少 5.8%/6.9%/7.4%。曾原型化
    把 lease 重算整体移入 residency 锁，虽单线程略快但扩大临界区，已否决并撤销。
89. Sparse Target attention 的 hybrid stopping 已为 always/guaranteed/按需 old chunks 计算 QK
    logits；最终输出现在保留并拼接这些 logits，只执行一次全选中 token 的 softmax×V，不再拼接
    keys 后重复整段 QK。新 helper 校验每个 logits/value chunk 的 token、KV-head 与 head-dim 对齐，
    数值测试与原 grouped GQA 输出逐元素一致。单线程 CPU、32 Q heads/4 KV heads/head-dim 64、
    每 chunk 64 tokens 的 2/4/8 chunks 旧/新中位数为 177.462→139.752、323.413→262.886、
    656.269→532.884 us，约减少 21.3%/18.7%/18.8%。相同 tiny batch-4 的 20-step profile
    由 10.460 降至 8.306 s，QK einsum 次数由 2,720 降至 2,400；这些仍是 CPU 证据。
90. 已知必选的 sink、recent 与 guaranteed old KV chunks 进一步先拼接 key、执行一次 grouped
    QK，再按原 chunk width 切分 logits 供顺序 marginal stopping；自适应追加 chunk 仍逐个执行，
    因此停止次序和选集不变。单个已知 chunk 不做无意义拼接。真实形态 `(2,8)` tokens 以及再加
    1/2/3 个 64-token chunks，batched QK 分别约快 10.7%、21.3%、25.7%、29.3%；两个等长
    64-token chunks 的低并发反例慢约 2.5%，故另以 `69c2609` 临时 worktree 做同机并行完整
    tiny batch-4 A/B。7 轮×40 decode 的中位为 11.942→10.008 ms/token-step，约减少 16.2%，
    同路径 20-step profile 的 QK einsum 调用由 2,080 降至 320。GPU 净收益仍须实测。
91. Grouped GQA 的两处固定三维 contraction 从通用 `einsum` 改为显式 batched matmul：QK 使用
    `(kv_heads,groups,dim) @ (kv_heads,dim,tokens)`，PV 使用
    `(kv_heads,groups,tokens) @ (kv_heads,tokens,dim)`，布局仅用 view/permute。10/74/138/266/522
    tokens 的单线程完整 QK+softmax×V 中位由 41.286/45.400/52.765/66.228/97.660 降至
    18.776/21.946/28.314/41.598/70.976 us，约减少 27–55%，输出逐元素一致。同机并行、7 轮
    ×40 steps 的 tiny batch-4 A/B 相对 `a99aa51` 为 9.947→9.743 ms/step，约减少 2.0%。
    这些仍不是 release GPU workload 的吞吐证据。
92. Batched known-QK 的连续 `combined_logits` 只为 marginal 计算切成 views，最终 attention
    现在直接保留原 tensor；known values 也预先合并成同一对齐块。若没有自适应 tail，输出 helper
    的 singleton fast path 不再把这些 views `cat` 回一份相同 logits；有 tail 时仍只拼接一次。
    `(2,8)`、`(2,8,64,64)`、`(2,8,64,64,64,64)` token layouts 的输出阶段由
    20.735/29.434/42.481 降至 13.986/15.777/20.727 us，约减少 32.5%/46.4%/51.2%；
    同机并行 tiny batch-4 相对 `b01fe5f` 为 9.207→9.177 ms/step（约 0.3%）。

## Next actions

1. GPU 空闲后先运行 opt-in CUDA 测试，再用 c512 demand/speculative H1 验证 grouped
   GQA、跨请求 KV demand 和 layer lookahead 2，并建立相同显存约束的 vLLM decode-only
   baseline。
2. c512 通过后重跑 batch-4/context-4K 的长输出 demand/speculative，并在同一 10 GiB cap
   下测量 resident KV 候选，核对 token 因果一致、
   TPOT、最大单批候选、dropped speculative、H2D overlap 与 demand wait。
3. 用 CUDA profiler 分解 Target kernel、Draft、Python 调度、queue/residency、同步和 H2D；
   对主导路径实施架构优化，必要时迁移到 C++/CUDA、融合 kernel 或 CUDA Graph，而不是
   继续堆叠 Python 微优化。
4. 若滚动窗口仍提交过量，再扫描 expert/KV 唯一资源预算；只保留降低 decode TPOT 的配置，
   未经 GPU 验证不设为默认。
5. 更新本文档并执行最终审计；所有正确性与 decode 性能门禁结算后再把 feature 分支合并到
   `main`。每个预先声明的主工作负载都必须达到 >=1.50x vLLM decode throughput；不能以
   单一有利 workload、微基准、Target-kernel 子计时或非匹配口径替代发布门禁。

## 当前 CPU 检查点边界

- 本检查点在四张 GPU 均被外部作业长期占用时提交，目的是先固化已通过 87 项 CPU
  回归的实现与审计结果；它不是性能发布版本，也不改变 `main`。
- 新增滚动 deadline 准入、过期 consumer 撤销、窗口内唯一资源预算、跨请求 KV demand、
  单次 Target marginal 同步和 grouped GQA 均只有 CPU 数值/状态机证据，不能据此宣称
  CUDA 正确性或速度提升。`speculative_layer_lookahead` 和两类预算默认未设置。
- 两类预算是每次滚动增量提交内的唯一资源上限，不是整个队列或完整窗口的全局 cap；
  是否需要全局背压必须由后续 GPU profile 决定，当前文档不作该声明。
- `runtime-batched-demand-vllm-kvbatch-b4-c4096-o17.json` 的 451.895 s 结果来自上述
  grouped GQA/跨请求合批改动之前，只证明上一轮 guaranteed-prefix KV demand batching；
  它不能替代本检查点的待跑 GPU 门禁。
- 旧 JSON 的 `request_seconds`、端到端 throughput 和首次 rollout 归属仍按旧 runner 定义；
  它们作为历史记录保留，但不满足新的 decode-only 发布口径。
- decode-only runner 改造提交 `e46565f` 已通过 89 项 CPU 回归（另 2 项 CUDA 跳过）和
  ruff；这只验证计时状态机与字段口径，尚未产生 GPU 性能证据。

### CPU 热路径微基准

驻留满缓存路径已把候选选择从构造 `order` 字典、候选列表并扫描两次，改为单遍维护
最小 rank，且准入检查与实际淘汰复用同一 victim。CPython 3.12、512 个 resident、每轮
2,000 次、7 轮中位数：旧参考路径 1681.729 us/次，新路径 348.423 us/次，即 4.83x；
这是 CPU bookkeeping 微基准，不代表端到端或 GPU 加速。测试同时锁定 priority、deadline、
LRU tie-break、pinned/protected 排除、低优先级拒绝，以及每次满缓存准入只选择一次。
可用 `PYTHONPATH=. python benchmarks/runtime_cpu_microbench.py --iterations 2000 --repeat 7`
复测。

同一资源的批量 upsert 现在先写入全部 consumer 映射，再按批内唯一资源各重算一次
deadline 并压入一个堆项；重复 consumer 仍以最后一次 probability/deadline 为准，
`miss_cost_ms` 仍取最大值，`demand` 仍作 OR 合并。CPython 3.12、重复更新同一资源的
2,000 个 consumers、每轮 20 次、7 轮中位数：旧的逐 consumer `min` 路径 53.966 ms/次，
新的逐资源 `min` 路径 2.431 ms/次，即 22.19x。这同样只是 CPU bookkeeping 微基准，
不代表端到端或 GPU 加速；可用同一脚本的 `--queue-consumers` 和 `--queue-iterations`
参数调整规模。等价性测试覆盖重复 consumer、deadline、miss cost、demand 和每唯一
资源只生成一个堆项。

TransferWorker 现在在每次 `pop_many` 成功后快照一次 logical step，同一批内所有
consumer lease 和资源 priority 共用该值，避免在每个 consumer 和每个资源上重复
获取 queue condition 锁。CPython 3.12、100,000 次逻辑读取、每轮 20 次、7 轮中位数：
旧的重复加锁路径 43.409 ms/次，新的单次快照 0.432 us/次，即 100,520x。该比较只
隔离逻辑时间读取开销，不包含批内其余 worker 工作，也不是端到端收益。测试使用
三资源、六 consumer 的实际 worker batch 锁定每批恰好一次 `current_step` 读取；
微基准可用 `--step-accesses` 和 `--step-iterations` 调整。

## Safety and versioning

All GPU entrypoints refuse to start on a selected busy physical GPU. CUDA tests require
`SPECFETCH_RUN_CUDA_TESTS=1`. Work was split into small pushed commits; no external GPU
jobs were killed or modified. Two interrupted vLLM processes created by this task were
explicitly terminated after their EngineCore children became zombies.
