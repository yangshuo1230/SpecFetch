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

## Next actions

1. GPU 空闲后先运行三项 opt-in CUDA 测试，再用 c512 demand/speculative H1 验证 grouped
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
