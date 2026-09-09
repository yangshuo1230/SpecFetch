# Sparse offload runtime design

The runtime separates model compute from memory movement. Compute submits resource
intent and waits only on a missing dependency; one transfer worker owns the H2D stream.

## Optimization scope and performance architecture

当前优化阶段只研究 decode。模型加载、kernel warmup、Target/Draft prefix prefill 和
prefix KV cache 构造不进入主性能门禁；prefill 可以使用独立、直接且高效的实现，不要求
经过本项目的 speculative queue、稀疏 KV 选择或专家 offload 调度路径。prefill 的职责是
为两种策略提供数值一致的起始 token 和 cache state，而不是验证本项目的核心假设。

Decode 计时从 Target 和 Draft prefix cache 均准备完成之后、首次 speculative rollout
之前开始。首次 rollout、后续 Draft refresh、预测准入与撤销、H2D、demand wait 和 Target
计算均是产生后续 token 所必需的 decode 工作，必须计入 wall time，不能移入 prefill。
主指标是固定长度 steady-state decode 的 TPOT、decode tokens/s 和 batch wall time；传输量、
命中率、queue/admission 时间与 kernel 时间用于解释结果。

Python 实现负责语义参考、因果测试和快速迭代，不被视为最终性能架构。如果 profile 表明
Python 调度、逐层同步、queue/residency bookkeeping 或 kernel launch 位于关键路径，性能
实现应下沉到 C++/CUDA，或采用 fused kernel、CUDA Graph 和设备侧调度。最终硬门禁是在
相同模型、batch/context/output、精度与 GPU 显存约束下，使 decode tokens/s 至少达到
vLLM CPU-weight-offload decode 的 1.5 倍；对固定生成 token 数，这等价于 batch decode
wall time 和 TPOT 不高于 vLLM 的 2/3。vLLM full-resident 仅作硬件上界，不是本阶段
主门禁。

## Resource lifecycle

~~~text
CPU_ONLY -> QUEUED -> IN_FLIGHT -> GPU_RESIDENT
    ^                                  |
    +--------------- EVICTED ----------+
~~~

In sparse mode, the authoritative copy of every old KV chunk and routed expert remains
in pinned CPU memory. Sink and recent KV are pinned in the GPU residency manager. Other
GPU objects are evictable. The decode runner also exposes an opt-in `resident` KV mode:
it preallocates a batch-contiguous cache for each uniform-length prefill group and layer
through the declared output bound, then runs one batched GQA attention per group/layer.
Completed rows are compacted into a smaller allocation, while later admissions with a
different length retain an independent group. This is an architecture experiment for
contexts whose complete KV state fits under the same measured memory cap as the baseline;
it does not weaken the expert-offload constraint or the matched-memory release gate.

## Unified queue

KV chunks and experts share one logical priority queue. A speculative request is scored
by expected avoided stall per transferred MiB, including probability, deadline urgency,
and batch reuse. An actual compute miss is marked as demand and sorts before every
speculative request.

An upsert on an existing resource merges consumers and refreshes its probability and
deadline. Heap entries carry versions, so old priorities are discarded after an update.
An in-flight DMA is not preempted.

同一预测窗口产生的 expert 与 KV 请求会先跨层、跨请求汇总，再通过一次队列事务完成
upsert 和唤醒；这只合并提交开销，不改变每个对象的 probability、deadline 或最终堆顺序。
预测窗口失效时，consumer cancellation 同样按资源批量合并，并在队列和驻留管理器中各
只获取一次锁；共享资源仅撤销对应 consumer，其余请求的 lease 和优先级继续保留。

Target router 的 GPU Top-K route IDs 每层只复制为一个很小的 CPU snapshot。unique expert
发现和 request-consumer 构造使用该 snapshot，避免为每个实际 expert 分别执行 GPU
`where` 和逐标量同步；原始 GPU selected/routing tensor 仍直接进入 fused MoE。
Preloaded expert registry entries use a read-only lock-free lookup. Within one predicted
layer/horizon, overlapping routes across requests also share one registry resolution.
The vLLM MoE backend also uses its fused router kernel, combining FP32 softmax, Top-K,
and Top-K renormalization instead of launching each PyTorch operation separately. The
torch backend retains the explicit reference path.

## Sparse KV stopping

Old KV chunks are visited in draft-attention order. The online controller stops only
after all three conditions hold:

1. cumulative draft-predicted mass reaches `predicted_mass_threshold`;
2. the target marginal softmax-partition contribution stays below
   `marginal_mass_threshold` for `marginal_patience` chunks;
3. at least `minimum_old_chunks` have been consumed.

The target marginal is computable from logits for arrived chunks. True mass relative to
all old KV is unavailable online and is calculated only in evaluation with a full-KV
shadow pass.

Default GPU residency is four sink tokens, 256 recent tokens per request, 64-token old
KV chunks, 64 expert slots, and 512 old-KV slots.

Resident KV mode bypasses KV queue traffic and hybrid stopping. Each decode append is an
in-place write into the preallocated cache and attention uses PyTorch SDPA with native
GQA, avoiding old-chunk concatenation and per-chunk target-marginal synchronization.
The Draft rollout independently disables attention outputs and CPU attention aggregation
for resident Target KV, while retaining hidden-state output when expert probes need it.
Prediction admission likewise bypasses per-request KV retain/prefetch bookkeeping in this
mode; expert prediction requests still use the unified queue and normal consumer leases.
Across all modes, mapped attention rows are transferred once per horizon and expert
features once per complete rollout, rather than one D2H synchronization per draft layer.
Each expert probe evaluates every horizon as one CPU batch.
When resident KV is paired with demand-only expert loading (or no expert probes), decode
has no consumer for any Draft signal. The release runner therefore does not load or run
the Draft model for that policy and records the resolved signal/prefetch state explicitly.
The sparse mode remains the default until matched GPU measurements establish which mode
wins at each release context.

The release runner computes the exact resident-buffer payload from every layer's K/V
projection width, dtype, batch size, and declared token capacity before prefill. It
rejects a run when persistent allocations plus that payload already exceed the matched
cap, and applies the same absolute cap to PyTorch's CUDA caching allocator so temporary
workspace cannot silently oversubscribe it. Results record the initial allocator state,
resident payload, conservative allocated-memory lower bound, and measured peak.
