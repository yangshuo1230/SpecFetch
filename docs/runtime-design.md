# Sparse offload runtime design

The runtime separates model compute from memory movement. Compute submits resource
intent and waits only on a missing dependency; one transfer worker owns the H2D stream.

## Optimization scope and performance architecture

当前优化阶段只研究 decode。模型加载、kernel warmup、Target/Draft prefix prefill 和
prefix KV cache 构造不进入主性能门禁；prefill 可以使用独立、直接且高效的实现，不要求
经过本项目的 speculative queue、稀疏 KV 选择或专家 offload 调度路径。prefill 的职责是
为两种策略提供数值一致的起始 token 和 cache state，而不是验证本项目的核心假设。
Serving prefill projects only the final hidden row through the vocabulary head; callers
that need a full-sequence numerical reference opt in explicitly. For batch 4/context 4K
and a 152K BF16 vocabulary, this avoids an otherwise discarded roughly 4.64 GiB logits
tensor and keeps the complete request inside the matched memory budget.
Draft initialization and multi-token cache advancement likewise request only one logits
row through Transformers' native `logits_to_keep=1` path. Draft rollout already advances
one token at a time, and uses the same explicit bound for consistency.

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

Each bias-free Target attention layer packs Q/K/V weights along the output dimension at
engine construction, replaces the three original parameters with empty placeholders, and
uses one linear projection followed by tensor splits. Persistent weight bytes do not grow,
while decode removes two GEMM launches per layer (96 launches per 48-layer token).
The vLLM backend also routes all input/post-attention/final normalization through vLLM's
fused RMSNorm op. This replaces each Transformers FP32 cast, square, reduction, rsqrt,
multiply, and cast chain with one kernel; the torch backend retains the reference module.
Per-head Q/K normalization uses the same fused entry point on its four-dimensional
tensor, removing two additional multi-op normalization chains per layer.
Hidden state uses the standard residual-stream form: attention output is fused with its
residual at post-attention norm, and MoE output is fused at the next input norm (or final
norm). Thus vLLM combines 96 residual additions with their norms per 48-layer token;
torch executes the same deferred-add dataflow explicitly for numerical reference.
For the vLLM backend, token-major normalized Q/K tensors use one in-place fused rotary
kernel per layer before transposing to attention layout. Its cache is initialized on the
target device with Qwen's full head dimension, NeoX half rotation, theta, scaling, and
maximum position; the torch backend retains HF rotary embeddings.

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
Demand and speculative transfer batching therefore use separate limits. Actual missing
experts retain the full transfer batch (32 by default), while speculative work uses an
independent 72 MiB byte cap plus the 32-object ceiling. This yields about eight 9 MiB
Qwen3 experts per non-preemptible batch while still allowing 32 small KV chunks; both
limits remain configurable.

Actual demand batches inspect hits, promote misses, mark active dependencies, and acquire
completed GPU payloads with one residency critical section before and after the wait.
This replaces per-resource state/mark/get/wait lock round trips without changing demand
priority or the single batched queue upsert.
Each record caches aggregate speculative lease priority/deadline whenever leases change.
Entering actual demand switches priority to infinity; batched release restores the cached
values, avoiding a sum/min scan of unchanged consumer leases twice per expert and layer.
Speculative residency preparation consumes the existing immutable prefetch intents
directly; it no longer duplicates every window into an intermediate five-field tuple list
before producing queue updates.
Lease additions and overwrites update the cached probability sum by delta; cancellation
subtracts the removed contribution. The deadline is rescanned only when an overwritten or
removed lease owned the current minimum, rather than rescanning every touched resource.
Immutable resource identities cache their composite hash at construction. Queue,
residency, lease, and transfer dictionaries reuse that integer instead of repeatedly
hashing kind/layer/object/request fields on every decode lookup.
Logical time still recomputes urgency exactly at every Target layer, but rebuilds the
queue by constructing one dense heap array and applying linear-time `heapify`. It no
longer performs one logarithmic `heappush` and version update per queued resource.
`set_step` itself only publishes the new logical time and marks the heap dirty. The
transfer worker performs the linear heapify immediately before its next pop, so Target
compute does O(1) work and multiple layer advances while DMA is busy coalesce into one
latest-step rebuild.
Each queued request caches the aggregate predicted-use probability when consumers are
upserted or cancelled, so urgency-only step rebuilds do not rescan consumer dictionaries.
All experts consumed by one MoE invocation are released with one residency critical
section. CUDA packed slots share one compute-stream completion event for that invocation,
rather than allocating and recording an equivalent event for every expert.
CPU expert storage packs equal-shaped gate/up matrices into adjacent views matching the
GPU fused `gate_up` slot. H2D therefore submits one gate/up copy plus one down copy per
expert, instead of three tensor copies, without duplicating persistent CPU weights.
Expert evictions update host slot ownership immediately but batch device-map removals with
the next transfer's assignments into one indexed update per layer. Experts protected by
the same compute completion event also enqueue only one transfer-stream wait for that
event, removing per-expert scalar map kernels and duplicate event waits.
All logical-to-physical expert maps are materialized for every Target layer during engine
initialization and included in the base memory measurement. Speculative updates therefore
target persistent tensors immediately, and timed decode never lazily allocates a layer map.

同一预测窗口产生的 expert 与 KV 请求会先跨层、跨请求汇总，再通过一次队列事务完成
upsert 和唤醒；这只合并提交开销，不改变每个对象的 probability、deadline 或最终堆顺序。
预测窗口失效时，consumer cancellation 同样按资源批量合并，并在队列和驻留管理器中各
只获取一次锁；共享资源仅撤销对应 consumer，其余请求的 lease 和优先级继续保留。
Resources whose final queued consumer is cancelled transition back to CPU-only through
one `unqueue_many` residency section with a single notification; the transition no longer
reacquires the lock and recomputes empty lease aggregates once per depleted resource.
Active prediction leases are bucketed by `(layer, absolute-token consumer)`. Layer
retirement directly removes only the current request buckets rather than filtering the
entire future-layer/horizon lease list at every layer; request completion removes its
owner buckets and cancels them as one batch.
Serving runners identify their causally shifted prediction windows. With unbudgeted
full-layer admission, the prior H2/H3/... leases are exactly the next H1/H2/... leases,
so admission retains matching `(resource, absolute consumer)` pairs and submits only the
new tail horizon. Budgeted, layer-windowed, and generic decode calls conservatively reset.
Existing layer/consumer buckets are filtered before expert Top-K and KV request creation,
so a shifted survivor row does not even rebuild intents that would later be deduplicated;
mixed continuous batches construct only newly admitted request rows plus the tail horizon.

Target router 的 GPU Top-K route IDs 每层只复制为一个很小的 CPU snapshot。unique expert
发现和 request-consumer 构造使用该 snapshot，避免为每个实际 expert 分别执行 GPU
`where` 和逐标量同步；原始 GPU selected/routing tensor 仍直接进入 fused MoE。
Preloaded expert registry entries use a read-only lock-free lookup. Within one predicted
layer/horizon, overlapping routes across requests also share one registry resolution.
The lookup returns a canonical cached `ResourceKey` per `(layer, expert)` rather than
constructing an equal identity object on every prediction refresh or actual route.
The vLLM MoE backend also uses its fused router kernel, combining FP32 softmax, Top-K,
and Top-K renormalization instead of launching each PyTorch operation separately. The
torch backend retains the explicit reference path.
For equal prediction priority and deadline, expert eviction is layer-balanced before its
LRU tie-break: entries are removed from the most represented layer. This avoids global
LRU's zero-hit cyclic-scan failure when every decode token revisits all model layers but
the expert working set is larger than the cache. KV eviction remains priority/deadline/LRU.
Within the selected layer, lower observed actual-demand frequency is evicted before LRU.
Only true demand increments this frequency, so unused speculative arrivals cannot displace
stable cross-token expert routes merely by being recent.

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
With the vLLM backend, FlashAttention's contiguous-KV decode kernel replaces the fallback:
it appends the new K/V rows in place and performs GQA attention in one launch per group
and layer. The result records `flash_kvcache` versus `sdpa` explicitly.
Before request timing, the release runner invokes that kernel on an isolated dummy cache
with the declared batch, context, capacity, Q/KV head counts, dtype, and decode-token
shape. This warms backend/JIT state without mutating real prefix KV and records its cost.
The Draft rollout independently disables attention outputs and CPU attention aggregation
for resident Target KV, while retaining hidden-state output when expert probes need it.
Prediction admission likewise bypasses per-request KV retain/prefetch bookkeeping in this
mode; expert prediction requests still use the unified queue and normal consumer leases.
Across all modes, mapped attention rows are transferred once per horizon and expert
features once per complete rollout, rather than one D2H synchronization per draft layer.
Draft signals cross D2H in their compact model dtype (BF16 for release runs) and promote
to FP32 only on CPU for aggregation/probes, halving signal-transfer bytes versus GPU-side
promotion before the copy.
The fixed-shape expert feature snapshot reuses one pinned compact staging buffer and one
CPU FP32 probe workspace per provider. Variable-length attention snapshots deliberately
remain uncached so growing context lengths cannot accumulate staging allocations.
All target-layer expert probes evaluate every horizon with one cached-parameter batched
CPU GEMM. Predicted routes use one matrix Top-K per target layer rather than one call per
request row.
Probe standardization is folded into cached linear parameters:
`W'=W/x_scale` and `b'=y_mean-(x_mean/x_scale)W`. Refresh therefore performs only the
batched GEMM plus bias and sigmoid, without materializing normalized feature tensors.
The provider carries raw probe scores into prediction admission. Because sigmoid is
strictly monotonic, admission selects Top-K on those scores and applies sigmoid only to
the selected values used as queue probabilities, rather than all 128 experts.
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
Packed expert slots are budgeted the same way from slot count, hidden/expert widths, and
dtype. The allocator cap is installed before model loading and warmup; pre-warmup and
post-warmup lower bounds account for expert slots even when lazy initialization has not
materialized them yet. This makes larger cache configurations reject safely before work.
The opt-in cache maximizer fills the remaining matched cap with whole expert slots after
an explicit workspace reserve, capped at the model's complete expert set. It records the
resolved slot count; the fixed 64-slot default remains until matched GPU validation chooses
a safe reserve for both release contexts.
