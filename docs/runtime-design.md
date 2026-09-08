# Sparse offload runtime design

The runtime separates model compute from memory movement. Compute submits resource
intent and waits only on a missing dependency; one transfer worker owns the H2D stream.

## Resource lifecycle

~~~text
CPU_ONLY -> QUEUED -> IN_FLIGHT -> GPU_RESIDENT
    ^                                  |
    +--------------- EVICTED ----------+
~~~

The authoritative copy of every old KV chunk and routed expert remains in pinned CPU
memory. Sink and recent KV are pinned in the GPU residency manager. Other GPU objects
are evictable.

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
