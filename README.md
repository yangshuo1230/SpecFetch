# Speculative Offload Prefetch Study

This repository tests whether signals already produced by speculative decoding can prefetch
offloaded memory for Qwen/Qwen3-30B-A3B. The draft model is Qwen/Qwen3-0.6B.

## Sparse offload inference runtime

The `feature/sparse-offload-runtime` implementation is a runnable Qwen3-MoE reference engine, not
only a trace simulator. It keeps four sink tokens and a 256-token recent window on GPU for each
request/layer. Older 64-token KV chunks and every routed expert have authoritative CPU copies and
enter GPU only through one decoupled memory-request queue.

The queue supports speculative probability/deadline priority, cross-request expert reuse, demand
promotion, duplicate upsert and reprioritization. GPU residency uses priority leases so a remote
low-value prefetch cannot evict a nearer dependency. The sparse attention stopping rule combines
95% cumulative draft mass, two consecutive target marginal contributions below 1%, and a minimum
of two old chunks.

Core modules are deliberately framework-neutral:

| Module | Responsibility |
| --- | --- |
| `memory_queue.py` | Mutable KV/expert priority queue and demand promotion |
| `residency.py` | Resource state machine, finite GPU capacity and eviction |
| `transfer.py` | Dedicated worker and CUDA H2D stream |
| `kv_cache.py` | Sink/recent layout, CPU old chunks and sparse retrieval |
| `hybrid_attention.py` | Online target marginal mass and sparse attention |
| `expert.py` | Original checkpoint expert source and exact Top-8 MoE |
| `predictor.py` | Incremental, rollback-safe four-token draft rollout |
| `qwen3_engine.py` | Transformers Qwen3 projections/router adapter |
| `continuous_engine.py` | Variable-length admission, completion and backfill loop |
| `model_loader.py` | Meta loader that never materializes experts on GPU |

Train the expert probes and run the two causally matched policies:

~~~bash
python -m scripts.run_offload_prefetch \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --prompts prompts.json \
  --output results/runtime-signal-training.json \
  --probe-output results/expert-probes.pt

python -m scripts.run_runtime_engine \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --probes results/expert-probes.pt \
  --prompts prompts.json \
  --context-tokens 512 --batch-size 4 \
  --output results/runtime-speculative.json

python -m scripts.run_runtime_engine \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --probes results/expert-probes.pt \
  --prompts prompts.json \
  --context-tokens 512 --batch-size 4 \
  --disable-prefetch \
  --output results/runtime-demand-only.json
~~~

Both runtime modes use the same draft-ranked sparse KV set. `demand-only` suppresses early H2D but
retains the predictor for an apples-to-apples sparse-attention choice. Timing includes draft prefill,
rollout, draft-cache advancement, target compute and every demand wait.

For quality analysis, one non-performance run can compare every sparse attention output with a
CPU full-attention shadow and evaluate several stopping thresholds on the same Target queries:

~~~bash
python -m scripts.run_runtime_engine \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --probes results/expert-probes.pt --prompts prompts.json \
  --context-tokens 512 --batch-size 4 \
  --shadow-attention --shadow-thresholds 0.90,0.95,0.99 \
  --output results/runtime-shadow-batch4-c512.json
~~~

Shadow runs deliberately mark latency as invalid. The vLLM fused-MoE adapter is opt-in with
`--moe-backend vllm`; the default remains `torch` until actual-model numerical and latency checks
pass. Both backends use the same CPU source, packed GPU expert slots, queue and residency policy.

Variable output lengths and request backfill are available through the continuous runner:

~~~bash
python -m scripts.run_continuous_runtime \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --probes results/expert-probes.pt --prompts prompts.json \
  --request-count 8 --max-batch-size 4 --context-tokens 512 \
  --output-lengths 4,8,12,16 \
  --output results/runtime-continuous.json
~~~

结果会分别记录等待队列时延、单请求预填充、解码服务、活跃服务、TTFT 和端到端请求
时延，使准入与回填成本保持可见，而不是全部折叠进单一吞吐量指标。
同一轮准入中长度相同的 prompt 会合并为一次 Target 预填充；不同长度仍分组执行，且每个
请求继续维护独立的 Draft KV 状态。同组请求的 Draft 前缀也只批量计算一次，随后拆成
请求私有 KV 缓存供独立推进；首轮 lookahead 同样先按组计算，再拆分为请求私有预测。
后续刷新只合并缓存长度与推进进度兼容的请求，完成批量 advance/rollout 后立即恢复私有
状态，并在结果中记录刷新批次及其峰值大小。
Draft attention 与 probe feature 按唯一 Draft 层批量搬到 CPU 后复用，避免在层、请求和
旧 KV 块的内层循环中反复产生设备同步。

The primary experiment treats the draft as a prefetch oracle, not as a source of tokens for target
verification. At every target step, the draft independently rolls out from only the currently known
target prefix. Its attention and hidden states issue hypothetical CPU-to-GPU prefetch requests;
future target accesses are used only as offline ground truth.

## Independent offload-prefetch experiment

~~~bash
python -m scripts.run_offload_prefetch \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --prompts prompts.json \
  --output results/offload-prefetch.json
~~~

The defaults use a four-token draft lookahead and a simulated batch size of four, and report results
separately for horizons 1 through 4. For each horizon, the experiment measures token agreement and
resource agreement independently.
It also reports resource recall conditioned on the draft token matching or differing from the
target token. This directly tests whether memory-access predictions remain useful after textual
rollouts diverge.

The batch policy compares fixed per-request Top-K with dynamic allocation under the same total
number of transfers. KV chunks remain request-private, while an expert loaded once is shared by all
requests that need that layer/expert pair. An oracle allocation is reported as an upper bound. This
stage measures aggregate hit utility; it does not yet claim an end-to-end latency speedup or model
the lifetime and eviction of resident GPU objects.

The script has an idle-GPU guard enabled by default. It refuses to load either model if a visible
GPU is using more than 1,000 MiB or has more than 10% utilization, and checks again between target
and draft phases. Override it only in an environment where GPU sharing is intentional.

### Independent rollout result

The full batch-4 run used four training prompts and four held-out prompts. Draft token agreement
falls with lookahead, but resource agreement remains useful even when the individual draft token
differs from the target token.

| Horizon | Token match | KV recall@2 | KV recall when token differs | Expert recall@8 | Expert recall when token differs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 95.31% | **80.70%** | **82.29%** | **62.61%** | **45.49%** |
| 2 | 90.00% | **83.66%** | **77.60%** | **59.34%** | **42.58%** |
| 3 | 87.50% | **85.86%** | **77.53%** | **56.46%** | **39.62%** |
| 4 | 82.69% | **87.40%** | **78.13%** | **54.22%** | **38.63%** |

At horizon 1, random and recency KV recall are 59.49% and 43.28%; expert frequency and random
recall are 35.97% and 6.51%. The mismatch-conditioned sample is small, but it supports the central
hypothesis: exact token acceptance is not necessary for a draft rollout to predict memory access.

The batch expert policy exploits cross-request reuse. Under exactly the same number of expert
transfers as the static union, aggregate recall improves consistently:

| Horizon | Static expert recall | Dynamic expert recall | Oracle | Dynamic hits / transfer |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 72.33% | **72.62%** | 86.00% | 1.877 |
| 2 | 69.64% | **70.20%** | 84.14% | 1.807 |
| 3 | 67.76% | **68.21%** | 83.31% | 1.713 |
| 4 | 66.30% | **66.68%** | 82.77% | 1.579 |

The first global KV allocator is not yet a win: compared with fixed Top-2 per request, it loses
0.81--1.62 recall points. It increases target attention-mass coverage by only 0.16--0.29 points at
horizons 1--3 and loses 0.04 points at horizon 4. A practical KV scheduler should therefore include
a per-request minimum quota or explicit miss cost instead of maximizing raw draft attention alone.
The complete result is in `results/offload-prefetch-batch4.json`.

## Physical CPU-offload replay

The system replay materializes the offloaded objects as BF16 tensors in pinned CPU memory and uses
a dedicated CUDA stream for non-blocking H2D prefetch. A persistent, finite LRU GPU cache stores 64
expert objects (576 MiB) and 512 KV chunks (8 MiB). At each target demand point, missing objects are
really copied synchronously from CPU and counted as visible stall. The payloads exactly match Qwen3
shapes: 9,437,184 bytes per expert and 16,384 bytes per eight-token KV chunk.

Generate the event trace and replay it with:

~~~bash
python -m scripts.run_offload_prefetch \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --prompts prompts.json --batch-size 4 \
  --output results/offload-prefetch-batch4.json \
  --events-output results/offload-events-batch4.json

python -m scripts.run_system_offload \
  --events results/offload-events-batch4.json \
  --output results/system-offload-batch4.json
~~~

For experts, the dynamic policy hits 58.26% of unique batch-layer expert objects. Pairwise request
recall is higher (72.62%) because one shared expert can satisfy several requests. Wrong prefetches
increase traffic from 120.52 GiB with demand-only loading to 151.88 GiB, so enough overlap is
essential:

| Available overlap | No-prefetch stall | Dynamic visible stall | Stall reduction |
| ---: | ---: | ---: | ---: |
| 0 ms | 4,527 ms | 5,625 ms | -24.3% |
| 0.5 ms | 4,539 ms | 5,189 ms | -14.3% |
| 1 ms | 4,541 ms | 4,798 ms | -5.6% |
| 2 ms | 4,538 ms | 4,026 ms | **11.3%** |
| 4 ms | 4,531 ms | 2,802 ms | **38.2%** |
| 6 ms | 4,532 ms | 2,150 ms | **52.6%** |
| 8 ms | 4,526 ms | 1,941 ms | **57.1%** |

The measured expert break-even is therefore between one and two milliseconds on this machine. At
two milliseconds the oracle reduces stall by 29.3%, showing that a better expert predictor or a
smaller prefetch set has substantial headroom.

For KV, the persistent cache already hits 80.68% without prefetch; static and dynamic prefetch raise
this to 94.08% and 93.33%. At zero artificial overlap they reduce measured transfer stall by 32.2%
and 27.7%. However, a 16 KiB object is so small that Python and CUDA-launch jitter are comparable to
the copy itself, and the longer-lead sweep is not monotonic. These measurements establish real
offload and recall behavior but not a reliable KV latency speedup. A production KV path must
coalesce chunks into larger DMA requests and be tested on long contexts.

The replay transfers real payload-sized bytes and measures real CUDA synchronization. It does not
execute the expert matrix multiplications or patch Hugging Face's attention/MoE kernels, so the
reported stall reduction is a system trace replay rather than end-to-end model latency.

## Questions and tests

### KV-cache blocks

For every generated token, the experiment records the attention mass assigned to contiguous KV
blocks by both models. Target layers are paired with draft layers at the nearest relative depth
(for example, target layer 47 maps to draft layer 27). The draft's top-K blocks are treated as the
prefetch prediction.

Reported metrics include recall@K, target attention mass covered by the prefetched blocks, NDCG@K,
and weighted Jaccard. Random-block and most-recent-block policies are included as baselines.
Attention distributions are normalized before comparison so the models' different head counts do
not bias the result.

### MoE experts

Qwen3-0.6B is dense, so it has no expert IDs that can be compared directly. Instead, the experiment
uses its aligned per-token hidden states:

1. Generate and trace half of the prompts as training data.
2. Fit one small multi-output ridge probe per target MoE layer to predict the target's top-8 route.
3. Evaluate only on tokens from held-out prompts.

The probe is compared with random experts and the eight most frequent training experts. This split
is important: training and evaluating a probe on the same tokens would not establish predictive
value. A draft token's representation is available before the target verifies that token, so this
signal is causally available to a prefetcher.

## Install and test

~~~bash
python -m pip install -e ".[dev]"
pytest -q
ruff check .
~~~

## Controlled aligned-sequence experiment

With local model snapshots:

~~~bash
python -m scripts.run_experiment \
  --target /root/models/Qwen3-30B-A3B \
  --draft /root/models/Qwen3-0.6B \
  --prompts prompts.json \
  --output results/qwen3-30b-a3b.json
~~~

Model IDs can be used in place of local paths. Defaults are eight prompts, a deterministic
16-token continuation, 8-token KV blocks, top-2 KV prediction, and a 50/50 prompt-level
train/evaluation split. Run python -m scripts.run_experiment --help for all controls.

For a quick pipeline check (KV only, because the target is dense):

~~~bash
python -m scripts.run_experiment \
  --target /root/models/Qwen3-0.6B \
  --draft /root/models/Qwen3-0.6B \
  --prompts prompts.json \
  --max-new-tokens 2 \
  --output results/smoke.json
~~~

## Interpreting the result

Evidence for useful prefetching requires the draft/probe recall to beat its relevant baseline on
held-out prompts. Recall estimates the fraction of required blocks or experts available after a
top-K prefetch. NDCG additionally rewards getting high-attention blocks or high-probability experts
near the front of the prefetch queue. These metrics establish predictability, not end-to-end
latency improvement; a runtime integration must still measure transfer overlap, bandwidth, and
eviction costs.

## Previous aligned-sequence result

This earlier experiment replays the target sequence through the draft. It is a controlled upper
bound rather than the independent prefetch setting above. The full bfloat16 run used four training
prompts and four held-out prompts, with 16 generated
tokens per prompt. Each task therefore contains 3,072 held-out layer-token observations
(48 target layers x 64 tokens).

| Prediction | Signal | Recall | Stronger baseline | Baseline recall |
| --- | --- | ---: | --- | ---: |
| KV top-2 blocks | Draft attention | **84.67%** | Random blocks | 56.84% |
| KV top-2 blocks | Draft attention | **84.67%** | Most-recent blocks | 47.95% |
| MoE top-8 experts | Draft hidden-state probe | **64.68%** | Training-frequency experts | 35.97% |
| MoE top-8 experts | Draft hidden-state probe | **64.68%** | Random experts | 6.51% |

The draft-selected KV blocks cover 81.04% of target attention mass, versus 57.35% for random
blocks and 30.62% for most-recent blocks. KV NDCG@2 is 0.945 and full-distribution weighted
Jaccard is 0.729. Expert NDCG@8 is 0.723.

On this workload, both signals provide substantial predictive value: KV recall improves by
27.83 percentage points over random, while expert recall improves by 28.71 points over the
stronger frequency baseline. The complete per-layer measurements are in
results/qwen3-30b-a3b.json. Draft KV recall beats random on 47 of 48 target layers, and the expert
probe beats both baselines on all 48 layers. This is a positive trace-study result, but the small
four-prompt held-out set should be expanded before making a production latency claim.

## Limitations

- Transformer attention mass is a proxy for KV utility, not a direct measurement of PCIe traffic.
- Relative-depth layer alignment is intentionally simple and may not be optimal.
- The prompt set is small; confidence should be strengthened with a larger, shuffled workload.
- The probe measures whether draft hidden states contain routing information. Production use must
  account for probe latency and overlap it with target execution.
- Physical replay uses payload-shaped tensors rather than the original checkpoint values. Tensor
  contents do not affect PCIe transfer timing, but an integrated engine is still required to test
  numerical output and full-model latency.
