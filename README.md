# Speculative Offload Prefetch Study

This repository tests whether signals already produced by speculative decoding can prefetch
offloaded memory for Qwen/Qwen3-30B-A3B. The draft model is Qwen/Qwen3-0.6B.

The study is observational: target decoding is unchanged, and every comparison uses the same
target-generated token sequence.

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

## Run

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

## Measured result

The full bfloat16 run used four training prompts and four held-out prompts, with 16 generated
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
