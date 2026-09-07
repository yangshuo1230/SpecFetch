# Production baseline

The machine image contains vLLM 0.11.1+cu129 but its global Python environment has an older,
incompatible Transformers. Keep the baseline isolated:

~~~bash
python -m venv --system-site-packages /root/.venvs/specfetch-vllm
/root/.venvs/specfetch-vllm/bin/python -m pip install \
  --index-url https://pypi.org/simple \
  -r benchmarks/vllm-requirements.txt
~~~

Full-resident production upper bound:

~~~bash
/root/.venvs/specfetch-vllm/bin/python -m scripts.run_vllm_baseline \
  --model /root/models/Qwen3-30B-A3B \
  --prompts prompts.json --batch-size 4 --context-tokens 512 \
  --output results/vllm-full.json
~~~

Production weight-offload comparison:

~~~bash
/root/.venvs/specfetch-vllm/bin/python -m scripts.run_vllm_baseline \
  --model /root/models/Qwen3-30B-A3B \
  --prompts prompts.json --batch-size 4 --context-tokens 512 \
  --gpu-memory-utilization 0.1 --cpu-offload-gb 54 \
  --output results/vllm-weight-offload.json
~~~

vLLM native KV offloading extends completed/prefix block storage. It is a useful production data
movement baseline, but it does not implement this project's sparse retrieval of old blocks from an
active decode sequence. Results must state that semantic difference.
