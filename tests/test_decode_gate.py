import pytest

from src.decode_gate import evaluate_decode_pair


def result_pair(runtime_throughput=6.0, vllm_throughput=4.0, context=512):
    shared_performance = {
        "decode_wall_seconds": 10.0,
        "decode_tokens_per_request": 64,
        "decode_tokens": 256,
    }
    runtime = {
        "timing_protocol": "batch_decode_after_all_prefix_caches_v1",
        "configuration": {
            "target": "/models/qwen",
            "batch_size": 4,
            "context_tokens": context,
            "max_new_tokens": 65,
            "gpu_memory_limit_gib": 10.0,
        },
        "performance": {
            **shared_performance,
            "latency_valid": True,
            "decode_throughput_tokens_per_second": runtime_throughput,
        },
        "gpu_memory": {
            "measurement_protocol": "post_initialization_peak_reserved_v1",
            "limit_gib": 10.0,
            "peak_reserved_gib": 4.5,
            "limit_satisfied": True,
        },
    }
    vllm = {
        "timing_protocol": "batch_decode_from_cached_first_token_prefix_v2",
        "configuration": {
            "model": "/models/qwen",
            "batch_size": 4,
            "context_tokens": context,
            "max_new_tokens": 65,
            "cpu_offload_gb": 54.0,
            "gpu_memory_limit_gib": 10.0,
        },
        "engine_options": {
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.1,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": False,
        },
        "performance": {
            **shared_performance,
            "decode_throughput_tokens_per_second": vllm_throughput,
        },
        "gpu_memory": {
            "measurement_protocol": "post_initialization_peak_reserved_v1",
            "limit_gib": 10.0,
            "total_device_gib": 100.0,
            "peak_reserved_gib": 9.9,
            "limit_satisfied": True,
        },
    }
    return runtime, vllm


def test_decode_gate_accepts_exact_threshold():
    runtime, vllm = result_pair()
    result = evaluate_decode_pair(runtime, vllm, context_tokens=512)
    assert result.speedup == 1.5
    assert result.passed


def test_decode_gate_rejects_below_threshold_without_rounding():
    runtime, vllm = result_pair(runtime_throughput=5.999)
    result = evaluate_decode_pair(runtime, vllm, context_tokens=512)
    assert not result.passed


def test_decode_gate_rejects_legacy_or_unmatched_results():
    runtime, vllm = result_pair()
    del vllm["timing_protocol"]
    with pytest.raises(ValueError, match="decode-only"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)

    runtime, vllm = result_pair()
    runtime["configuration"]["max_new_tokens"] = 64
    with pytest.raises(ValueError, match="max_new_tokens=65"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)


def test_decode_gate_requires_matched_satisfied_gpu_memory_limits():
    runtime, vllm = result_pair()
    vllm["gpu_memory"]["limit_gib"] = 11.0
    vllm["configuration"]["gpu_memory_limit_gib"] = 11.0
    with pytest.raises(ValueError, match="limits differ"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)

    runtime, vllm = result_pair()
    runtime["gpu_memory"]["peak_reserved_gib"] = 10.1
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)

    runtime, vllm = result_pair()
    del runtime["gpu_memory"]
    with pytest.raises(ValueError, match="measurement"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)

    runtime, vllm = result_pair()
    vllm["engine_options"]["gpu_memory_utilization"] = 0.2
    with pytest.raises(ValueError, match="engine was not configured"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)

    runtime, vllm = result_pair()
    vllm["engine_options"]["enable_prefix_caching"] = False
    with pytest.raises(ValueError, match="prefix caching"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)

    runtime, vllm = result_pair()
    vllm["engine_options"]["enable_chunked_prefill"] = True
    with pytest.raises(ValueError, match="chunked prefill"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)

    runtime, vllm = result_pair()
    runtime["gpu_memory"]["limit_gib"] = float("nan")
    with pytest.raises(ValueError, match="positive GPU-memory limit"):
        evaluate_decode_pair(runtime, vllm, context_tokens=512)
