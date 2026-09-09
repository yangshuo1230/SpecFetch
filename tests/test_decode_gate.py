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
        },
        "performance": {
            **shared_performance,
            "latency_valid": True,
            "decode_throughput_tokens_per_second": runtime_throughput,
        },
    }
    vllm = {
        "timing_protocol": "batch_decode_after_all_first_tokens_v1",
        "configuration": {
            "model": "/models/qwen",
            "batch_size": 4,
            "context_tokens": context,
            "max_new_tokens": 65,
            "cpu_offload_gb": 54.0,
        },
        "engine_options": {"dtype": "bfloat16"},
        "performance": {
            **shared_performance,
            "decode_throughput_tokens_per_second": vllm_throughput,
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
