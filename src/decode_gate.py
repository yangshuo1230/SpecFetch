from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELEASE_CONTEXTS = (512, 4096)
RELEASE_BATCH_SIZE = 4
RELEASE_DECODE_TOKENS_PER_REQUEST = 64
RELEASE_SPEEDUP = 1.5
MEMORY_PROTOCOL = "post_initialization_peak_reserved_v1"


@dataclass(frozen=True)
class DecodeGateResult:
    context_tokens: int
    runtime_tokens_per_second: float
    vllm_tokens_per_second: float
    speedup: float
    passed: bool


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_identity(value: str) -> str:
    return Path(value).resolve().as_posix()


def _positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def evaluate_decode_pair(
    runtime_result: dict[str, Any],
    vllm_result: dict[str, Any],
    *,
    context_tokens: int,
) -> DecodeGateResult:
    runtime_config = runtime_result["configuration"]
    vllm_config = vllm_result["configuration"]
    runtime_performance = runtime_result["performance"]
    vllm_performance = vllm_result["performance"]

    runtime_memory = runtime_result.get("gpu_memory", {})
    vllm_memory = vllm_result.get("gpu_memory", {})
    for label, config, memory in (
        ("SpecFetch", runtime_config, runtime_memory),
        ("vLLM", vllm_config, vllm_memory),
    ):
        if memory.get("measurement_protocol") != MEMORY_PROTOCOL:
            raise ValueError(f"{label} result lacks the required GPU-memory measurement")
        limit = memory.get("limit_gib")
        peak = memory.get("peak_reserved_gib")
        if not _positive_finite_number(limit):
            raise ValueError(f"{label} result must declare a positive GPU-memory limit")
        if not _positive_finite_number(peak):
            raise ValueError(f"{label} result must report positive peak reserved GPU memory")
        configured_limit = config.get("gpu_memory_limit_gib")
        if not _positive_finite_number(configured_limit) or not math.isclose(
            float(configured_limit), float(limit), rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"{label} configured and measured GPU-memory limits differ")
        if float(peak) > float(limit) or memory.get("limit_satisfied") is not True:
            raise ValueError(f"{label} result exceeds its GPU-memory limit")
    runtime_limit = float(runtime_memory["limit_gib"])
    vllm_limit = float(vllm_memory["limit_gib"])
    if not math.isclose(runtime_limit, vllm_limit, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("SpecFetch and vLLM GPU-memory limits differ")
    vllm_total = float(vllm_memory.get("total_device_gib", 0))
    vllm_utilization = float(vllm_result.get("engine_options", {}).get("gpu_memory_utilization", 0))
    if (
        not math.isfinite(vllm_total)
        or not math.isfinite(vllm_utilization)
        or vllm_total <= 0
        or not math.isclose(vllm_total * vllm_utilization, vllm_limit, rel_tol=0.0, abs_tol=1e-5)
    ):
        raise ValueError("vLLM engine was not configured with the declared GPU-memory limit")

    if runtime_result.get("timing_protocol") != "batch_decode_after_all_prefix_caches_v1":
        raise ValueError("SpecFetch result does not use the decode-only timing protocol")
    if vllm_result.get("timing_protocol") != "batch_decode_after_all_first_tokens_v1":
        raise ValueError("vLLM result does not use the decode-only timing protocol")
    if not runtime_performance.get("latency_valid", False):
        raise ValueError("SpecFetch latency is invalidated by shadow evaluation")
    if float(vllm_config.get("cpu_offload_gb", 0)) <= 0:
        raise ValueError("vLLM release baseline must enable CPU weight offload")
    if vllm_result.get("engine_options", {}).get("dtype") != "bfloat16":
        raise ValueError("vLLM release baseline must use bfloat16")
    if _model_identity(runtime_config["target"]) != _model_identity(vllm_config["model"]):
        raise ValueError("SpecFetch and vLLM model paths differ")

    expected = {
        "batch_size": RELEASE_BATCH_SIZE,
        "context_tokens": context_tokens,
        "max_new_tokens": RELEASE_DECODE_TOKENS_PER_REQUEST + 1,
    }
    for name, value in expected.items():
        if runtime_config.get(name) != value or vllm_config.get(name) != value:
            raise ValueError(f"matched release configuration requires {name}={value}")
    expected_decode_tokens = RELEASE_BATCH_SIZE * RELEASE_DECODE_TOKENS_PER_REQUEST
    for label, performance in (
        ("SpecFetch", runtime_performance),
        ("vLLM", vllm_performance),
    ):
        if performance.get("decode_tokens_per_request") != RELEASE_DECODE_TOKENS_PER_REQUEST:
            raise ValueError(f"{label} result does not time 64 decode tokens per request")
        if performance.get("decode_tokens") != expected_decode_tokens:
            raise ValueError(f"{label} result has the wrong batch decode-token count")
        if float(performance.get("decode_wall_seconds", 0)) <= 0:
            raise ValueError(f"{label} decode wall time must be positive")

    runtime_throughput = float(runtime_performance["decode_throughput_tokens_per_second"])
    vllm_throughput = float(vllm_performance["decode_throughput_tokens_per_second"])
    if runtime_throughput <= 0 or vllm_throughput <= 0:
        raise ValueError("decode throughput must be positive")
    speedup = runtime_throughput / vllm_throughput
    return DecodeGateResult(
        context_tokens,
        runtime_throughput,
        vllm_throughput,
        speedup,
        speedup >= RELEASE_SPEEDUP,
    )


def evaluate_release_gate(paths: dict[int, tuple[Path, Path]]) -> list[DecodeGateResult]:
    if tuple(sorted(paths)) != RELEASE_CONTEXTS:
        raise ValueError(f"release gate requires contexts {RELEASE_CONTEXTS}")
    return [
        evaluate_decode_pair(
            _load(paths[context][0]), _load(paths[context][1]), context_tokens=context
        )
        for context in RELEASE_CONTEXTS
    ]
