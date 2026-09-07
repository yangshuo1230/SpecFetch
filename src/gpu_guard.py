from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuState:
    index: int
    memory_used_mib: int
    utilization_percent: int


def query_gpu_states() -> list[GpuState]:
    """Read GPU load without initializing CUDA in the experiment process."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    states = []
    for line in output.splitlines():
        index, memory, utilization = (int(value.strip()) for value in line.split(","))
        states.append(GpuState(index, memory, utilization))
    return states


def require_idle_gpus(max_memory_mib: int, max_utilization: int) -> list[GpuState]:
    states = query_gpu_states()
    busy = [
        state
        for state in states
        if state.memory_used_mib > max_memory_mib or state.utilization_percent > max_utilization
    ]
    if busy:
        details = ", ".join(
            f"GPU {state.index}: {state.memory_used_mib} MiB, {state.utilization_percent}% util"
            for state in busy
        )
        raise RuntimeError(f"refusing to disturb busy GPUs ({details})")
    return states
