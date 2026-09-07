from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import torch

from src.gpu_guard import require_idle_gpus

PAYLOADS = {
    "expert": 3 * 2048 * 768,  # gate, up, and down projections in BF16
    "kv": 2 * 8 * 4 * 128,  # K/V, 8 tokens, 4 KV heads, head_dim 128
}


class GpuObjectCache:
    def __init__(self, capacity: int, elements: int, device: torch.device):
        self.storage = torch.empty((capacity, elements), dtype=torch.bfloat16, device=device)
        self.resident: OrderedDict[tuple[str, int, int], int] = OrderedDict()
        self.free = list(range(capacity))
        self.evictions = 0

    def ensure_slot(self, key: tuple[str, int, int], protected: set) -> tuple[int, bool]:
        if key in self.resident:
            slot = self.resident.pop(key)
            self.resident[key] = slot
            return slot, False
        if self.free:
            slot = self.free.pop()
        else:
            victim = next(item for item in self.resident if item not in protected)
            slot = self.resident.pop(victim)
            self.evictions += 1
        self.resident[key] = slot
        return slot, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay real pinned-CPU offload transfers.")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lead-ms", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--expert-cache-objects", type=int, default=64)
    parser.add_argument("--kv-cache-objects", type=int, default=512)
    parser.add_argument("--source-slots", type=int, default=8)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def replay(
    events: list[dict[str, Any]],
    kind: str,
    policy: str,
    lead_ms: float,
    cache_objects: int,
    source_slots: int,
    device: torch.device,
) -> dict[str, float]:
    elements = PAYLOADS[kind]
    object_bytes = elements * torch.tensor([], dtype=torch.bfloat16).element_size()
    sources = torch.empty((source_slots, elements), dtype=torch.bfloat16, pin_memory=True)
    cache = GpuObjectCache(cache_objects, elements, device)
    stream = torch.cuda.Stream(device=device)
    rows: dict[str, list[float]] = defaultdict(list)
    transferred = 0
    demand_objects = 0
    hits = 0
    wasted = 0
    for event in (item for item in events if item["kind"] == kind):
        selected_ids = [] if policy == "none" else event[policy]
        actual_ids = event["actual"]
        selected = {(kind, event["layer"], item) for item in selected_ids}
        actual = {(kind, event["layer"], item) for item in actual_ids}
        already_resident = set(cache.resident)
        to_prefetch = []
        for key in selected:
            slot, missing = cache.ensure_slot(key, selected | actual)
            if missing:
                to_prefetch.append((key, slot))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            for key, slot in to_prefetch:
                cache.storage[slot].copy_(sources[key[2] % source_slots], non_blocking=True)
            end.record()
        if lead_ms:
            time.sleep(lead_ms / 1000)
        wait_start = time.perf_counter()
        end.synchronize()
        rows["prefetch_wait_ms"].append((time.perf_counter() - wait_start) * 1000)
        rows["prefetch_copy_ms"].append(start.elapsed_time(end))
        transferred += len(to_prefetch)
        hits += len(actual & set(cache.resident))
        demand_objects += len(actual)
        wasted += len(selected - actual - already_resident)

        missing_actual = []
        for key in actual:
            if key not in cache.resident:
                slot, _ = cache.ensure_slot(key, actual)
                missing_actual.append((key, slot))
        demand_start = time.perf_counter()
        with torch.cuda.stream(stream):
            for key, slot in missing_actual:
                cache.storage[slot].copy_(sources[key[2] % source_slots], non_blocking=True)
        stream.synchronize()
        rows["demand_stall_ms"].append((time.perf_counter() - demand_start) * 1000)
        transferred += len(missing_actual)
    result = {
        "events": float(sum(item["kind"] == kind for item in events)),
        "object_bytes": float(object_bytes),
        "demand_objects": float(demand_objects),
        "hit_rate": hits / demand_objects if demand_objects else 0.0,
        "transferred_objects": float(transferred),
        "transferred_gib": transferred * object_bytes / 2**30,
        "wasted_prefetch_objects": float(wasted),
        "evictions": float(cache.evictions),
    }
    result.update({f"mean_{name}": mean(values) for name, values in rows.items()})
    result.update({f"total_{name}": sum(values) for name, values in rows.items()})
    del cache, sources
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    require_idle_gpus(1000, 10)
    data = json.loads(args.events.read_text(encoding="utf-8"))
    events = data["events"]
    device = torch.device(args.device)
    result = {
        "configuration": {
            "device": args.device,
            "lead_ms": args.lead_ms,
            "expert_cache_objects": args.expert_cache_objects,
            "kv_cache_objects": args.kv_cache_objects,
            "source_slots": args.source_slots,
            "cpu_memory": "torch pinned BF16",
            "transfer": "non-blocking H2D on a dedicated CUDA stream",
        },
        "payload_bytes": {kind: elements * 2 for kind, elements in PAYLOADS.items()},
        "results": {},
    }
    for kind, capacity in (
        ("expert", args.expert_cache_objects),
        ("kv", args.kv_cache_objects),
    ):
        result["results"][kind] = {}
        for lead_ms in args.lead_ms:
            key = str(lead_ms)
            result["results"][kind][key] = {}
            for policy in ("none", "static", "dynamic", "oracle"):
                print(f"[{kind}] lead={lead_ms} ms policy={policy}", flush=True)
                result["results"][kind][key][policy] = replay(
                    events,
                    kind,
                    policy,
                    lead_ms,
                    capacity,
                    args.source_slots,
                    device,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
