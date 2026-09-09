from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.decode_gate import RELEASE_SPEEDUP, evaluate_release_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check both matched decode-only release workloads."
    )
    parser.add_argument("--runtime-c512", required=True, type=Path)
    parser.add_argument("--vllm-c512", required=True, type=Path)
    parser.add_argument("--runtime-c4096", required=True, type=Path)
    parser.add_argument("--vllm-c4096", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_release_gate(
        {
            512: (args.runtime_c512, args.vllm_c512),
            4096: (args.runtime_c4096, args.vllm_c4096),
        }
    )
    print(
        json.dumps(
            {"required_speedup": RELEASE_SPEEDUP, "workloads": [asdict(item) for item in results]},
            indent=2,
        )
    )
    if not all(item.passed for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
