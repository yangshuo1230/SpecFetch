from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare runtime output with a dense baseline.")
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def main() -> None:
    args = parse_args()
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    runtime_tokens = runtime["generated_token_ids"]
    baseline_tokens = baseline["generated_token_ids"]
    if len(runtime_tokens) != len(baseline_tokens):
        raise ValueError("batch sizes differ")
    total = sum(min(len(left), len(right)) for left, right in zip(runtime_tokens, baseline_tokens))
    matches = sum(
        sum(a == b for a, b in zip(left, right))
        for left, right in zip(runtime_tokens, baseline_tokens)
    )
    prefixes = [common_prefix(left, right) for left, right in zip(runtime_tokens, baseline_tokens)]
    result = {
        "token_agreement": matches / total if total else 0.0,
        "exact_sequence_agreement": sum(
            left == right for left, right in zip(runtime_tokens, baseline_tokens)
        )
        / len(runtime_tokens),
        "common_prefix_tokens": prefixes,
        "runtime_end_to_end_tokens_per_second": runtime["performance"][
            "end_to_end_tokens_per_second"
        ],
        "baseline_tokens_per_second": baseline["performance"]["throughput_tokens_per_second"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
