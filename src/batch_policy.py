from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.metrics import top_k


def allocate_private(
    scores: Sequence[Mapping[int, float]], per_request_budget: int
) -> tuple[list[set[int]], list[set[int]]]:
    """Allocate equal static quotas or the same total quota dynamically.

    Private resource IDs live in separate namespaces for each request, as KV chunks do.
    """
    static = [set(top_k(request, per_request_budget)) for request in scores]
    slots = sum(len(selection) for selection in static)
    ranked = sorted(
        (
            (score, request_id, resource)
            for request_id, request in enumerate(scores)
            for resource, score in request.items()
        ),
        reverse=True,
    )[:slots]
    dynamic = [set() for _ in scores]
    for _, request_id, resource in ranked:
        dynamic[request_id].add(resource)
    return static, dynamic


def allocate_shared(
    scores: Sequence[Mapping[int, float]], per_request_budget: int
) -> tuple[set[int], set[int]]:
    """Allocate shared objects, aggregating predicted benefit across batch requests."""
    static = set().union(*(top_k(request, per_request_budget) for request in scores))
    return static, allocate_shared_slots(scores, len(static))


def allocate_shared_slots(scores: Sequence[Mapping[int, float]], slots: int) -> set[int]:
    """Choose a fixed number of shared objects by aggregate batch value."""
    aggregate: dict[int, float] = {}
    for request in scores:
        for resource, score in request.items():
            aggregate[resource] = aggregate.get(resource, 0.0) + score
    return set(top_k(aggregate, slots))


def private_utility(
    selected: Sequence[set[int]], actual: Sequence[Mapping[int, float]], required: int
) -> dict[str, float]:
    demands = sum(min(required, len(request)) for request in actual)
    hits = 0
    mass = 0.0
    for selection, request in zip(selected, actual):
        required_ids = set(top_k(request, required))
        hits += len(selection & required_ids)
        mass += sum(request.get(resource, 0.0) for resource in selection)
    return {
        "recall": hits / demands if demands else 0.0,
        "target_mass": mass / len(actual) if actual else 0.0,
        "transfers": float(sum(len(items) for items in selected)),
    }


def shared_utility(
    selected: set[int], actual: Sequence[Mapping[int, float]], required: int
) -> dict[str, float]:
    demands = sum(min(required, len(request)) for request in actual)
    hits = sum(len(selected & set(top_k(request, required))) for request in actual)
    return {
        "recall": hits / demands if demands else 0.0,
        "hits_per_transfer": hits / len(selected) if selected else 0.0,
        "transfers": float(len(selected)),
    }
