from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def top_k(values: Mapping[int, float], k: int) -> list[int]:
    return sorted(values, key=lambda key: values[key], reverse=True)[: max(0, k)]


def overlap_at_k(predicted: Sequence[int], actual: Sequence[int], k: int) -> float:
    effective_k = min(k, len(predicted), len(actual))
    if effective_k <= 0:
        return 0.0
    return len(set(predicted[:effective_k]) & set(actual[:effective_k])) / effective_k


def recall_at_k(predicted: Sequence[int], actual: Sequence[int], k: int) -> float:
    actual_set = set(actual[:k])
    return len(set(predicted[:k]) & actual_set) / len(actual_set) if actual_set else 0.0


def weighted_jaccard(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    keys = set(left) | set(right)
    intersection = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    union = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return intersection / union if union else 0.0


def ndcg_at_k(predicted: Sequence[int], actual: Mapping[int, float], k: int) -> float:
    actual_top = top_k(actual, k)
    if not actual_top:
        return 0.0
    dcg = sum(actual.get(key, 0.0) / math.log2(rank + 2) for rank, key in enumerate(predicted[:k]))
    ideal = sum(actual[key] / math.log2(rank + 2) for rank, key in enumerate(actual_top))
    return dcg / ideal if ideal else 0.0


def fit_ridge_probe(
    features: torch.Tensor, labels: torch.Tensor, alpha: float
) -> dict[str, torch.Tensor]:
    """Fit a centered multi-output ridge probe using the smaller dual system."""
    if features.ndim != 2 or labels.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must be aligned 2-D tensors")
    if not len(features):
        raise ValueError("cannot fit an empty probe")
    x = features.float()
    y = labels.float()
    x_mean = x.mean(0, keepdim=True)
    x_scale = x.std(0, correction=0, keepdim=True).clamp_min(1e-5)
    y_mean = y.mean(0, keepdim=True)
    x = (x - x_mean) / x_scale
    y = y - y_mean
    eye = torch.eye(x.shape[0], dtype=x.dtype)
    coefficients = x.T @ torch.linalg.solve(x @ x.T + alpha * eye, y)
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "coef": coefficients}


def predict_probe(probe: dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
    x = (features.float() - probe["x_mean"]) / probe["x_scale"]
    return x @ probe["coef"] + probe["y_mean"]
