"""Confidence intervals for proportion metrics (accuracy).

Wilson score interval — well-behaved for small n and proportions near 0/1, unlike the
normal approximation. Returns ``None`` when n is 0.
"""

from __future__ import annotations

import math
import random


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    """Two-sided Wilson score interval for a binomial proportion (default 95%)."""
    if n <= 0:
        return None
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return [lo, hi]


def bootstrap_mean_interval(values: list[float], *, confidence: float = 0.95,
                            iterations: int = 5000, seed: int = 20260721) -> list[float] | None:
    """Deterministic percentile bootstrap interval for a mean of continuous scores."""
    if not values:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(iterations))
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * iterations))]
    hi = means[min(iterations - 1, int((1.0 - alpha) * iterations) - 1)]
    return [lo, hi]


def bootstrap_weighted_mean_interval(
    values: list[float],
    weights: list[float],
    *,
    confidence: float = 0.95,
    iterations: int = 5000,
    seed: int = 20260721,
) -> list[float] | None:
    """Deterministic case-resampling interval for a weighted mean."""
    if not values:
        return None
    if len(values) != len(weights) or any(weight <= 0 for weight in weights):
        raise ValueError("values and positive weights must have the same length")
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        weight_sum = sum(weights[index] for index in indices)
        means.append(sum(values[index] * weights[index] for index in indices) / weight_sum)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * iterations))]
    hi = means[min(iterations - 1, int((1.0 - alpha) * iterations) - 1)]
    return [lo, hi]


def bootstrap_clustered_weighted_mean_interval(
    clusters: list[list[tuple[float, float]]],
    *,
    confidence: float = 0.95,
    iterations: int = 5000,
    seed: int = 20260721,
) -> list[float] | None:
    """Deterministic cluster bootstrap for repeated responses from the same sample.

    Each inner list contains ``(score, weight)`` pairs for one source sample.  Sampling
    complete clusters rather than individual responses preserves the correlation among
    ``generation.n`` repeats and therefore avoids reporting an artificially narrow
    confidence interval.
    """
    if not clusters:
        return None
    if any(not cluster for cluster in clusters):
        raise ValueError("clusters must not be empty")
    if any(weight <= 0 for cluster in clusters for _, weight in cluster):
        raise ValueError("cluster weights must be positive")

    def weighted_mean(sampled_clusters: list[list[tuple[float, float]]]) -> float:
        weight_sum = sum(weight for cluster in sampled_clusters for _, weight in cluster)
        return sum(score * weight for cluster in sampled_clusters for score, weight in cluster) / weight_sum

    if len(clusters) == 1:
        point = weighted_mean(clusters)
        return [point, point]

    rng = random.Random(seed)
    n = len(clusters)
    means = sorted(
        weighted_mean([clusters[rng.randrange(n)] for _ in range(n)])
        for _ in range(iterations)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * iterations))]
    hi = means[min(iterations - 1, int((1.0 - alpha) * iterations) - 1)]
    return [lo, hi]
