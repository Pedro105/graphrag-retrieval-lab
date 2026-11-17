"""Confidence intervals and significance tests over per-question scores."""

import random
from collections.abc import Sequence
from statistics import fmean


def bootstrap_ci(
    scores: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, low, high): the mean with a percentile bootstrap interval."""
    if not scores:
        raise ValueError("no scores")
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(fmean(rng.choices(scores, k=n)) for _ in range(resamples))
    tail = (1.0 - confidence) / 2
    low = means[int(tail * resamples)]
    high = means[min(resamples - 1, int((1.0 - tail) * resamples))]
    return fmean(scores), low, high


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for "systems a and b have the same mean score".

    `a[i]` and `b[i]` must be the two systems' scores on the same question.
    Under the null hypothesis the two could be swapped on any question, so
    the sign of each difference is flipped at random and the observed mean
    difference is compared with the resulting distribution.
    """
    if len(a) != len(b):
        raise ValueError("a and b must score the same questions")
    if not a:
        raise ValueError("no scores")
    differences = [x - y for x, y in zip(a, b, strict=True)]
    observed = abs(fmean(differences))
    rng = random.Random(seed)
    as_extreme = 0
    for _ in range(resamples):
        flipped = fmean(d if rng.random() < 0.5 else -d for d in differences)
        # The tolerance stops float noise deciding ties, which are common
        # with 0/1 scores such as exact match.
        if abs(flipped) >= observed - 1e-12:
            as_extreme += 1
    # Add-one so the estimate is never exactly zero.
    return (as_extreme + 1) / (resamples + 1)
