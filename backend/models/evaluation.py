"""Small, dependency-free metrics for time-ordered model evaluation."""

from __future__ import annotations

import math


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("Probabilities and outcomes must have equal non-zero length")
    return sum((float(p) - int(y)) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def log_loss(probabilities: list[float], outcomes: list[int], epsilon: float = 1e-15) -> float:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("Probabilities and outcomes must have equal non-zero length")
    losses = []
    for probability, outcome in zip(probabilities, outcomes):
        p = min(1 - epsilon, max(epsilon, float(probability)))
        losses.append(-math.log(p if outcome else 1 - p))
    return sum(losses) / len(losses)


def flat_stake_roi(probabilities: list[float], odds: list[float], outcomes: list[int]) -> float:
    if not (len(probabilities) == len(odds) == len(outcomes)) or not outcomes:
        raise ValueError("Inputs must have equal non-zero length")
    returns = sum(float(price) if outcome else 0.0 for price, outcome in zip(odds, outcomes))
    return (returns - len(outcomes)) / len(outcomes)
