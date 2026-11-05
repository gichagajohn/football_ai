"""Leakage-aware walk-forward evaluation for Dixon-Coles predictions."""

from __future__ import annotations

from datetime import datetime
import math

from backend.models.dixon_coles import fit_dixon_coles
from backend.models.markets import blend_probabilities, normalized_implied, value_metrics


def _kickoff(match: dict) -> datetime:
    return datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))


def walk_forward_backtest(
    matches: list[dict],
    initial_train: int | None = None,
    retrain_every: int = 20,
    time_decay_days: float = 240.0,
    market_weight: float = 0.35,
    min_selection_probability: float = 0.50,
) -> dict:
    """Evaluate chronologically, never fitting on the match being predicted.

    Historical matches may optionally contain an ``odds_snapshot`` with home_win,
    draw and away_win prices. When present, positive-EV 1X2 bets are evaluated at
    one flat unit each. Calibration metrics do not require historical odds.
    """
    ordered = sorted(matches, key=lambda row: row.get("utcDate") or "")
    if initial_train is None:
        initial_train = max(60, len(ordered) // 2)
    if initial_train < 24 or len(ordered) <= initial_train:
        raise ValueError("Not enough chronological matches for train/test evaluation")

    predictions: list[list[float]] = []
    outcomes: list[list[int]] = []
    bet_returns: list[float] = []
    skipped = 0
    model = None
    refits = 0

    for index in range(initial_train, len(ordered)):
        match = ordered[index]
        if model is None or (index - initial_train) % max(1, retrain_every) == 0:
            try:
                model = fit_dixon_coles(
                    ordered[:index], as_of=_kickoff(match), time_decay_days=time_decay_days
                )
                refits += 1
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
        try:
            home_id = int(match["homeTeam"]["id"])
            away_id = int(match["awayTeam"]["id"])
            model_markets = model.probabilities(home_id, away_id)
            full_time = match["score"]["fullTime"]
            home_goals, away_goals = int(full_time["home"]), int(full_time["away"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        odds = match.get("odds_snapshot") or match.get("odds") or {}
        market_probabilities, _ = normalized_implied({
            key: odds.get(key) for key in ("home_win", "draw", "away_win")
        }) if all(odds.get(key) for key in ("home_win", "draw", "away_win")) else ({}, None)
        probabilities = blend_probabilities(model_markets, market_probabilities, market_weight)
        vector = [probabilities[key] for key in ("home_win", "draw", "away_win")]
        actual_index = 0 if home_goals > away_goals else 2 if away_goals > home_goals else 1
        actual = [int(i == actual_index) for i in range(3)]
        predictions.append(vector)
        outcomes.append(actual)

        if market_probabilities:
            candidates = []
            for key in ("home_win", "draw", "away_win"):
                try:
                    metrics = value_metrics(probabilities[key], odds[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if probabilities[key] >= min_selection_probability and metrics["expected_value"] >= 0:
                    candidates.append((metrics["expected_value"], odds[key], key))
            if candidates:
                _, price, key = max(candidates)
                won = (key == ("home_win", "draw", "away_win")[actual_index])
                bet_returns.append(float(price) if won else 0.0)

    count = len(predictions)
    if not count:
        raise ValueError("No test predictions could be generated")
    multiclass_brier = sum(
        sum((p - y) ** 2 for p, y in zip(prob_row, actual_row))
        for prob_row, actual_row in zip(predictions, outcomes)
    ) / count
    multiclass_log_loss = -sum(
        math.log(min(1 - 1e-15, max(1e-15, prob_row[actual_row.index(1)])))
        for prob_row, actual_row in zip(predictions, outcomes)
    ) / count
    accuracy = sum(
        prob_row.index(max(prob_row)) == actual_row.index(1)
        for prob_row, actual_row in zip(predictions, outcomes)
    ) / count
    staked = len(bet_returns)
    return {
        "predictions": count,
        "model_refits": refits,
        "skipped": skipped,
        "accuracy": accuracy,
        "multiclass_brier_score": multiclass_brier,
        "multiclass_log_loss": multiclass_log_loss,
        "value_bets": staked,
        "flat_stake_roi": (sum(bet_returns) - staked) / staked if staked else None,
        "initial_train": initial_train,
        "retrain_every": retrain_every,
        "time_decay_days": time_decay_days,
        "market_weight": market_weight,
        "note": "Historical evaluation only; no future performance is guaranteed.",
    }
