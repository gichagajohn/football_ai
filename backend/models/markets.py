"""Bookmaker-margin and expected-value helpers."""

from __future__ import annotations

import math


def valid_probability(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 < value < 1 else None


def normalized_implied(odds_by_market: dict) -> tuple[dict[str, float], float | None]:
    """Remove overround from mutually exclusive decimal-odds outcomes."""
    raw = {}
    for market, odds in odds_by_market.items():
        try:
            odds = float(odds)
        except (TypeError, ValueError):
            continue
        if math.isfinite(odds) and odds > 1.0:
            raw[market] = 1.0 / odds
    overround = sum(raw.values())
    if not raw or overround <= 0:
        return {}, None
    return {market: value / overround for market, value in raw.items()}, overround


def value_metrics(model_probability: float, bookmaker_odds: float) -> dict[str, float]:
    probability = valid_probability(model_probability)
    if probability is None:
        raise ValueError("Invalid model probability")
    odds = float(bookmaker_odds)
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("Invalid decimal odds")
    implied = 1.0 / odds
    return {
        "bookmaker_implied_prob": implied,
        "fair_odds": 1.0 / probability,
        "edge": probability - implied,
        "expected_value": probability * odds - 1.0,
    }


def blend_probabilities(model: dict, market: dict, market_weight: float) -> dict:
    """Shrink model probabilities toward margin-free market probabilities."""
    weight = min(1.0, max(0.0, float(market_weight)))
    blended = dict(model)
    for market_key in ("home_win", "draw", "away_win"):
        if market_key in market and market_key in blended:
            blended[market_key] = (1 - weight) * blended[market_key] + weight * market[market_key]
    total = sum(blended[key] for key in ("home_win", "draw", "away_win"))
    for key in ("home_win", "draw", "away_win"):
        blended[key] /= total
    blended["double_chance_home"] = blended["home_win"] + blended["draw"]
    blended["double_chance_away"] = blended["away_win"] + blended["draw"]
    non_draw = max(blended["home_win"] + blended["away_win"], 1e-12)
    blended["draw_no_bet_home"] = blended["home_win"] / non_draw
    blended["draw_no_bet_away"] = blended["away_win"] / non_draw
    return blended
