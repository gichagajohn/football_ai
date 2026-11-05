import pytest

from backend.models.markets import blend_probabilities, normalized_implied, value_metrics
from backend.models.evaluation import brier_score, flat_stake_roi, log_loss


def test_margin_removal_normalizes_mutually_exclusive_prices():
    probabilities, overround = normalized_implied({"home": 2.0, "draw": 4.0, "away": 4.0})
    assert overround == pytest.approx(1.0)
    assert sum(probabilities.values()) == pytest.approx(1.0)

    probabilities, overround = normalized_implied({"home": 1.9, "draw": 3.4, "away": 4.2})
    assert overround > 1.0
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_expected_value_uses_raw_bookmaker_implied_probability():
    metrics = value_metrics(0.60, 2.0)
    assert metrics["bookmaker_implied_prob"] == pytest.approx(0.50)
    assert metrics["fair_odds"] == pytest.approx(1 / 0.60)
    assert metrics["expected_value"] == pytest.approx(0.20)
    assert value_metrics(0.45, 2.0)["expected_value"] < 0


def test_market_blend_retains_coherent_1x2_derivatives():
    model = {
        "home_win": 0.50, "draw": 0.30, "away_win": 0.20,
        "double_chance_home": 0.80, "double_chance_away": 0.50,
        "draw_no_bet_home": 0.50 / 0.70, "draw_no_bet_away": 0.20 / 0.70,
    }
    market = {"home_win": 0.60, "draw": 0.25, "away_win": 0.15}
    blended = blend_probabilities(model, market, 0.40)
    assert blended["home_win"] == pytest.approx(0.54)
    assert sum(blended[k] for k in ("home_win", "draw", "away_win")) == pytest.approx(1.0)
    assert blended["double_chance_home"] == pytest.approx(blended["home_win"] + blended["draw"])


def test_evaluation_metrics():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert log_loss([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert flat_stake_roi([0.6, 0.4], [2.0, 2.0], [1, 0]) == pytest.approx(0.0)
