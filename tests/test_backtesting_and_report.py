from datetime import date, timedelta

import numpy as np
import pytest

from backend.model_report import summarize_predictions
from backend.models.backtesting import walk_forward_backtest


def test_walk_forward_backtest_produces_metrics_without_future_data():
    rng = np.random.default_rng(81)
    matches = []
    sequence = 0
    start = date(2025, 1, 1)
    for _ in range(2):
        for home in range(1, 7):
            for away in range(1, 7):
                if home == away:
                    continue
                matches.append({
                    "utcDate": (start + timedelta(days=sequence)).isoformat() + "T12:00:00+00:00",
                    "homeTeam": {"id": home, "name": str(home)},
                    "awayTeam": {"id": away, "name": str(away)},
                    "score": {"fullTime": {
                        "home": int(rng.poisson(1.5)), "away": int(rng.poisson(1.1)),
                    }},
                })
                sequence += 1
    report = walk_forward_backtest(matches, initial_train=30, retrain_every=30)
    assert report["predictions"] == 30
    assert report["model_refits"] == 1
    assert 0 <= report["accuracy"] <= 1
    assert report["multiclass_brier_score"] >= 0
    assert report["multiclass_log_loss"] >= 0
    assert report["value_bets"] == 0


def test_settled_log_calibration_report():
    logs = [
        {"ticket_date": "2026-11-01", "fixture_id": 1, "market": "home_win", "selected": True,
         "model_prob": 0.8, "bookmaker_odds": 1.5},
        {"ticket_date": "2026-11-02", "fixture_id": 2, "market": "away_win", "selected": True,
         "model_prob": 0.6, "bookmaker_odds": 2.0},
        {"ticket_date": "2026-11-03", "fixture_id": 3, "market": "draw", "selected": True,
         "model_prob": 0.5, "bookmaker_odds": 3.0},
    ]
    selections = [
        {"ticket_date": "2026-11-01", "fixture_id": 1, "market": "home_win", "outcome": "win"},
        {"ticket_date": "2026-11-02", "fixture_id": 2, "market": "away_win", "outcome": "loss"},
    ]
    report = summarize_predictions(logs, selections)
    assert report["settled_predictions"] == 2
    assert report["wins"] == 1
    assert report["pending_or_unmatched"] == 1
    assert report["hit_rate"] == pytest.approx(0.5)
    assert report["flat_stake_roi"] == pytest.approx(-0.25)
    assert report["brier_score"] >= 0
    assert report["sample_warning"]
