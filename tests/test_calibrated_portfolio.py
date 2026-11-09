import pytest

from backend.agents.pipeline_agents import run_portfolio


def test_portfolio_uses_guarded_calibrated_probability_and_keeps_raw_value():
    match = {
        "fixture_id": 1, "home_team": "H", "away_team": "A", "league": "L",
        "markets": {"home_win": 0.60}, "odds_snapshot": {"home_win": 1.80},
        "kickoff_utc": "2099-11-09T12:00:00Z",
        "model_version": "dixon_coles_v1",
        "probability_calibration": {
            "active": True, "version": "guarded_platt_v1", "intercept": 1.0, "slope": 1.0,
        },
        "feature_snapshot": {
            "training_matches": 100, "home_team_history": 20, "away_team_history": 20,
            "probability_calibration": {"active": True},
        },
    }
    portfolio = run_portfolio([match])
    selection = portfolio["selections"][0]
    assert selection["raw_model_prob"] == pytest.approx(0.60)
    assert selection["model_prob"] > 0.60
    evaluation = next(e for e in portfolio["candidate_evaluations"] if e["market"] == "home_win")
    assert evaluation["raw_model_prob"] == pytest.approx(0.60)
    assert evaluation["model_prob"] == pytest.approx(selection["model_prob"])
    assert evaluation["calibration_version"] == "guarded_platt_v1"
