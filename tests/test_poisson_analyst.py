from dataclasses import dataclass

import pytest

from backend.agents.pipeline_agents import run_analyst


@dataclass
class StubModel:
    training_matches: int = 100
    time_decay_days: float = 240.0
    rho: float = -0.05
    home_advantage: float = 0.2
    matches_per_team: dict = None

    def __post_init__(self):
        self.matches_per_team = {10: 20, 20: 20}

    def probabilities(self, home_id, away_id):
        if (home_id, away_id) != (10, 20):
            raise KeyError("Unknown team")
        return {
            "home_win": 0.50, "draw": 0.30, "away_win": 0.20,
            "btts_yes": 0.55, "btts_no": 0.45,
            "over25": 0.52, "under25": 0.48,
            "over15": 0.75, "under45": 0.90,
            "double_chance_home": 0.80, "double_chance_away": 0.50,
            "draw_no_bet_home": 0.50 / 0.70, "draw_no_bet_away": 0.20 / 0.70,
            "expected_home_goals": 1.6, "expected_away_goals": 1.0,
        }


def test_analyst_uses_fitted_probabilities_and_market_margin_baseline():
    match = {
        "fixture_id": 999,
        "home_team": "Home",
        "away_team": "Away",
        "home_team_id": 10,
        "away_team_id": 20,
        "competition_code": "PL",
        "league": "Premier League",
        "odds_snapshot": {"home_win": 2.0, "draw": 4.0, "away_win": 5.0},
        "odds_source": "the-odds-api",
        "odds_bookmakers": ["Book"],
    }
    result = run_analyst([match], {"models": {"PL": StubModel()}})
    assert len(result) == 1
    item = result[0]
    assert item["fixture_id"] == 999
    assert item["home_team_id"] == 10
    assert item["model_version"] == "dixon_coles_v1"
    assert item["poisson_markets"]["home_win"] == pytest.approx(0.50)
    # Market baseline: raw implied 0.5/0.25/0.2, overround 0.95.
    # At 35% weight, home becomes 0.65*0.50 + 0.35*(0.5/0.95).
    assert item["markets"]["home_win"] == pytest.approx(0.5092105, abs=1e-6)
    assert sum(item["markets"][k] for k in ("home_win", "draw", "away_win")) == pytest.approx(1.0)
    assert item["feature_snapshot"]["training_matches"] == 100
    assert item["feature_snapshot"]["home_team_history"] == 20


def test_analyst_skips_missing_team_history_instead_of_guessing():
    match = {
        "fixture_id": 1, "home_team": "New", "away_team": "Away",
        "home_team_id": 999, "away_team_id": 20, "competition_code": "PL",
        "odds_snapshot": {},
    }
    assert run_analyst([match], {"models": {"PL": StubModel()}}) == []
