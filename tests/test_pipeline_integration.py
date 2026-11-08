import asyncio
from dataclasses import dataclass
from datetime import date
from unittest.mock import AsyncMock, patch

from backend import pipeline
from backend.agents import pipeline_agents as agents


@dataclass
class PipelineModel:
    training_matches: int = 120
    time_decay_days: float = 240.0
    rho: float = -0.04
    home_advantage: float = 0.2
    matches_per_team: dict = None

    def __post_init__(self):
        self.matches_per_team = {10: 20, 20: 20}

    def probabilities(self, home_id, away_id):
        return {
            "home_win": 0.65, "draw": 0.20, "away_win": 0.15,
            "btts_yes": 0.55, "btts_no": 0.45,
            "over25": 0.52, "under25": 0.48,
            "over15": 0.75, "under45": 0.90,
            "double_chance_home": 0.85, "double_chance_away": 0.35,
            "draw_no_bet_home": 0.65 / 0.80, "draw_no_bet_away": 0.15 / 0.80,
            "expected_home_goals": 1.8, "expected_away_goals": 0.9,
        }


def test_full_pipeline_can_store_a_valid_single(monkeypatch):
    match = {
        "fixture_id": 999,
        "home_team": "Home",
        "away_team": "Away",
        "home_team_id": 10,
        "away_team_id": 20,
        "competition_code": "PL",
        "league": "Premier League",
        "kickoff_utc": "2026-11-05T15:00:00Z",
        "data_completeness": 1.0,
        "odds_snapshot": {"home_win": 2.0, "draw": 4.0, "away_win": 5.0},
        "recent_form": {"home": {}, "away": {}},
        "standings": {"home": {}, "away": {}},
        "injuries": {"home": [], "away": []},
    }
    context = {"models": {"PL": PipelineModel()}}
    monkeypatch.setattr(agents, "FORCE_PUBLISH", True)

    with (
        patch.object(pipeline, "run_scout", new=AsyncMock(return_value=[match])),
        patch.object(pipeline, "build_poisson_context", new=AsyncMock(return_value=context)),
        patch.object(pipeline, "load_probability_calibrator", return_value={"active": False, "version": "test"}),
        patch.object(pipeline, "run_risk_filter", side_effect=lambda probabilities, intelligence: probabilities),
        patch.object(pipeline, "run_auditor", return_value={"auditor_verdict": "REJECT", "critical_flags": ["test"]}),
        patch.object(pipeline, "run_publisher", return_value="Single ticket"),
        patch.object(pipeline.supabase_client, "store_ticket") as store,
    ):
        ticket = asyncio.run(pipeline.run_pipeline(date(2026, 11, 5)))

    assert ticket == "Single ticket"
    store.assert_called_once()
    portfolio = store.call_args.args[2]
    decision = store.call_args.args[3]
    assert len(portfolio["selections"]) == 1
    assert decision["decision"] == "PUBLISH"
    assert portfolio["selections"][0]["model_version"] == "dixon_coles_v1"
    assert store.call_args.kwargs["shadow_portfolio"]["is_shadow"] is True


def test_shadow_failure_does_not_block_official_pipeline(monkeypatch):
    match = {
        "fixture_id": 999, "home_team": "Home", "away_team": "Away",
        "home_team_id": 10, "away_team_id": 20, "competition_code": "PL",
        "league": "Premier League", "kickoff_utc": "2026-11-05T15:00:00Z",
        "data_completeness": 1.0,
        "odds_snapshot": {"home_win": 2.0, "draw": 4.0, "away_win": 5.0},
        "recent_form": {"home": {}, "away": {}},
        "standings": {"home": {}, "away": {}}, "injuries": {"home": [], "away": []},
    }
    context = {"models": {"PL": PipelineModel()}}
    monkeypatch.setattr(agents, "FORCE_PUBLISH", True)
    with (
        patch.object(pipeline, "run_scout", new=AsyncMock(return_value=[match])),
        patch.object(pipeline, "build_poisson_context", new=AsyncMock(return_value=context)),
        patch.object(pipeline, "load_probability_calibrator", return_value={"active": False}),
        patch.object(pipeline, "run_risk_filter", side_effect=lambda probabilities, intelligence: probabilities),
        patch.object(pipeline, "run_conservative_shadow", side_effect=RuntimeError("shadow failed")),
        patch.object(pipeline, "run_auditor", return_value={"auditor_verdict": "APPROVE", "critical_flags": []}),
        patch.object(pipeline, "run_publisher", return_value="Official ticket"),
        patch.object(pipeline.supabase_client, "store_ticket") as store,
    ):
        ticket = asyncio.run(pipeline.run_pipeline(date(2026, 11, 5)))
    assert ticket == "Official ticket"
    store.assert_called_once()
    assert store.call_args.kwargs["shadow_portfolio"] is None
    assert store.call_args.args[2]["selections"]


def test_llm_portfolio_excludes_private_candidate_evaluations():
    portfolio = agents.run_portfolio([{
        "fixture_id": 1, "home_team": "H", "away_team": "A", "league": "L",
        "markets": {"home_win": 0.7}, "odds_snapshot": {"home_win": 1.6},
        "model_version": "dixon_coles_v1", "feature_snapshot": {"training_matches": 100},
    }])
    compact = agents._portfolio_for_llm(portfolio)
    assert "candidate_evaluations" not in compact
    assert "feature_snapshot" not in compact["selections"][0]
    assert portfolio["candidate_evaluations"]
