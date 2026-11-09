from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from backend.agents.pipeline_agents import _derive_odds
from backend.agents.scout_agent import _extract_odds_snapshot
from backend.db import supabase_client
from backend.model_report import summarize_strategy_predictions
from backend import result_checker
from backend.models.conservative_shadow import SHADOW_STRATEGY, build_conservative_shadow
from backend.models.portfolio_backtesting import (
    settle_accumulator,
    summarize_bet_records,
    walk_forward_portfolio_backtest,
)
from backend.models.portfolio_optimizer import PortfolioLimits

LIMITS = PortfolioLimits(
    min_selections=1, max_selections=0, combined_odds_min=1.0,
    combined_odds_max=0.0, beam_width=200, max_stake=0.02,
)


def test_additional_markets_require_direct_executable_prices():
    odds = {"home_win": 2.0, "draw": 3.4, "away_win": 4.0}
    assert _derive_odds("draw_no_bet_home", odds) is None
    assert _derive_odds("double_chance_home", odds) is None
    assert _derive_odds("home_win", {"home_win": float("nan")}) is None
    assert _derive_odds("home_win", {"home_win": float("inf")}) is None
    assert _derive_odds("draw_no_bet_home", {**odds, "draw_no_bet_home": 1.44}) == 1.44
    assert _derive_odds("double_chance_home", {**odds, "double_chance_home": 1.25}) == 1.25


def test_provider_parser_retains_only_direct_additional_market_quotes():
    event = {
        "home_team": "Arsenal", "away_team": "Liverpool",
        "bookmakers": [{
            "title": "Smoke Book", "last_update": "2026-11-08T10:00:00Z",
            "markets": [
                {"key": "draw_no_bet", "outcomes": [
                    {"name": "Arsenal", "price": 1.50},
                    {"name": "Liverpool", "price": 2.50},
                ]},
                {"key": "double_chance", "outcomes": [
                    {"name": "Arsenal or Draw", "price": 1.20},
                    {"name": "Liverpool or Draw", "price": 1.30},
                ]},
            ],
        }],
    }
    snapshot = _extract_odds_snapshot(event)
    assert snapshot["draw_no_bet_home"] == 1.50
    assert snapshot["draw_no_bet_away"] == 2.50
    assert snapshot["double_chance_home"] == 1.20
    assert snapshot["double_chance_away"] == 1.30
    assert snapshot["_price_sources"]["draw_no_bet_home"]["bookmaker"] == "Smoke Book"


def test_shadow_requires_genuine_prices_and_uses_fifty_fifty_anchor():
    evaluations = [{
        "fixture_id": 1, "home_team": "H", "away_team": "A", "league": "L",
        "kickoff_utc": "2099-11-09T12:00:00+00:00", "market": "home_win",
        "model_version": "test", "raw_model_prob": 0.70, "model_prob": 0.70,
        "bookmaker_odds": 2.0, "market_implied_prob": 0.50,
        "expected_value": 0.40, "odds_source": "bookmaker",
        "feature_snapshot": {},
    }]
    shadow = build_conservative_shadow(evaluations, LIMITS)
    assert shadow["strategy"] == SHADOW_STRATEGY
    row = shadow["candidate_evaluations"][0]
    assert row["robust_model_prob"] == pytest.approx(0.60)
    assert row["robust_expected_value"] == pytest.approx(0.20)
    assert row["shadow_selected"] is True
    assert row["selected"] is False

    derived = build_conservative_shadow(
        [{**evaluations[0], "odds_source": "derived_from_bookmaker_1x2"}],
        LIMITS,
    )
    assert derived["selections"] == []
    assert "genuine bookmaker price" in derived["candidate_evaluations"][0]["rejection_reason"]
    invalid = build_conservative_shadow(
        [{**evaluations[0], "bookmaker_odds": float("nan")}], LIMITS
    )
    assert invalid["selections"] == []
    assert invalid["candidate_evaluations"][0]["eligible"] is False


def _single(outcome, odds=2.0, fixture_id=1, closing=1.9):
    gross = odds if outcome == "win" else 1.0 if outcome == "void" else 0.0
    return {
        "strategy": "test", "date": "2026-11-08", "fixture_id": fixture_id,
        "fixture_ids": [fixture_id], "league": "L", "market": "draw_no_bet_home",
        "odds": odds, "closing_odds": closing, "outcome": outcome,
        "return": gross, "profit": gross - 1.0, "sequence": fixture_id,
    }


def test_dnb_push_reduces_accumulator_and_all_push_returns_stake():
    acca = settle_accumulator([_single("void"), _single("win", 3.0, 2)])
    assert acca["outcome"] == "win"
    assert acca["odds"] == pytest.approx(6.0)
    assert acca["effective_odds"] == pytest.approx(3.0)
    assert acca["return"] == pytest.approx(3.0)
    assert acca["profit"] == pytest.approx(2.0)
    assert acca["push_legs"] == 1

    pushes = settle_accumulator([_single("void"), _single("void", 2.5, 2)])
    assert pushes["outcome"] == "void"
    assert pushes["return"] == 1.0
    assert pushes["profit"] == 0.0


def test_betting_metrics_include_drawdown_clv_and_unique_fixtures():
    rows = [_single("win", fixture_id=1), _single("loss", fixture_id=2)]
    report = summarize_bet_records(rows)
    assert report["bets"] == 2
    assert report["wins"] == 1
    assert report["losses"] == 1
    assert report["hit_rate"] == 0.5
    assert report["roi"] == pytest.approx(0.0)
    assert report["max_drawdown"] == pytest.approx(1.0)
    assert report["unique_fixtures"] == 2
    assert report["clv_bets"] == 2
    assert report["average_clv"] == pytest.approx(2.0 / 1.9 - 1.0)
    assert report["by_odds_band"]["2.00-2.99"]["bets"] == 2


def _match(identifier, day, home=1, away=2, home_goals=1, away_goals=0):
    return {
        "id": identifier,
        "utcDate": day.isoformat() + "T12:00:00+00:00",
        "competition": {"code": "TEST", "name": "Test League"},
        "homeTeam": {"id": home, "name": f"Team {home}"},
        "awayTeam": {"id": away, "name": f"Team {away}"},
        "score": {"fullTime": {"home": home_goals, "away": away_goals}},
        "odds_snapshot": {"home_win": 2.0, "draw": 4.0, "away_win": 4.0},
        "closing_odds_snapshot": {"home_win": 1.8 if identifier == 100 else 2.2},
    }


def test_portfolio_walk_forward_has_no_future_fixtures_in_fit(monkeypatch):
    import backend.models.portfolio_backtesting as module

    start = date(2026, 1, 1)
    matches = []
    for index in range(24):
        home, away = ((index % 4) + 1, ((index + 1) % 4) + 1)
        if home == away:
            away = away % 4 + 1
        matches.append(_match(index + 1, start, home, away))
    matches.extend([
        _match(100, start + timedelta(days=1), home_goals=2, away_goals=0),
        _match(101, start + timedelta(days=2), home_goals=0, away_goals=1),
    ])
    fit_calls = []

    class FakeModel:
        def __init__(self, ids):
            self.ids = ids

    def fake_fit(history, as_of, time_decay_days):
        ids = [row["id"] for row in history]
        assert all(datetime.fromisoformat(row["utcDate"].replace("Z", "+00:00")) < as_of for row in history)
        fit_calls.append((ids, as_of))
        return FakeModel(ids)

    def fake_analyst(clean_matches, context):
        analyzed = []
        for match in clean_matches:
            assert all(
                fixture_id not in context["models"]["TEST"].ids
                for fixture_id in ([100] if match["fixture_id"] == 100 else [101])
            )
            analyzed.append({
                **match,
                "markets": {
                    "home_win": 0.70, "draw": 0.20, "away_win": 0.10,
                    "double_chance_home": 0.90, "double_chance_away": 0.30,
                    "draw_no_bet_home": 0.70 / 0.80, "draw_no_bet_away": 0.10 / 0.80,
                },
                "market_probabilities": {
                    "home_win": 0.50, "draw": 0.25, "away_win": 0.25,
                    "double_chance_home": 0.75, "double_chance_away": 0.50,
                    "draw_no_bet_home": 0.50 / 0.75, "draw_no_bet_away": 0.25 / 0.75,
                },
                "model_version": "fake_dixon_coles",
                "probability_calibration": context["calibration"],
                "feature_snapshot": {"probability_calibration": context["calibration"]},
            })
        return analyzed

    monkeypatch.setattr(module, "fit_dixon_coles", fake_fit)
    monkeypatch.setattr(module, "run_analyst", fake_analyst)
    report = walk_forward_portfolio_backtest(matches, initial_train=24, include_records=True)

    assert report["test_days"] == 2
    assert len(fit_calls) == 2
    assert 100 not in fit_calls[0][0] and 101 not in fit_calls[0][0]
    assert 100 in fit_calls[1][0] and 101 not in fit_calls[1][0]
    singles = report["betting"]["production_singles"]
    assert singles["bets"] == 2
    assert singles["wins"] == 1 and singles["losses"] == 1
    assert singles["roi"] == pytest.approx(0.0)
    assert singles["clv_bets"] == 2
    assert report["betting"]["production_accumulator"]["bets"] == 2
    assert report["betting"]["robust_conservative_v1_shadow_singles"]["bets"] == 2
    assert report["betting"]["bookmaker_consensus_1x2_singles"]["bets"] == 2
    assert report["probability"]["production_model_all_candidates"]["brier_score"] is not None
    assert "never activate" in report["methodology"]["activation"]


def test_shadow_storage_marks_private_selection_without_official_selected_flag():
    response = Mock()
    response.raise_for_status.return_value = None
    shadow = {
        "strategy": SHADOW_STRATEGY,
        "config": {"model_weight": 0.5, "market_weight": 0.5},
        "candidate_evaluations": [{
            "fixture_id": 1, "home_team": "H", "away_team": "A", "league": "L",
            "kickoff_utc": "2099-11-09T12:00:00+00:00", "market": "home_win",
            "model_version": "test", "raw_model_prob": 0.7, "model_prob": 0.7,
            "robust_model_prob": 0.6, "robust_expected_value": 0.2,
            "bookmaker_odds": 2.0, "odds_source": "bookmaker",
            "shadow_selected": True, "selected": False,
        }],
    }
    with patch.object(supabase_client.httpx, "post", return_value=response) as post:
        supabase_client._store_shadow_prediction_logs(date(2026, 11, 8), shadow)
    row = post.call_args.kwargs["json"][0]
    assert row["strategy"] == SHADOW_STRATEGY
    assert row["shadow_selected"] is True
    assert row["selected"] is False
    assert row["model_version"].endswith("__shadow_robust_conservative_v1")
    assert row["robust_model_prob"] == pytest.approx(0.6)


def test_result_checker_settles_shadow_logs_without_pending_official_ticket(monkeypatch):
    monkeypatch.setattr(supabase_client, "get_pending_tickets", lambda before_date: [])
    monkeypatch.setattr(supabase_client, "get_pending_model_logs", lambda before_date: [{
        "id": 55, "fixture_id": 100, "market": "draw_no_bet_home",
        "strategy": SHADOW_STRATEGY,
    }])
    monkeypatch.setattr(result_checker, "fetch_result", lambda fixture_id: {
        "home_goals": 1, "away_goals": 1, "status": "FINISHED",
    })
    settled = []
    monkeypatch.setattr(
        supabase_client, "update_model_log_outcome",
        lambda log_id, outcome, home_score, away_score: settled.append(
            (log_id, outcome, home_score, away_score)
        ) or True,
    )
    result_checker.run(today=date(2026, 11, 9))
    assert settled == [(55, "void", 1, 1)]


def test_settled_strategy_report_compares_production_and_private_shadow():
    common = {
        "ticket_date": "2026-11-08", "fixture_id": 1, "home_team": "H",
        "away_team": "A", "league": "L", "market": "home_win",
        "model_prob": 0.70, "market_implied_prob": 0.50,
        "bookmaker_odds": 2.0, "closing_bookmaker_odds": 1.9,
        "outcome": "win",
    }
    logs = [
        {**common, "strategy": "production", "selected": True, "shadow_selected": False},
        {
            **common, "strategy": SHADOW_STRATEGY, "selected": False,
            "shadow_selected": True, "robust_model_prob": 0.60,
        },
    ]
    report = summarize_strategy_predictions(logs)
    production = report["production"]
    shadow = report[SHADOW_STRATEGY]
    assert production["singles"]["wins"] == 1
    assert production["singles"]["average_clv"] == pytest.approx(2.0 / 1.9 - 1.0)
    assert production["accumulator"]["bets"] == 1
    assert "not necessarily published" in production["scope"]
    assert shadow["selected_robust_probability"]["predictions"] == 1
    assert shadow["selected_robust_probability"]["brier_score"] == pytest.approx(0.16)
    assert shadow["singles"]["roi"] == pytest.approx(1.0)
    assert "never an official publication" in shadow["scope"]
