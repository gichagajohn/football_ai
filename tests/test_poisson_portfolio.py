import math

import pytest

from backend.agents import pipeline_agents as agents


def make_match(fixture_id=1, probability=0.70, odds=1.60):
    return {
        "fixture_id": fixture_id,
        "home_team": f"Home {fixture_id}",
        "away_team": f"Away {fixture_id}",
        "league": "Test League",
        "markets": {"home_win": probability},
        "market_probabilities": {"home_win": 1 / odds},
        "odds_snapshot": {"home_win": odds},
        "model_version": "dixon_coles_v1",
        "feature_snapshot": {
            "training_matches": 100, "home_team_history": 20,
            "away_team_history": 20, "expected_home_goals": 1.5,
        },
    }


def test_single_strong_selection_is_allowed():
    portfolio = agents.run_portfolio([make_match()])
    assert "decision" not in portfolio
    assert len(portfolio["selections"]) == 1
    assert portfolio["combined_odds"] == pytest.approx(1.60)
    assert portfolio["portfolio_confidence"] == pytest.approx(0.70)
    assert agents.validate_portfolio(portfolio) == []


def test_no_bet_when_all_candidates_have_negative_ev():
    portfolio = agents.run_portfolio([make_match(probability=0.55, odds=1.70)])
    assert portfolio["decision"] == "NO_BET"
    assert not portfolio.get("selections")
    rejected = [e for e in portfolio["candidate_evaluations"] if e.get("bookmaker_odds")]
    assert any("Negative expected value" in (e.get("rejection_reason") or "") for e in rejected)


def test_negative_ev_match_is_never_added_as_filler():
    portfolio = agents.run_portfolio([
        make_match(1, probability=0.70, odds=1.60),
        make_match(2, probability=0.55, odds=1.70),
    ])
    assert len(portfolio["selections"]) == 1
    assert portfolio["selections"][0]["fixture_id"] == 1


def test_six_legs_and_twenty_odds_are_not_hard_limits():
    matches = [make_match(i, probability=0.66, odds=1.80) for i in range(1, 9)]
    portfolio = agents.run_portfolio(matches)
    assert len(portfolio["selections"]) > 6
    assert len(portfolio["selections"]) <= len(matches)
    assert portfolio["combined_odds"] > 20.0
    assert math.prod(s["odds"] for s in portfolio["selections"]) == pytest.approx(
        portfolio["combined_odds"], abs=0.011
    )
    assert len({s["fixture_id"] for s in portfolio["selections"]}) == len(portfolio["selections"])
    assert all(s["expected_value"] >= 0 for s in portfolio["selections"])
    assert agents.validate_portfolio(portfolio) == []


def test_only_one_market_per_fixture_survives():
    match = make_match(1, probability=0.65, odds=1.80)
    match["markets"]["double_chance_home"] = 0.85
    match["odds_snapshot"].update({"draw": 3.8})
    portfolio = agents.run_portfolio([match])
    assert len(portfolio["selections"]) == 1


def test_force_publish_bypasses_auditor_only_for_valid_portfolio(monkeypatch):
    monkeypatch.setattr(agents, "FORCE_PUBLISH", True)
    portfolio = agents.run_portfolio([make_match()])
    decision = agents.run_decision({"auditor_verdict": "REJECT", "critical_flags": ["warning"]}, portfolio)
    assert decision["decision"] == "PUBLISH"
    assert decision["final_confidence"] == pytest.approx(0.70)


@pytest.mark.parametrize("tamper", ["negative_ev", "nonfinite_odds", "duplicate_fixture", "missing_metadata", "insufficient_history"])
def test_force_publish_never_bypasses_hard_gates(monkeypatch, tamper):
    monkeypatch.setattr(agents, "FORCE_PUBLISH", True)
    portfolio = agents.run_portfolio([make_match()])
    if tamper == "negative_ev":
        portfolio["selections"][0]["model_prob"] = 0.40
    elif tamper == "nonfinite_odds":
        portfolio["selections"][0]["odds"] = float("inf")
        portfolio["combined_odds"] = float("inf")
    elif tamper == "duplicate_fixture":
        portfolio["selections"] = [portfolio["selections"][0], dict(portfolio["selections"][0])]
        portfolio["combined_odds"] = 1.60 ** 2
    elif tamper == "missing_metadata":
        portfolio["selections"][0]["feature_snapshot"] = {}
    elif tamper == "insufficient_history":
        portfolio["selections"][0]["feature_snapshot"]["home_team_history"] = 1

    decision = agents.run_decision({"auditor_verdict": "APPROVE", "critical_flags": []}, portfolio)
    assert decision["decision"] == "NO_BET"
    assert "Hard portfolio validation failed" in decision["reason"]
