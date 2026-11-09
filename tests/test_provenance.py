"""Hard pre-publication provenance gate (Stage A).

These tests pin the invariants that a candidate must satisfy before it can
be selected — and therefore published — by the production pipeline:

  * prediction strictly before kickoff;
  * price timestamp at-or-before kickoff, within the allowed skew of the
    prediction moment;
  * decimal odds > 1 and probabilities in (0, 1);
  * FORCE_PUBLISH semantics: an unprovenanced candidate is simply not a
    candidate, so the optimiser never sees it.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from backend.agents import pipeline_agents as agents
from backend.models.conservative_shadow import build_conservative_shadow
from backend.models.portfolio_optimizer import PortfolioLimits

LIMITS = PortfolioLimits(
    min_selections=1, max_selections=0, combined_odds_min=1.0,
    combined_odds_max=0.0, beam_width=200, max_stake=0.02,
)
from backend.models.provenance import parse_time, validate_candidate
from backend.models.learning import training_examples_from_rows
from backend.capture_closing import closing_price_from_snapshot

NOW = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)
KICKOFF = "2026-11-10T17:30:00Z"
PREDICTION = "2026-11-10T07:00:00Z"
PRICE = "2026-11-10T06:30:00Z"


def test_parse_time_handles_z_and_naive():
    assert parse_time("2026-11-10T07:00:00Z") == datetime(2026, 11, 10, 7, 0, tzinfo=timezone.utc)
    assert parse_time("2026-11-10T07:00:00+03:00") == datetime(2026, 11, 10, 4, 0, tzinfo=timezone.utc)
    assert parse_time("2026-11-10 07:00:00") == datetime(2026, 11, 10, 7, 0, tzinfo=timezone.utc)
    assert parse_time("not a date") is None
    assert parse_time(None) is None


def test_valid_candidate_passes_gate():
    verdict = validate_candidate(
        created_at=PREDICTION,
        kickoff_utc=KICKOFF,
        price_updated_at=PRICE,
        bookmaker_odds=2.1,
        model_prob=0.62,
        market_implied_prob=0.48,
    )
    assert verdict == {"ok": True, "issues": []}


def test_missing_kickoff_is_rejected():
    verdict = validate_candidate(
        created_at=PREDICTION, kickoff_utc=None, price_updated_at=PRICE,
        bookmaker_odds=2.1, model_prob=0.62,
    )
    assert not verdict["ok"]
    assert "kickoff_missing" in verdict["issues"]


def test_prediction_at_or_after_kickoff_is_rejected():
    verdict = validate_candidate(
        created_at=KICKOFF, kickoff_utc=KICKOFF, price_updated_at=PRICE,
        bookmaker_odds=2.1, model_prob=0.62,
    )
    assert "prediction_not_before_kickoff" in verdict["issues"]


def test_price_timestamp_after_prediction_beyond_skew_is_rejected():
    verdict = validate_candidate(
        created_at=PREDICTION, kickoff_utc=KICKOFF,
        price_updated_at="2026-11-10T09:00:00Z",  # two hours after prediction
        bookmaker_odds=2.1, model_prob=0.62,
        max_future_price_skew_seconds=3600,
    )
    assert "price_updated_after_prediction" in verdict["issues"]


def test_price_timestamp_after_kickoff_is_rejected_even_with_huge_skew():
    verdict = validate_candidate(
        created_at=PREDICTION, kickoff_utc=KICKOFF,
        price_updated_at="2026-11-10T18:00:00Z",
        bookmaker_odds=2.1, model_prob=0.62,
        max_future_price_skew_seconds=24 * 3600,
    )
    assert "price_updated_after_kickoff" in verdict["issues"]


def test_small_provider_clock_skew_is_tolerated():
    verdict = validate_candidate(
        created_at=PREDICTION, kickoff_utc=KICKOFF,
        price_updated_at="2026-11-10T07:30:00Z",  # 30 min ahead of prediction
        bookmaker_odds=2.1, model_prob=0.62,
        max_future_price_skew_seconds=3600,
    )
    assert verdict["ok"]


def test_zero_skew_tolerance_rejects_any_future_price():
    verdict = validate_candidate(
        created_at=PREDICTION, kickoff_utc=KICKOFF,
        price_updated_at="2026-11-10T07:00:01Z",
        bookmaker_odds=2.1, model_prob=0.62,
        max_future_price_skew_seconds=0,
    )
    assert "price_updated_after_prediction" in verdict["issues"]


def test_missing_price_timestamp_rejected_only_when_required():
    base = dict(created_at=PREDICTION, kickoff_utc=KICKOFF, price_updated_at=None,
                bookmaker_odds=2.1, model_prob=0.62)
    assert validate_candidate(**base)["ok"]  # default: tolerated, flagged nowhere
    verdict = validate_candidate(**base, require_price_timestamp_flag=True)
    assert "price_updated_at_missing" in verdict["issues"]


@pytest.mark.parametrize("odds,prob", [(1.0, 0.6), (0.9, 0.6), (float("nan"), 0.6), (2.1, 1.0), (2.1, 0.0), (2.1, float("inf"))])
def test_invalid_price_or_probability_is_rejected(odds, prob):
    verdict = validate_candidate(
        created_at=PREDICTION, kickoff_utc=KICKOFF, price_updated_at=PRICE,
        bookmaker_odds=odds, model_prob=prob,
    )
    assert not verdict["ok"]


def test_closure_price_extraction_never_derives():
    assert closing_price_from_snapshot({"home_win": 2.15}, "home_win") == 2.15
    assert closing_price_from_snapshot({"home_win": 1.0}, "home_win") is None
    assert closing_price_from_snapshot({"home_win": "abc"}, "home_win") is None
    assert closing_price_from_snapshot({}, "home_win") is None
    assert closing_price_from_snapshot(None, "home_win") is None
    assert closing_price_from_snapshot({"home_win": float("nan")}, "home_win") is None


def _future_match(fixture_id: int = 1, last_update: str | None = PRICE):
    snapshot = {"home_win": 2.0, "_source": "the-odds-api"}
    if last_update is not None:
        snapshot["_price_sources"] = {
            "home_win": {"bookmaker": "Test Book", "price": 2.0, "last_update": last_update}
        }
    return {
        "fixture_id": fixture_id,
        "home_team": f"Home {fixture_id}",
        "away_team": f"Away {fixture_id}",
        "league": "Test League",
        "kickoff_utc": "2099-01-07T12:00:00Z",
        "markets": {"home_win": 0.70},
        "market_probabilities": {"home_win": 0.5},
        "odds_snapshot": snapshot,
        "model_version": "dixon_coles_v1",
        "feature_snapshot": {"training_matches": 100},
    }


def test_portfolio_rejects_candidate_with_future_price_timestamp():
    portfolio = agents.run_portfolio(
        [_future_match(last_update="2099-01-06T12:00:00Z")],  # >1h beyond prediction time
        reference_time=NOW,
    )
    assert not portfolio.get("selections")
    rejected = next(e for e in portfolio["candidate_evaluations"] if e.get("market") == "home_win")
    assert rejected["eligible"] is False
    assert "price_updated_after_prediction" in (rejected["rejection_reason"] or "")
    assert rejected["provenance_ok"] is False


def test_portfolio_accepts_candidate_with_clean_price_provenance():
    portfolio = agents.run_portfolio([_future_match()], reference_time=NOW)
    assert len(portfolio.get("selections") or []) == 1


def test_provenance_failure_blocks_force_publish_path():
    """FORCE_PUBLISH bypasses Auditor warnings only; provenance blocks selection."""
    portfolio = agents.run_portfolio(
        [_future_match(last_update="2099-01-06T12:00:00Z")],
        reference_time=NOW,
    )
    assert portfolio.get("decision") == "NO_BET" or not portfolio.get("selections")
    for evaluation in portfolio["candidate_evaluations"]:
        if evaluation.get("provenance_issues"):
            assert evaluation["eligible"] is False
            assert evaluation["selected"] is False


def test_shadow_excludes_unprovenanced_candidate():
    evaluation = {
        "fixture_id": 1, "home_team": "H", "away_team": "A", "league": "L",
        "kickoff_utc": "2099-01-07T12:00:00Z", "market": "home_win",
        "model_version": "test", "raw_model_prob": 0.70, "model_prob": 0.70,
        "bookmaker_odds": 2.0, "market_implied_prob": 0.50,
        "expected_value": 0.40, "odds_source": "bookmaker",
        "feature_snapshot": {
            "price_sources": {"home_win": {"last_update": "2099-01-06T12:00:00Z"}},
        },
    }
    shadow = build_conservative_shadow([evaluation], LIMITS, reference_time=NOW)
    shadow_eval = shadow["candidate_evaluations"][0]
    assert shadow_eval["eligible"] is False
    assert "provenance" in (shadow_eval.get("rejection_reason") or "").lower()


def test_training_examples_dedupe_reruns_keep_latest():
    logs = [
        {"ticket_date": "2026-11-06", "fixture_id": 1, "market": "home_win",
         "model_prob": 0.60, "selected": True, "created_at": "2026-11-06T04:00:00+00:00"},
        {"ticket_date": "2026-11-06", "fixture_id": 1, "market": "home_win",
         "model_prob": 0.65, "selected": True, "created_at": "2026-11-06T09:30:00+00:00"},
        {"ticket_date": "2026-11-07", "fixture_id": 2, "market": "draw",
         "model_prob": 0.40, "selected": True, "created_at": "2026-11-07T04:00:00+00:00"},
    ]
    selections = [
        {"ticket_date": "2026-11-06", "fixture_id": 1, "market": "home_win", "outcome": "win"},
        {"ticket_date": "2026-11-07", "fixture_id": 2, "market": "draw", "outcome": "loss"},
    ]
    examples = training_examples_from_rows(logs, selections)
    assert len(examples) == 2
    by_date = {str(row["ticket_date"]): row["probability"] for row in examples}
    assert by_date["2026-11-06"] == pytest.approx(0.65)  # latest rerun wins, not double-counted
