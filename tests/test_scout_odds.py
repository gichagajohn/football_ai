import logging

import pytest

from backend.agents.scout_agent import (
    _TEAM_NAME_ALIASES,
    _canonical_team_name,
    _extract_odds_snapshot,
    _find_match_odds,
    _log_unmatched_odds,
)


def test_best_prices_and_two_sided_markets_are_extracted():
    event = {
        "home_team": "Home FC",
        "away_team": "Away FC",
        "bookmakers": [
            {
                "title": "Book A", "last_update": "2026-11-04T05:00:00Z",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Home FC", "price": 2.0},
                        {"name": "Draw", "price": 3.3},
                        {"name": "Away FC", "price": 4.0},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 1.9},
                        {"name": "Under", "point": 2.5, "price": 1.9},
                    ]},
                ],
            },
            {
                "title": "Book B", "last_update": "2026-11-04T05:01:00Z",
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Home FC", "price": 2.1}]},
                    {"key": "btts", "outcomes": [
                        {"name": "Yes", "price": 1.8}, {"name": "No", "price": 2.0},
                    ]},
                ],
            },
        ],
    }
    snapshot = _extract_odds_snapshot(event)
    assert snapshot["home_win"] == pytest.approx(2.1)
    assert snapshot["away_win"] == pytest.approx(4.0)
    assert snapshot["under25"] == pytest.approx(1.9)
    assert snapshot["btts_yes"] == pytest.approx(1.8)
    assert snapshot["btts_no"] == pytest.approx(2.0)
    assert snapshot["_price_sources"]["home_win"]["bookmaker"] == "Book B"
    assert snapshot["_source"] == "the-odds-api"


@pytest.mark.parametrize(
    ("football_data_name", "odds_api_name"),
    [
        ("Clube de Regatas do Flamengo", "Flamengo"),
        ("Clube Atlético Mineiro", "Atletico-MG"),
        ("Sport Club Corinthians Paulista", "Corinthians"),
        ("Grêmio Foot-Ball Porto Alegrense", "Gremio"),
        ("Club Athletico Paranaense", "Athletico-PR"),
        ("FC Bayern München", "Bayern Munich"),
        ("Nott'm Forest", "Nottingham Forest"),
    ],
)
def test_cross_provider_team_names_match_conservatively(football_data_name, odds_api_name):
    event = {"home_team": odds_api_name, "away_team": "Opponent"}
    assert _find_match_odds([event], football_data_name, "Opponent") == event


def test_milan_alias_protection_is_not_relaxed():
    event = {"home_team": "Inter Milan", "away_team": "Opponent"}
    assert _find_match_odds([event], "AC Milan", "Opponent") == {}


def test_every_existing_alias_still_matches_its_canonical_value():
    for variant, canonical in _TEAM_NAME_ALIASES.items():
        assert _canonical_team_name(variant) == _canonical_team_name(canonical), variant


def test_unmatched_diagnostic_lists_fixture_and_provider_candidates(caplog):
    fixture = {
        "homeTeam": {"name": "Clube Atlético Mineiro"},
        "awayTeam": {"name": "Flamengo"},
        "utcDate": "2026-11-05T22:00:00Z",
    }
    event = {
        "home_team": "Atletico-MG",
        "away_team": "Flamengo",
        "commence_time": "2026-11-05T22:00:00Z",
    }
    with caplog.at_level(logging.WARNING):
        _log_unmatched_odds("BSA", [fixture], [event])
    assert "Clube Atlético Mineiro vs Flamengo" in caplog.text
    assert "Atletico-MG vs Flamengo" in caplog.text
    assert "2026-11-05T22:00:00Z" in caplog.text
    assert "atletico mineiro / flamengo" in caplog.text
