import pytest

from backend.agents.scout_agent import _extract_odds_snapshot


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
