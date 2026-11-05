import asyncio
from datetime import date
from unittest.mock import AsyncMock, Mock, patch

from backend.agents import scout_agent
from backend.models import poisson_service


def test_many_competitions_and_unlimited_scout_are_default():
    expected = {"PL", "ELC", "FAC", "PD", "CDR", "BL1", "DFB", "SA", "CIT", "FL1", "DED", "PPL", "BSA", "CL", "EL", "UCL"}
    assert expected <= set(scout_agent.DEFAULT_LEAGUE_IDS)
    assert expected <= set(scout_agent.ODDS_SPORT_KEYS)
    assert scout_agent.HARD_CAP_MATCHES == 0


def test_calendar_year_and_european_seasons_are_distinguished():
    assert poisson_service.season_start_year(date(2026, 11, 5), "PL") == 2026
    assert poisson_service.season_start_year(date(2027, 2, 1), "PL") == 2026
    assert poisson_service.season_start_year(date(2027, 2, 1), "BSA") == 2027


def test_domestic_cup_loads_three_seasons_of_history():
    fixture = {"competition_code": "DFB"}
    with patch.object(poisson_service, "_fd_get", new=AsyncMock(return_value={"matches": []})) as get:
        context = asyncio.run(poisson_service.build_poisson_context([fixture], date(2026, 11, 5)))
    seasons = [call.kwargs["params"]["season"] for call in get.call_args_list]
    assert seasons == [2026, 2025, 2024]
    assert context["models"] == {}


def test_odds_catalog_resolves_provider_specific_aliases():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = [
        {"key": "soccer_fa_cup", "active": True},
        {"key": "soccer_efl_champ", "active": True},
        {"key": "soccer_epl", "active": True},
    ]
    client = AsyncMock()
    client.get.return_value = response
    client.__aenter__.return_value = client
    with patch.object(scout_agent.httpx, "AsyncClient", return_value=client):
        resolved = asyncio.run(scout_agent.resolve_odds_sport_keys({"FAC", "ELC", "PL"}))
    assert resolved == {
        "FAC": "soccer_fa_cup", "ELC": "soccer_efl_champ", "PL": "soccer_epl",
    }
