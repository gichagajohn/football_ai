import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.agents import scout_agent


@pytest.mark.parametrize(
    ("team_name", "expected_city"),
    [
        ("SE Palmeiras", "São Paulo"),
        ("Clube do Remo", "Belém"),
        ("CA Mineiro", "Belo Horizonte"),
        ("EC Vitória", "Salvador"),
        ("Botafogo FR", "Rio de Janeiro"),
        ("Botafogo SP", "Ribeirão Preto"),
        ("Mirassol FC", "Mirassol"),
        ("Chapecoense", "Chapecó"),
    ],
)
def test_brazilian_home_city_fast_path(team_name, expected_city):
    assert scout_agent._get_venue_city(team_name) == expected_city


@pytest.mark.parametrize(
    ("address", "expected_city"),
    [
        ("Sir Matt Busby Way, Manchester M16 0RA, England", "Manchester"),
        ("Werner-Heisenberg-Allee 25, 80939 München, Germany", "München"),
        ("Avenida Example 10, 28036 Madrid, Spain", "Madrid"),
        ("Rua Palestra Itália 200, 05005-000 São Paulo, Brazil", "São Paulo"),
        ("Rua Principal 10, 12345 Vespasiano, 33200-000, Brazil", "Vespasiano"),
        ("Anfield Road Liverpool L4 0TH", "Liverpool"),
    ],
)
def test_registered_address_city_parsing(address, expected_city):
    assert scout_agent._city_from_team_profile({"address": address}) == expected_city


def test_ambiguous_or_missing_address_does_not_invent_a_city():
    assert scout_agent._city_from_team_profile({}) is None
    assert scout_agent._city_from_team_profile({"address": "Rua Principal 10, 12345"}) is None


def test_competition_team_profiles_supply_home_cities():
    client = Mock()
    payload = {"teams": [
        {"id": 1769, "name": "SE Palmeiras", "address": "Rua A 1, 05005-000 São Paulo, Brazil"},
        {"id": 4287, "name": "Clube do Remo", "address": "Avenida B 2, 66050-000 Belém, Brazil"},
        {"id": 999, "name": "No Address FC", "address": ""},
    ]}
    with patch.object(scout_agent, "_fd_get", new=AsyncMock(return_value=payload)) as get:
        cities = asyncio.run(scout_agent.fetch_competition_home_cities(client, "BSA"))
    assert cities == {1769: "São Paulo", 4287: "Belém"}
    get.assert_awaited_once_with(
        client, "https://api.football-data.org/v4/competitions/BSA/teams"
    )


def test_unmapped_team_uses_cached_profile_city_for_weather():
    fixture = {
        "id": 1,
        "_competition_code": "BSA",
        "homeTeam": {"id": 123, "name": "Unmapped FC"},
        "awayTeam": {"id": 456, "name": "Away FC"},
        "competition": {"name": "Brazil Série A"},
        "utcDate": "2026-11-05T22:30:00Z",
    }
    city_cache = {}
    with (
        patch.object(scout_agent, "_get_venue_city", return_value=None),
        patch.object(
            scout_agent, "fetch_competition_home_cities",
            new=AsyncMock(return_value={123: "Vespasiano"}),
        ) as profile_fetch,
        patch.object(
            scout_agent, "fetch_weather", new=AsyncMock(return_value={"temp_c": None})
        ) as weather_fetch,
        patch.object(scout_agent, "ENABLE_REAL_FORM_DATA", False),
        patch.object(scout_agent, "analyze_with_gemini", return_value={"data_completeness": 0.7}),
    ):
        first = asyncio.run(scout_agent._deep_analyze(fixture, {}, {}, Mock(), {}, city_cache, {}))
        second = asyncio.run(scout_agent._deep_analyze(fixture, {}, {}, Mock(), {}, city_cache, {}))

    assert first["venue_city"] == "Vespasiano"
    assert second["venue_city_source"] == "football-data registered club address"
    assert profile_fetch.await_count == 1
    assert weather_fetch.await_count == 2
    weather_fetch.assert_awaited_with("Vespasiano", "2026-11-05T22:30:00Z")
