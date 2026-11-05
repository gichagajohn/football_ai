"""Provider-contract tests for bulk Odds API market collection."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.scout_agent import fetch_league_odds


def _response(status: int, payload=None, text: str = ""):
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload if payload is not None else []
    response.text = text
    response.raise_for_status.return_value = None
    if status >= 400:
        response.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
    return response


def _client(responses):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.side_effect = responses
    return client


def test_bulk_odds_requests_only_featured_markets():
    events = [{"id": "event-1"}]
    client = _client([_response(200, events)])
    with patch("backend.agents.scout_agent.httpx.AsyncClient", return_value=client):
        result = asyncio.run(fetch_league_odds("soccer_brazil_campeonato"))
    assert result == events
    assert client.get.call_count == 1
    params = client.get.call_args.kwargs["params"]
    assert params["markets"] == "h2h,totals"
    assert "btts" not in params["markets"]


def test_bulk_odds_falls_back_to_h2h_after_invalid_market(caplog):
    events = [{"id": "event-1"}]
    secret = "private-odds-key"
    client = _client([
        _response(422, text=f'{{"message":"INVALID_MARKET: totals {secret}"}}'),
        _response(200, events),
    ])
    with (
        patch("backend.agents.scout_agent.httpx.AsyncClient", return_value=client),
        patch.dict("os.environ", {"ODDS_API_KEY": secret}),
        caplog.at_level(logging.WARNING),
    ):
        result = asyncio.run(fetch_league_odds("soccer_brazil_campeonato"))
    assert result == events
    assert client.get.call_count == 2
    assert client.get.call_args_list[0].kwargs["params"]["markets"] == "h2h,totals"
    assert client.get.call_args_list[1].kwargs["params"]["markets"] == "h2h"
    assert "INVALID_MARKET" in caplog.text
    assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_bulk_odds_rejects_non_list_payload():
    client = _client([_response(200, {"error": "unexpected"})])
    with patch("backend.agents.scout_agent.httpx.AsyncClient", return_value=client):
        assert asyncio.run(fetch_league_odds("soccer_brazil_campeonato")) == []


def test_plan_or_quota_errors_do_not_trigger_market_fallback():
    for status in (401, 403, 429, 500):
        client = _client([_response(status, text='{"message":"provider error"}')])
        with patch("backend.agents.scout_agent.httpx.AsyncClient", return_value=client):
            assert asyncio.run(fetch_league_odds("soccer_brazil_campeonato")) == []
        assert client.get.call_count == 1
