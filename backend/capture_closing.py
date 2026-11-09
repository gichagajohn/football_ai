"""
FOOTBALL PULSE AI — prospective closing-price capture.

Looks at today's *pending published selections* and records, for each
one, the price still available just before kickoff. Only fixtures that
have not started yet are captured; a price observed after kickoff would
be in-play noise, not a closing price. Missing coverage is left NULL on
purpose — it must never be backfilled with fabricated or derived values.

Runs from .github/workflows/closing_prices.yml a few times a day. The Odds
API free tier allows 500 requests per month, so this script only queries
competitions that actually hold a pending published leg for the target
date, and each league call also fetches only the featured markets
(h2h,totals). Nothing here changes tickets, probabilities or calibration.

Usage:
    python -m backend.capture_closing            # today (UTC)
    TARGET_DATE=2026-11-11 python -m backend.capture_closing
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timezone

from backend.agents.scout_agent import (
    _extract_odds_snapshot,
    _find_match_odds,
    fetch_fixtures,
    fetch_league_odds,
    resolve_odds_sport_keys,
)
from backend.db import supabase_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("football_pulse.closing")


def closing_price_from_snapshot(snapshot: dict | None, market: str) -> float | None:
    """Return the directly-quoted price for a market, or None.

    Never derives prices from other markets — an unquotable leg keeps an
    honest NULL closing price rather than a sum, difference or average of
    other markets.
    """
    if not isinstance(snapshot, dict):
        return None
    price = snapshot.get(market)
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if price <= 1.0 or price != price:
        return None
    return round(price, 2)


def _parse_kickoff(value) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


async def capture_closing_prices(target_date: date) -> int:
    if not supabase_client.storage_configured():
        logger.warning("[CLOSING] Supabase is not configured; nothing captured.")
        return 0

    selections = [
        s for s in supabase_client.get_selections_for_date(target_date)
        if str(s.get("outcome") or "").lower() == "pending"
    ]
    if not selections:
        logger.info(f"[CLOSING] No pending published selections for {target_date}; nothing to capture.")
        return 0

    markets_by_fixture: dict[int, set[str]] = {}
    for selection in selections:
        try:
            fixture_id = int(selection.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        market = selection.get("market")
        if market:
            markets_by_fixture.setdefault(fixture_id, set()).add(str(market))

    fixtures = await fetch_fixtures(target_date)
    needed = {
        int(m["id"]): m
        for m in fixtures
        if isinstance(m, dict) and str(m.get("id")) in {str(fid) for fid in markets_by_fixture}
    }
    if not needed:
        logger.warning(
            f"[CLOSING] {len(markets_by_fixture)} pending fixture(s) not found in today's "
            "configured competitions; closing prices stay NULL (explicitly missing)."
        )
        return 0

    now = datetime.now(timezone.utc)
    codes = {m.get("_competition_code") for m in needed.values() if m.get("_competition_code")}
    sport_keys = await resolve_odds_sport_keys(codes) if codes else {}
    events_by_code: dict[str, list[dict]] = {}
    for code, sport_key in sport_keys.items():
        events_by_code[code] = await fetch_league_odds(sport_key)

    updated = 0
    for fixture_id, fixture in needed.items():
        kickoff = _parse_kickoff(fixture.get("utcDate"))
        if kickoff is not None and kickoff <= now:
            logger.info(f"[CLOSING] fixture {fixture_id} already started; skipping (no in-play capture).")
            continue
        code = fixture.get("_competition_code") or ""
        events = events_by_code.get(code, [])
        if not events:
            logger.info(f"[CLOSING] fixture {fixture_id}: no odds events for competition {code}; closing stays NULL.")
            continue
        home = (fixture.get("homeTeam") or {}).get("name", "")
        away = (fixture.get("awayTeam") or {}).get("name", "")
        event = _find_match_odds(events, home, away)
        snapshot = _extract_odds_snapshot(event) if event else {}
        for market in sorted(markets_by_fixture.get(fixture_id, ())):
            price = closing_price_from_snapshot(snapshot, market)
            if price is None:
                logger.info(f"[CLOSING] fixture {fixture_id}: no direct quote for {market}; closing stays NULL.")
                continue
            updated += supabase_client.update_closing_price(target_date, fixture_id, market, price)

    logger.info(f"[CLOSING] Captured {updated} closing price(s) for {target_date}.")
    return updated


def main() -> None:
    raw = os.environ.get("TARGET_DATE", "").strip()
    target = date.fromisoformat(raw) if raw else datetime.now(timezone.utc).date()
    asyncio.run(capture_closing_prices(target))


if __name__ == "__main__":
    main()
