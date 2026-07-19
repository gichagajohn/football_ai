"""
SCOUT AGENT — Football Pulse AI (GitHub Actions edition)
Pulls fixtures, odds, form, injuries, lineups, weather, travel, market movement.

Filters to top European leagues for better data quality/completeness,
and increases the per-day match cap since GitHub Actions runs once daily
(no repeated manual testing burning through quota).
"""

import asyncio
import json
import logging
import os
import re
from datetime import date, timedelta

import httpx
from groq import Groq

logger = logging.getLogger(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# API-Football league IDs for the top 5 European leagues + Champions League.
# https://www.api-football.com/documentation-v3#tag/Leagues
TOP_LEAGUE_IDS = {
    39: "Premier League",
    140: "La Liga",
    78: "Bundesliga",
    135: "Serie A",
    61: "Ligue 1",
    2: "UEFA Champions League",
}

# Max matches processed per day. Each match = 1 odds call + 2 injury calls
# (3 API-Football calls) + 1 Groq call for Scout analysis. At 15 matches:
# ~46 API-Football calls (well under 100/day free tier) and ~15 Groq calls
# for Scout alone (leaving budget for Analyst/Risk/Portfolio/Auditor/Decision/
# Publisher on the matches that make it through the Cleaner).
MAX_MATCHES_PER_DAY = 15

SYSTEM_PROMPT = """You are the SCOUT AGENT for Football Pulse AI.
Your job is to gather and structure football match intelligence.

IMPORTANT TIMING CONTEXT: You are analyzing matches roughly 24-31 hours
before kickoff (this runs once daily, the morning before matchday).
Official lineups are almost NEVER confirmed this far ahead — they
typically post about 1 hour before kickoff. An unconfirmed lineup at
this stage is NORMAL, not a data quality problem.

Given raw data from APIs and web sources, you:
1. Extract the most relevant fixtures for today + next 24h
2. Identify key injury/suspension concerns
3. Flag meaningful odds movements (>10% from open)
4. Note travel distances >500km for away teams
5. Assess weather risk (heavy rain, wind >50km/h, extreme cold)
6. Note lineup status as informational only (expected to be unconfirmed
   at this stage — this is not itself a red flag)

IMPORTANT OUTPUT RULES:
- Respond with ONLY the JSON object. No markdown code fences, no commentary, no explanation before or after.
- data_completeness must be a float between 0.0 and 1.0 reflecting how much real data you actually received
  (team names, odds present, injury reports present, weather present). Do NOT penalize completeness for
  lineups being unconfirmed — that's expected at this stage, not missing data. If odds/injuries/weather
  are genuinely missing/UNKNOWN, data_completeness should be LOW (e.g. 0.3-0.5); lineup status has no
  bearing on this score.
- Never hallucinate injury or lineup data — if unknown, state 'UNKNOWN'."""


async def fetch_fixtures(target_date: date) -> list[dict]:
    """Fetch fixtures for the target date, filtered to top leagues."""
    all_fixtures = []
    async with httpx.AsyncClient(timeout=30) as http:
        for league_id, league_name in TOP_LEAGUE_IDS.items():
            try:
                resp = await http.get(
                    "https://v3.football.api-sports.io/fixtures",
                    headers={"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")},
                    params={
                        "date": target_date.isoformat(),
                        "timezone": "Africa/Nairobi",
                        "league": league_id,
                        "season": _season_for_league(target_date),
                    },
                )
                resp.raise_for_status()
                fixtures = resp.json().get("response", [])
                if fixtures:
                    logger.info(f"[SCOUT] {league_name}: {len(fixtures)} fixture(s) found.")
                all_fixtures.extend(fixtures)
            except Exception as e:
                logger.warning(f"[SCOUT] Fixture fetch failed for {league_name}: {e}")

    return all_fixtures


def _season_for_league(target_date: date) -> int:
    """
    European league seasons span Aug-May, labeled by the starting year.
    e.g. the 2026-27 season is queried as season=2026.
    For Jan-Jul dates, the season is the PREVIOUS year.
    """
    if target_date.month >= 8:
        return target_date.year
    return target_date.year - 1


async def fetch_odds(fixture_id: int) -> dict:
    """Fetch bookmaker odds for a fixture. Returns {} if no odds available."""
    async with httpx.AsyncClient(timeout=15) as http:
        try:
            resp = await http.get(
                "https://v3.football.api-sports.io/odds",
                headers={"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")},
                params={"fixture": fixture_id, "bookmaker": 6},  # Bet365
            )
            resp.raise_for_status()
            odds_data = resp.json().get("response", [])
            if odds_data:
                return odds_data[0]
            logger.info(f"[SCOUT] No odds available yet for fixture {fixture_id}")
            return {}
        except Exception as e:
            logger.warning(f"Odds fetch failed for {fixture_id}: {e}")
            return {}


async def fetch_injuries(team_id: int | None) -> list[dict]:
    """Fetch current injury/suspension list for a team."""
    if not team_id:
        return []
    async with httpx.AsyncClient(timeout=15) as http:
        try:
            resp = await http.get(
                "https://v3.football.api-sports.io/injuries",
                headers={"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")},
                params={"team": team_id},
            )
            resp.raise_for_status()
            return resp.json().get("response", [])
        except Exception as e:
            logger.warning(f"Injury fetch failed for team {team_id}: {e}")
            return []


async def fetch_weather(venue_city: str | None) -> dict:
    """Fetch weather forecast for match venue. Returns blank dict if no city known."""
    if not venue_city:
        logger.info("[SCOUT] No venue city available — skipping weather fetch.")
        return {"temp_c": None, "wind_kmh": None, "rain_probability": 0, "conditions": "unknown"}

    async with httpx.AsyncClient(timeout=10) as http:
        try:
            resp = await http.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": venue_city, "appid": os.environ.get("OPENWEATHER_KEY", ""), "units": "metric"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "temp_c": data["main"]["temp"],
                "wind_kmh": data["wind"]["speed"] * 3.6,
                "rain_probability": data.get("rain", {}).get("1h", 0),
                "conditions": data["weather"][0]["description"],
            }
        except Exception as e:
            logger.warning(f"Weather fetch failed for {venue_city}: {e}")
            return {"temp_c": None, "wind_kmh": None, "rain_probability": 0, "conditions": "unknown"}


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from an LLM response."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def analyze_with_groq(raw_data: dict) -> dict:
    """Use the LLM to structure and enrich the raw scout data."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""Analyze this raw fixture data and return structured match intelligence JSON.

RAW DATA:
{raw_data}

Return a JSON object with this structure (and nothing else):
{{
  "fixture_id": int,
  "home_team": str,
  "away_team": str,
  "league": str,
  "kickoff_utc": str,
  "form": {{"home": [...], "away": [...]}},
  "injuries": {{"home": [...], "away": [...]}},
  "odds_snapshot": {{"home_win": float, "draw": float, "away_win": float, "btts_yes": float, "over25": float}},
  "odds_movement": str,
  "lineup_status": {{"home_confirmed": bool, "away_confirmed": bool}},
  "weather_risk": {{"level": "low|medium|high", "reason": str}},
  "travel_km": int,
  "scout_flags": [str],
  "data_completeness": float
}}""",
            }
        ],
    )
    text = response.choices[0].message.content
    result = _extract_json(text)
    if not result:
        logger.error(f"[SCOUT] Failed to parse LLM response as JSON: {text[:200]}")
    return result


async def run(target_date: date | None = None) -> list[dict]:
    """Main scout agent entrypoint. Returns list of structured match intelligence."""
    target_date = target_date or date.today() + timedelta(days=1)
    logger.info(f"[SCOUT] Collecting intelligence for {target_date} (top-5 leagues + UCL)")

    fixtures = await fetch_fixtures(target_date)
    if not fixtures:
        logger.warning("[SCOUT] No fixtures found in top leagues for this date.")
        return []

    logger.info(f"[SCOUT] {len(fixtures)} total fixtures found across top leagues, processing up to {MAX_MATCHES_PER_DAY}.")

    results = []
    for fixture in fixtures[:MAX_MATCHES_PER_DAY]:
        fixture_id = fixture["fixture"]["id"]
        home_id = fixture["teams"]["home"]["id"]
        away_id = fixture["teams"]["away"]["id"]
        venue_city = fixture["fixture"].get("venue", {}).get("city") or None

        odds, home_injuries, away_injuries, weather = await asyncio.gather(
            fetch_odds(fixture_id),
            fetch_injuries(home_id),
            fetch_injuries(away_id),
            fetch_weather(venue_city),
        )

        raw = {
            "fixture": fixture,
            "odds": odds,
            "home_injuries": home_injuries,
            "away_injuries": away_injuries,
            "weather": weather,
        }

        structured = analyze_with_groq(raw)

        # Fallback completeness score, odds weighted lower (often post late)
        if "data_completeness" not in structured or structured.get("data_completeness") is None:
            score = 0.0
            if structured.get("home_team") and structured.get("away_team"):
                score += 0.4
            if home_injuries or away_injuries:
                score += 0.3
            if odds:
                score += 0.2
            if weather.get("temp_c") is not None:
                score += 0.1
            structured["data_completeness"] = round(score, 2)

        structured.setdefault("fixture_id", fixture_id)
        # Carry forward raw odds_snapshot fallback in case LLM omitted it
        if not structured.get("odds_snapshot") and odds:
            structured["odds_snapshot"] = _extract_odds_snapshot(odds)

        results.append(structured)
        logger.info(
            f"[SCOUT] ✓ {structured.get('home_team', '?')} vs {structured.get('away_team', '?')} "
            f"({structured.get('league', '?')}) — completeness={structured.get('data_completeness')}"
        )

        await asyncio.sleep(1)  # ease rate limits

    logger.info(f"[SCOUT] Collected {len(results)} match intelligence packages.")
    return results


def _extract_odds_snapshot(odds_response: dict) -> dict:
    """
    Best-effort extraction of home/draw/away/btts/over25 odds from
    API-Football's odds response structure, as a fallback if the
    LLM didn't populate odds_snapshot itself.
    """
    snapshot = {}
    try:
        bookmakers = odds_response.get("bookmakers", [])
        if not bookmakers:
            return snapshot
        bets = bookmakers[0].get("bets", [])
        for bet in bets:
            name = bet.get("name", "")
            values = bet.get("values", [])
            if name == "Match Winner":
                for v in values:
                    if v["value"] == "Home":
                        snapshot["home_win"] = float(v["odd"])
                    elif v["value"] == "Draw":
                        snapshot["draw"] = float(v["odd"])
                    elif v["value"] == "Away":
                        snapshot["away_win"] = float(v["odd"])
            elif name == "Both Teams Score":
                for v in values:
                    if v["value"] == "Yes":
                        snapshot["btts_yes"] = float(v["odd"])
            elif name == "Goals Over/Under":
                for v in values:
                    if v["value"] == "Over 2.5":
                        snapshot["over25"] = float(v["odd"])
    except (KeyError, ValueError, IndexError, TypeError):
        pass
    return snapshot
