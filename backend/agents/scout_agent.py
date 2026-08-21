"""
SCOUT AGENT — Football Pulse AI (GitHub Actions edition)
Pulls fixtures, odds, form, injuries, lineups, weather, travel, market movement.

Filters to top European leagues for better data quality/completeness.

Data sources (as of this version):
  - Fixtures:   football-data.org v4 (free tier, current season, no season-year guessing)
  - Odds:       The Odds API (free tier, 500 credits/month — one call per
                competition returns odds for every fixture in it)
  - Injuries:   Official Fantasy Premier League API — Premier League only.
                Other leagues get an empty injuries list; this is a known
                gap, not a bug (see fetch_epl_injuries docstring).
  - Weather:    OpenWeatherMap, unchanged from before.

Uses a two-pass strategy: fetch odds for every fixture in a competition in
ONE call, match each fixture to its odds by team name, then run full deep
analysis (odds+injuries+weather+LLM) on odds-matched fixtures first —
automatically pulling in more matches if too few pass the quality bar,
rather than being stuck with whatever the API happened to return first.
"""

import asyncio
import json
import logging
import os
import re
from datetime import date

import httpx
from groq import Groq

logger = logging.getLogger(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

DEFAULT_LEAGUE_IDS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "UEFA Champions League",
}

KNOWN_LEAGUE_NAMES = {
    **DEFAULT_LEAGUE_IDS,
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "ELC": "Championship (England)",
    "BSA": "Campeonato Brasileiro Série A",
    "WC": "FIFA World Cup",
    "EC": "European Championship",
}

ODDS_SPORT_KEYS = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "CL": "soccer_uefa_champs_league",
}

FPL_NAME_ALIASES = {
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Wolves": "Wolverhampton Wanderers",
}

# Model in use — llama-3.1-8b-instant has 20,000 TPM on the free tier,
# which is enough headroom for back-to-back fixture analyses without 429s.
GROQ_MODEL = "llama-3.1-8b-instant"

# Seconds to wait between consecutive Groq calls inside the fixture loop.
# 6s gives ~10 calls/min, well under the 30 RPM free-tier cap.
GROQ_CALL_DELAY_SECONDS = 6


def _load_league_ids() -> dict[str, str]:
    override = os.environ.get("LEAGUE_IDS", "").strip()
    if not override:
        return DEFAULT_LEAGUE_IDS
    codes = [part.strip().upper() for part in override.split(",") if part.strip()]
    if not codes:
        logger.warning("[SCOUT] LEAGUE_IDS was set but contained no valid codes — falling back to default top-5+UCL.")
        return DEFAULT_LEAGUE_IDS
    result = {c: KNOWN_LEAGUE_NAMES.get(c, f"Competition {c}") for c in codes}
    logger.info(f"[SCOUT] LEAGUE_IDS override active — using {len(result)} competition(s): {list(result.values())}")
    return result


TOP_LEAGUE_IDS = _load_league_ids()


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    raw = raw.strip() if raw else ""
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[SCOUT] Env var {name}='{raw}' is not a valid float — using default {default}.")
        return default


CLEANER_THRESHOLD_ENV = _float_env("CLEANER_THRESHOLD", 0.5)
MAX_MATCHES_PER_DAY = 12
FALLBACK_BATCH_SIZE = 6

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


FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"


async def fetch_fixtures(target_date: date) -> list[dict]:
    all_matches = []
    date_str = target_date.isoformat()
    async with httpx.AsyncClient(timeout=30) as http:
        for code, league_name in TOP_LEAGUE_IDS.items():
            try:
                resp = await http.get(
                    f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches",
                    headers={"X-Auth-Token": os.environ.get("FOOTBALL_DATA_KEY", "")},
                    params={"dateFrom": date_str, "dateTo": date_str},
                )
                resp.raise_for_status()
                matches = resp.json().get("matches", [])
                for m in matches:
                    m["_competition_code"] = code
                if matches:
                    logger.info(f"[SCOUT] {league_name}: {len(matches)} fixture(s) found.")
                all_matches.extend(matches)
            except Exception as e:
                logger.warning(f"[SCOUT] Fixture fetch failed for {league_name}: {e}")
            await asyncio.sleep(1)
    return all_matches


async def fetch_league_odds(sport_key: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=20) as http:
        try:
            resp = await http.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    "apiKey": os.environ.get("ODDS_API_KEY", ""),
                    "regions": "eu",
                    "markets": "h2h,totals",
                    "oddsFormat": "decimal",
                },
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"[SCOUT] Odds fetch failed for {sport_key}: {e}")
            return []


def _normalize_team_name(name: str) -> str:
    name = name.strip()
    for suffix in (" FC", " CF", " AFC", " CD", " SD", " AC"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = name.replace("&", "and")
    name = re.sub(r"\s+", " ", name)
    return name.strip().lower()


def _find_match_odds(odds_events: list[dict], home_name: str, away_name: str) -> dict:
    home_norm = _normalize_team_name(home_name)
    away_norm = _normalize_team_name(away_name)
    for event in odds_events:
        eh = _normalize_team_name(event.get("home_team", ""))
        ea = _normalize_team_name(event.get("away_team", ""))
        if (eh in home_norm or home_norm in eh) and (ea in away_norm or away_norm in ea):
            return event
    return {}


def _extract_odds_snapshot(odds_event: dict) -> dict:
    snapshot = {}
    bookmakers = odds_event.get("bookmakers", [])
    if not bookmakers:
        return snapshot
    book = bookmakers[0]
    home_team = odds_event.get("home_team")
    away_team = odds_event.get("away_team")
    for market in book.get("markets", []):
        if market.get("key") == "h2h":
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                if name == home_team:
                    snapshot["home_win"] = outcome.get("price")
                elif name == away_team:
                    snapshot["away_win"] = outcome.get("price")
                elif name.lower() == "draw":
                    snapshot["draw"] = outcome.get("price")
        elif market.get("key") == "totals":
            for outcome in market.get("outcomes", []):
                if outcome.get("point") == 2.5 and outcome.get("name", "").lower() == "over":
                    snapshot["over25"] = outcome.get("price")
    return snapshot


async def fetch_epl_injuries() -> dict[str, list[dict]]:
    async with httpx.AsyncClient(timeout=15) as http:
        try:
            resp = await http.get("https://fantasy.premierleague.com/api/bootstrap-static/")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"[SCOUT] FPL injury fetch failed: {e}")
            return {}

    teams_by_id = {t["id"]: t["name"] for t in data.get("teams", [])}
    injuries_by_team: dict[str, list[dict]] = {}
    for player in data.get("elements", []):
        if player.get("status") == "a":
            continue
        team_name = teams_by_id.get(player.get("team"))
        if not team_name:
            continue
        injuries_by_team.setdefault(team_name, []).append(
            {
                "player": player.get("web_name"),
                "status": player.get("status"),
                "news": player.get("news") or "No details provided",
                "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
            }
        )
    return injuries_by_team


def _lookup_epl_injuries(injuries_by_team: dict, fd_team_name: str) -> list[dict]:
    target = _normalize_team_name(fd_team_name)
    for fpl_name, injuries in injuries_by_team.items():
        candidate = _normalize_team_name(FPL_NAME_ALIASES.get(fpl_name, fpl_name))
        if candidate == target or candidate in target or target in candidate:
            return injuries
    return []


async def fetch_weather(venue_city: str | None) -> dict:
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
        model=GROQ_MODEL,
        max_tokens=2048,
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


MIN_QUALIFYING_MATCHES = 2
QUALIFYING_COMPLETENESS = CLEANER_THRESHOLD_ENV
HARD_CAP_MATCHES = 18


async def _deep_analyze(fixture: dict, odds_event: dict, epl_injuries: dict) -> dict:
    fixture_id = fixture["id"]
    home_name = fixture["homeTeam"]["name"]
    away_name = fixture["awayTeam"]["name"]
    competition_code = fixture.get("_competition_code")
    venue_city = None

    odds_snapshot = _extract_odds_snapshot(odds_event) if odds_event else {}

    home_injuries = []
    away_injuries = []
    if competition_code == "PL":
        home_injuries = _lookup_epl_injuries(epl_injuries, home_name)
        away_injuries = _lookup_epl_injuries(epl_injuries, away_name)

    weather = await fetch_weather(venue_city)

    raw = {
        "fixture": fixture,
        "odds": odds_snapshot,
        "home_injuries": home_injuries,
        "away_injuries": away_injuries,
        "weather": weather,
    }

    structured = analyze_with_groq(raw)

    if "data_completeness" not in structured or structured.get("data_completeness") is None:
        score = 0.0
        if structured.get("home_team") and structured.get("away_team"):
            score += 0.4
        if home_injuries or away_injuries:
            score += 0.3
        if odds_snapshot:
            score += 0.2
        if weather.get("temp_c") is not None:
            score += 0.1
        structured["data_completeness"] = round(score, 2)

    structured.setdefault("fixture_id", fixture_id)
    if not structured.get("odds_snapshot") and odds_snapshot:
        structured["odds_snapshot"] = odds_snapshot

    logger.info(
        f"[SCOUT] ✓ {structured.get('home_team', '?')} vs {structured.get('away_team', '?')} "
        f"({structured.get('league', '?')}) — completeness={structured.get('data_completeness')}"
    )
    return structured


async def run(target_date: date | None = None) -> list[dict]:
    """
    Main scout agent entrypoint. Returns list of structured match intelligence.
    Model: llama-3.1-8b-instant (20,000 TPM free tier — replaces openai/gpt-oss-120b which had only 8,000 TPM).
    Call delay: 6s between fixture analyses to stay safely under rate limits.
    """
    target_date = target_date or date.today()
    league_names = list(TOP_LEAGUE_IDS.values())
    logger.info(f"[SCOUT] Collecting intelligence for {target_date} (leagues: {league_names})")

    fixtures = await fetch_fixtures(target_date)
    if not fixtures:
        logger.warning("[SCOUT] No fixtures found in configured leagues for this date.")
        return []

    codes_present = {f["_competition_code"] for f in fixtures}

    logger.info(f"[SCOUT] Fetching odds for {len(codes_present)} competition(s) with fixtures today...")
    odds_by_league: dict[str, list[dict]] = {}
    for code in codes_present:
        sport_key = ODDS_SPORT_KEYS.get(code)
        if not sport_key:
            logger.info(f"[SCOUT] No odds source configured for competition {code} — skipping.")
            continue
        odds_by_league[code] = await fetch_league_odds(sport_key)
        await asyncio.sleep(0.5)

    epl_injuries: dict[str, list[dict]] = {}
    if "PL" in codes_present:
        epl_injuries = await fetch_epl_injuries()

    scanned = []
    for fixture in fixtures:
        events = odds_by_league.get(fixture["_competition_code"], [])
        matched = _find_match_odds(events, fixture["homeTeam"]["name"], fixture["awayTeam"]["name"])
        scanned.append((fixture, matched))

    with_odds = [(f, o) for f, o in scanned if o]
    without_odds = [(f, o) for f, o in scanned if not o]
    prioritized = with_odds + without_odds

    logger.info(f"[SCOUT] {len(with_odds)}/{len(scanned)} fixtures matched to odds. Prioritizing those first.")

    results: list[dict] = []
    cursor = 0
    is_first_batch = True

    while cursor < len(prioritized) and len(results) < HARD_CAP_MATCHES:
        batch_size = MAX_MATCHES_PER_DAY if is_first_batch else FALLBACK_BATCH_SIZE
        remaining_budget = HARD_CAP_MATCHES - len(results)
        batch_size = min(batch_size, remaining_budget)

        batch = prioritized[cursor:cursor + batch_size]
        cursor += batch_size
        is_first_batch = False

        logger.info(f"[SCOUT] Analyzing batch of {len(batch)} fixture(s) (processed so far: {len(results)})...")

        for fixture, odds_event in batch:
            structured = await _deep_analyze(fixture, odds_event, epl_injuries)
            results.append(structured)
            await asyncio.sleep(GROQ_CALL_DELAY_SECONDS)

        qualifying = sum(1 for r in results if r.get("data_completeness", 0) >= QUALIFYING_COMPLETENESS)
        logger.info(f"[SCOUT] {qualifying}/{len(results)} analyzed matches meet completeness >= {QUALIFYING_COMPLETENESS}.")

        if qualifying >= MIN_QUALIFYING_MATCHES:
            logger.info("[SCOUT] Enough qualifying matches found — stopping further batches.")
            break

        if cursor < len(prioritized) and len(results) < HARD_CAP_MATCHES:
            logger.info(
                f"[SCOUT] Only {qualifying} qualifying match(es) so far (need {MIN_QUALIFYING_MATCHES}). "
                f"Pulling one fallback batch automatically..."
            )
        else:
            logger.info(
                f"[SCOUT] Only {qualifying} qualifying match(es), but no more fixtures or budget remaining."
            )

    logger.info(f"[SCOUT] Collected {len(results)} match intelligence packages total.")
    return results
