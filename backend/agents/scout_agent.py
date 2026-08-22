"""
SCOUT AGENT — Football Pulse AI (GitHub Actions edition)
Pulls fixtures, odds, form, injuries, lineups, weather, travel, market movement.
"""

import asyncio
import json
import logging
import os
import re
from datetime import date

import time

import httpx
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)
# max_retries=0: we do our own TPM-aware pacing + retry (see _groq_chat below)
# instead of relying on the SDK's blind exponential backoff.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"), max_retries=0)

GROQ_TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", 18000))
GROQ_TPM_WINDOW_SECONDS = 60
GROQ_MAX_LOCAL_RETRIES = 4
_token_usage_log: list[tuple[float, int]] = []

# Model fallback chain: on a daily-quota 429 (RPD/TPD) we switch to the next
# model rather than sleeping out a multi-minute daily cooldown, since each
# model on Groq carries its own independent daily budget. See GROQ_MODEL
# below for the primary model — this chain must start with it.
GROQ_MODEL_FALLBACK_CHAIN = [
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]
_current_model_index = 0


def _current_model() -> str:
    return GROQ_MODEL_FALLBACK_CHAIN[_current_model_index]


def _switch_to_next_model() -> bool:
    global _current_model_index
    if _current_model_index >= len(GROQ_MODEL_FALLBACK_CHAIN) - 1:
        return False
    _current_model_index += 1
    _token_usage_log.clear()
    logger.warning(
        f"[FALLBACK] Daily quota hit on previous model — switching to {_current_model()}"
    )
    return True


def _is_daily_limit_error(err: RateLimitError) -> bool:
    msg = str(err).lower()
    return "per day" in msg or "tpd" in msg or "rpd" in msg


def _tokens_used_in_window(now: float) -> int:
    cutoff = now - GROQ_TPM_WINDOW_SECONDS
    while _token_usage_log and _token_usage_log[0][0] < cutoff:
        _token_usage_log.pop(0)
    return sum(tokens for _, tokens in _token_usage_log)


def _pace_before_call(estimated_tokens: int) -> None:
    now = time.time()
    used = _tokens_used_in_window(now)
    if used + estimated_tokens <= GROQ_TPM_LIMIT:
        return
    wait = (_token_usage_log[0][0] + GROQ_TPM_WINDOW_SECONDS) - now
    if wait > 0:
        logger.info(
            f"[RATE LIMIT] {used}+{estimated_tokens} tokens would exceed the "
            f"{GROQ_TPM_LIMIT} TPM budget — pacing {wait:.1f}s before next call"
        )
        time.sleep(wait)


def _retry_after_seconds(err: RateLimitError) -> float | None:
    try:
        header = err.response.headers.get("retry-after")
        return float(header) if header is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _groq_chat(*, max_tokens: int, messages: list[dict]) -> str:
    """Same rate-limit-aware choke point as backend/agents/pipeline_agents.py —
    paces calls against real rolling TPM usage, falls forward to the next
    model in the chain on a daily-quota 429, and otherwise honors the
    server's Retry-After header if a per-minute 429 still slips through."""
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    estimated_tokens = (prompt_chars // 4) + max_tokens

    for attempt in range(GROQ_MAX_LOCAL_RETRIES):
        _pace_before_call(estimated_tokens)
        try:
            response = client.chat.completions.create(
                model=_current_model(),
                max_tokens=max_tokens,
                messages=messages,
            )
        except RateLimitError as e:
            if _is_daily_limit_error(e) and _switch_to_next_model():
                continue
            wait = _retry_after_seconds(e) or (2 ** attempt) * 5
            logger.warning(
                f"[RATE LIMIT] Groq 429 on {_current_model()} "
                f"(attempt {attempt + 1}/{GROQ_MAX_LOCAL_RETRIES}) — sleeping {wait:.1f}s"
            )
            time.sleep(wait)
            continue

        usage = getattr(response, "usage", None)
        actual_tokens = getattr(usage, "total_tokens", None) or estimated_tokens
        _token_usage_log.append((time.time(), actual_tokens))
        return response.choices[0].message.content

    raise RuntimeError(
        f"Groq API: still rate-limited on {_current_model()} after "
        f"{GROQ_MAX_LOCAL_RETRIES} local retries (fallback chain exhausted: "
        f"{_current_model_index == len(GROQ_MODEL_FALLBACK_CHAIN) - 1})"
    )

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

# ─────────────────────────────────────────────────────────────────────────────
# Home city lookup — used when football-data.org doesn't return a venue city.
# Keys are lowercased team name fragments so partial matches work too.
# ─────────────────────────────────────────────────────────────────────────────
TEAM_HOME_CITY: dict[str, str] = {
    # Premier League
    "arsenal": "London",
    "chelsea": "London",
    "tottenham": "London",
    "spurs": "London",
    "west ham": "London",
    "crystal palace": "London",
    "fulham": "London",
    "brentford": "London",
    "wimbledon": "London",
    "charlton": "London",
    "millwall": "London",
    "manchester city": "Manchester",
    "manchester united": "Manchester",
    "liverpool": "Liverpool",
    "everton": "Liverpool",
    "newcastle": "Newcastle upon Tyne",
    "sunderland": "Sunderland",
    "aston villa": "Birmingham",
    "birmingham": "Birmingham",
    "wolverhampton": "Wolverhampton",
    "wolves": "Wolverhampton",
    "west bromwich": "West Bromwich",
    "leicester": "Leicester",
    "nottingham": "Nottingham",
    "derby": "Derby",
    "sheffield united": "Sheffield",
    "sheffield wednesday": "Sheffield",
    "leeds": "Leeds",
    "burnley": "Burnley",
    "bolton": "Bolton",
    "blackburn": "Blackburn",
    "brighton": "Brighton",
    "southampton": "Southampton",
    "portsmouth": "Portsmouth",
    "watford": "Watford",
    "luton": "Luton",
    "norwich": "Norwich",
    "ipswich": "Ipswich",
    "coventry": "Coventry",
    "stoke": "Stoke-on-Trent",
    "middlesbrough": "Middlesbrough",
    "swansea": "Swansea",
    "cardiff": "Cardiff",
    # La Liga
    "real madrid": "Madrid",
    "atletico madrid": "Madrid",
    "atletico de madrid": "Madrid",
    "getafe": "Madrid",
    "rayo vallecano": "Madrid",
    "barcelona": "Barcelona",
    "espanyol": "Barcelona",
    "valencia": "Valencia",
    "villarreal": "Villarreal",
    "sevilla": "Seville",
    "real betis": "Seville",
    "athletic bilbao": "Bilbao",
    "athletic club": "Bilbao",
    "real sociedad": "San Sebastian",
    "osasuna": "Pamplona",
    "deportivo alaves": "Vitoria-Gasteiz",
    "alaves": "Vitoria-Gasteiz",
    "celta vigo": "Vigo",
    "malaga": "Malaga",
    "granada": "Granada",
    "real valladolid": "Valladolid",
    "cadiz": "Cadiz",
    "almeria": "Almeria",
    "girona": "Girona",
    "las palmas": "Las Palmas",
    "leganes": "Leganes",
    # Bundesliga
    "bayern munich": "Munich",
    "fc bayern": "Munich",
    "borussia dortmund": "Dortmund",
    "bvb": "Dortmund",
    "rb leipzig": "Leipzig",
    "bayer leverkusen": "Leverkusen",
    "borussia monchengladbach": "Monchengladbach",
    "eintracht frankfurt": "Frankfurt",
    "sc freiburg": "Freiburg",
    "vfb stuttgart": "Stuttgart",
    "stuttgar": "Stuttgart",
    "wolfsburg": "Wolfsburg",
    "werder bremen": "Bremen",
    "hamburger": "Hamburg",
    "hertha berlin": "Berlin",
    "union berlin": "Berlin",
    "schalke": "Gelsenkirchen",
    "augsburg": "Augsburg",
    "mainz": "Mainz",
    "hoffenheim": "Sinsheim",
    "koln": "Cologne",
    "fc koln": "Cologne",
    "cologne": "Cologne",
    "heidenheim": "Heidenheim",
    "darmstadt": "Darmstadt",
    # Serie A
    "juventus": "Turin",
    "torino": "Turin",
    "ac milan": "Milan",
    "inter milan": "Milan",
    "internazionale": "Milan",
    "como": "Como",
    "as roma": "Rome",
    "lazio": "Rome",
    "napoli": "Naples",
    "atalanta": "Bergamo",
    "fiorentina": "Florence",
    "bologna": "Bologna",
    "genoa": "Genoa",
    "sampdoria": "Genoa",
    "udinese": "Udine",
    "cagliari": "Cagliari",
    "sassuolo": "Sassuolo",
    "empoli": "Empoli",
    "lecce": "Lecce",
    "frosinone": "Frosinone",
    "monza": "Monza",
    "hellas verona": "Verona",
    "venezia": "Venice",
    "parma": "Parma",
    # Ligue 1
    "psg": "Paris",
    "paris saint-germain": "Paris",
    "paris saint germain": "Paris",
    "olympique de marseille": "Marseille",
    "marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "lyon": "Lyon",
    "monaco": "Monaco",
    "nice": "Nice",
    "stade rennais": "Rennes",
    "rennes": "Rennes",
    "lille": "Lille",
    "montpellier": "Montpellier",
    "nantes": "Nantes",
    "strasbourg": "Strasbourg",
    "toulouse": "Toulouse",
    "reims": "Reims",
    "lens": "Lens",
    "brest": "Brest",
    "auxerre": "Auxerre",
    "angers": "Angers",
    "le havre": "Le Havre",
    "clermont": "Clermont-Ferrand",
    "metz": "Metz",
    "lorient": "Lorient",
    "saint-etienne": "Saint-Etienne",
    # Eredivisie
    "ajax": "Amsterdam",
    "psv": "Eindhoven",
    "feyenoord": "Rotterdam",
    "az alkmaar": "Alkmaar",
    "az": "Alkmaar",
    "vitesse": "Arnhem",
    "utrecht": "Utrecht",
    "twente": "Enschede",
    # Primeira Liga
    "benfica": "Lisbon",
    "sporting cp": "Lisbon",
    "porto": "Porto",
    "braga": "Braga",
    "vitoria guimaraes": "Guimaraes",
    # World Cup / international — use host city where known
    "united states": "New York",
    "usa": "New York",
    "mexico": "Mexico City",
    "canada": "Toronto",
    "brazil": "Rio de Janeiro",
    "argentina": "Buenos Aires",
    "germany": "Berlin",
    "france": "Paris",
    "england": "London",
    "spain": "Madrid",
    "italy": "Rome",
    "portugal": "Lisbon",
    "netherlands": "Amsterdam",
    "belgium": "Brussels",
    "morocco": "Casablanca",
    "senegal": "Dakar",
    "nigeria": "Lagos",
    "egypt": "Cairo",
    "japan": "Tokyo",
    "south korea": "Seoul",
    "australia": "Sydney",
}


def _get_venue_city(home_team_name: str) -> str | None:
    """
    Look up a home city for the given team name using the TEAM_HOME_CITY table.
    Uses case-insensitive substring matching so partial names work too.
    """
    name_lower = home_team_name.lower().strip()
    # Try exact match first
    if name_lower in TEAM_HOME_CITY:
        return TEAM_HOME_CITY[name_lower]
    # Try substring match
    for key, city in TEAM_HOME_CITY.items():
        if key in name_lower or name_lower in key:
            return city
    return None


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
        logger.info("[SCOUT] No venue city — skipping weather fetch.")
        return {"temp_c": None, "wind_kmh": None, "rain_probability": 0, "conditions": "unknown"}

    async with httpx.AsyncClient(timeout=10) as http:
        try:
            resp = await http.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": venue_city,
                    "appid": os.environ.get("OPENWEATHER_KEY", ""),
                    "units": "metric",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            weather = {
                "temp_c": data["main"]["temp"],
                "wind_kmh": round(data["wind"]["speed"] * 3.6, 1),
                "rain_probability": data.get("rain", {}).get("1h", 0),
                "conditions": data["weather"][0]["description"],
            }
            logger.info(
                f"[SCOUT] Weather for {venue_city}: {weather['conditions']}, "
                f"{weather['temp_c']}°C, wind {weather['wind_kmh']} km/h"
            )
            return weather
        except Exception as e:
            logger.warning(f"[SCOUT] Weather fetch failed for {venue_city}: {e}")
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
    text = _groq_chat(
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
    result = _extract_json(text)

    # Guard: if the model returned a list instead of a dict, unwrap it
    if isinstance(result, list):
        logger.warning("[SCOUT] LLM returned a list instead of a dict — unwrapping first element.")
        result = result[0] if result and isinstance(result[0], dict) else {}

    if not isinstance(result, dict):
        logger.error(f"[SCOUT] Failed to parse LLM response as JSON object: {text[:200]}")
        return {}

    return result


MIN_QUALIFYING_MATCHES = 2
QUALIFYING_COMPLETENESS = CLEANER_THRESHOLD_ENV
HARD_CAP_MATCHES = 18


async def _deep_analyze(fixture: dict, odds_event: dict, epl_injuries: dict) -> dict:
    fixture_id = fixture["id"]
    home_name = fixture["homeTeam"]["name"]
    away_name = fixture["awayTeam"]["name"]
    competition_code = fixture.get("_competition_code")

    # Try to get venue city from the API response first,
    # then fall back to our home-team lookup table.
    venue_city = (
        fixture.get("venue")
        or fixture.get("homeTeam", {}).get("venue")
        or _get_venue_city(home_name)
    )
    if venue_city:
        logger.info(f"[SCOUT] Venue city for {home_name}: {venue_city}")
    else:
        logger.info(f"[SCOUT] No venue city found for {home_name} — skipping weather.")

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
        "venue_city": venue_city,
    }

    structured = analyze_with_groq(raw)

    # If LLM returned nothing usable, build a minimal skeleton so we don't crash
    if not structured:
        logger.warning(f"[SCOUT] analyze_with_groq returned empty result for {home_name} vs {away_name} — using skeleton.")
        structured = {
            "fixture_id": fixture_id,
            "home_team": home_name,
            "away_team": away_name,
            "league": fixture.get("competition", {}).get("name", "Unknown"),
            "data_completeness": 0.2,
        }

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
    Main scout agent entrypoint.
    Model: openai/gpt-oss-20b (20,000 TPM free tier).
    Weather: fetched via TEAM_HOME_CITY lookup when API doesn't return venue.
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
            # No fixed sleep here anymore — analyze_with_groq's _groq_chat call
            # already paces itself against the real rolling TPM budget.

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
