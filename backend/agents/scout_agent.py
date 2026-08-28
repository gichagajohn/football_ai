"""
SCOUT AGENT — Football Pulse AI (GitHub Actions edition)
Pulls fixtures, odds, form, injuries, lineups, weather, travel, market movement.
"""

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from datetime import date

import httpx

from backend.gemini_client import gemini_chat as _gemini_chat, get_last_finish_reason

logger = logging.getLogger(__name__)

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

TEAM_HOME_CITY: dict[str, str] = {
    # England
    "arsenal": "London", "chelsea": "London", "tottenham": "London", "spurs": "London",
    "west ham": "London", "crystal palace": "London", "fulham": "London",
    "brentford": "London", "wimbledon": "London", "charlton": "London",
    "millwall": "London", "qpr": "London", "leyton orient": "London",
    "manchester city": "Manchester", "manchester united": "Manchester",
    "liverpool": "Liverpool", "everton": "Liverpool",
    "newcastle": "Newcastle upon Tyne", "newcastle united": "Newcastle upon Tyne",
    "sunderland": "Sunderland", "middlesbrough": "Middlesbrough",
    "aston villa": "Birmingham", "birmingham": "Birmingham", "west bromwich": "West Bromwich",
    "wolves": "Wolverhampton", "wolverhampton": "Wolverhampton",
    "leicester": "Leicester", "nottingham": "Nottingham", "derby": "Derby",
    "sheffield united": "Sheffield", "sheffield wednesday": "Sheffield",
    "leeds": "Leeds", "burnley": "Burnley", "bolton": "Bolton",
    "blackburn": "Blackburn", "brighton": "Brighton", "southampton": "Southampton",
    "portsmouth": "Portsmouth", "watford": "Watford", "luton": "Luton",
    "norwich": "Norwich", "ipswich": "Ipswich", "coventry": "Coventry",
    "stoke": "Stoke-on-Trent", "swansea": "Swansea", "cardiff": "Cardiff",
    "bournemouth": "Bournemouth", "bristol city": "Bristol", "plymouth": "Plymouth",
    # Spain
    "real madrid": "Madrid", "atletico madrid": "Madrid", "atletico de madrid": "Madrid",
    "getafe": "Madrid", "rayo vallecano": "Madrid", "barcelona": "Barcelona",
    "espanyol": "Barcelona", "valencia": "Valencia", "villarreal": "Villarreal",
    "sevilla": "Seville", "real betis": "Seville", "athletic bilbao": "Bilbao",
    "athletic club": "Bilbao", "real sociedad": "San Sebastian", "osasuna": "Pamplona",
    "deportivo alaves": "Vitoria-Gasteiz", "alaves": "Vitoria-Gasteiz",
    "celta vigo": "Vigo", "malaga": "Malaga", "granada": "Granada",
    "real valladolid": "Valladolid", "cadiz": "Cadiz", "almeria": "Almeria",
    "girona": "Girona", "las palmas": "Las Palmas", "leganes": "Leganes",
    "elche": "Elche", "racing": "Santander", "racing santander": "Santander",
    "real sociedad de futbol": "San Sebastian", "deportivo": "La Coruna",
    # Germany (Bundesliga 2025/26 - promoted + relegated)
    "bayern munich": "Munich", "fc bayern": "Munich", "fc bayern munchen": "Munich",
    "borussia dortmund": "Dortmund", "bvb": "Dortmund", "rb leipzig": "Leipzig",
    "bayer leverkusen": "Leverkusen",
    "borussia monchengladbach": "Monchengladbach", "gladbach": "Monchengladbach",
    "eintracht frankfurt": "Frankfurt", "sc freiburg": "Freiburg",
    "vfb stuttgart": "Stuttgart", "stuttgart": "Stuttgart",
    "vfl wolfsburg": "Wolfsburg", "wolfsburg": "Wolfsburg",
    "werder bremen": "Bremen", "hamburger sv": "Hamburg", "hamburg": "Hamburg",
    "hertha berlin": "Berlin", "union berlin": "Berlin",
    "fc schalke 04": "Gelsenkirchen", "schalke": "Gelsenkirchen",
    "fc augsburg": "Augsburg", "augsburg": "Augsburg",
    "mainz 05": "Mainz", "fsv mainz": "Mainz", "mainz": "Mainz",
    "tsg 1899 hoffenheim": "Sinsheim", "hoffenheim": "Sinsheim",
    "1 fc koln": "Cologne", "fc koln": "Cologne", "koln": "Cologne",
    "1 fc heidenheim": "Heidenheim", "heidenheim": "Heidenheim",
    "sv darmstadt 98": "Darmstadt", "darmstadt": "Darmstadt",
    "1 fc kaiserslautern": "Kaiserslautern", "kaiserslautern": "Kaiserslautern",
    "1 fc magdeburg": "Magdeburg", "magdeburg": "Magdeburg",
    "1 fc nurnberg": "Nuremberg", "nurnberg": "Nuremberg",
    "fc schalke": "Gelsenkirchen",
    # Italy (Serie A 2025/26)
    "juventus": "Turin", "torino": "Turin", "ac milan": "Milan",
    "inter milan": "Milan", "internazionale": "Milan", "fc internazionale": "Milan",
    "como 1907": "Como", "como": "Como",
    "as roma": "Rome", "ss lazio": "Rome", "lazio": "Rome",
    "napoli": "Naples", "ssc napoli": "Naples",
    "atalanta": "Bergamo", "atalanta bc": "Bergamo",
    "acf fiorentina": "Florence", "fiorentina": "Florence",
    "bologna": "Bologna", "bologna fc": "Bologna",
    "genoa": "Genoa", "sampdoria": "Genoa",
    "udinese": "Udine", "cagliari": "Cagliari",
    "sassuolo": "Sassuolo", "us sassuolo": "Sassuolo",
    "empoli": "Empoli", "lecce": "Lecce", "frosinone": "Frosinone",
    "monza": "Monza", "hellas verona": "Verona", "verona": "Verona",
    "venezia": "Venice", "parma": "Parma", "palermo": "Palermo",
    "brescia": "Brescia", "cremonese": "Cremona", "spezia": "La Spezia",
    "ternana": "Terni", "reggina": "Reggio Calabria",
    # France
    "psg": "Paris", "paris saint-germain": "Paris", "paris saint germain": "Paris",
    "olympique de marseille": "Marseille", "om": "Marseille", "marseille": "Marseille",
    "olympique lyonnais": "Lyon", "ol": "Lyon", "lyon": "Lyon",
    "as monaco": "Monaco", "monaco": "Monaco",
    "ogc nice": "Nice", "nice": "Nice",
    "stade rennais": "Rennes", "rennes": "Rennes",
    "losc lille": "Lille", "lille": "Lille",
    "montpellier": "Montpellier", "fc nantes": "Nantes", "nantes": "Nantes",
    "rc strasbourg": "Strasbourg", "strasbourg": "Strasbourg",
    "toulouse fc": "Toulouse", "toulouse": "Toulouse",
    "stade de reims": "Reims", "reims": "Reims",
    "rc lens": "Lens", "lens": "Lens", "stade brestois": "Brest", "brest": "Brest",
    "aj auxerre": "Auxerre", "auxerre": "Auxerre",
    "angers sco": "Angers", "angers": "Angers",
    "le havre ac": "Le Havre", "le havre": "Le Havre",
    "clermont foot": "Clermont-Ferrand", "clermont": "Clermont-Ferrand",
    "fc metz": "Metz", "metz": "Metz",
    "fc lorient": "Lorient", "lorient": "Lorient",
    "as saint-etienne": "Saint-Etienne", "saint-etienne": "Saint-Etienne",
    "fc nantes": "Nantes",
    # Netherlands / Portugal
    "ajax": "Amsterdam", "psv eindhoven": "Eindhoven", "psv": "Eindhoven",
    "feyenoord": "Rotterdam", "az alkmaar": "Alkmaar", "az": "Alkmaar",
    "vitesse": "Arnhem", "fc utrecht": "Utrecht", "utrecht": "Utrecht",
    "fc twente": "Enschede", "twente": "Enschede",
    "sl benfica": "Lisbon", "benfica": "Lisbon", "sporting cp": "Lisbon",
    "sporting clube de portugal": "Lisbon",
    "fc porto": "Porto", "porto": "Porto",
    "sc braga": "Braga", "braga": "Braga",
    "vitoria sc": "Guimaraes", "vitoria guimaraes": "Guimaraes",
    "rio ave fc": "Vila do Conde", "rio ave": "Vila do Conde",
    "gil vicente fc": "Barcelos", "gil vicente": "Barcelos",
    # Countries (fallback)
    "united states": "New York", "usa": "New York", "mexico": "Mexico City",
    "canada": "Toronto", "brazil": "Rio de Janeiro", "argentina": "Buenos Aires",
    "germany": "Berlin", "france": "Paris", "england": "London",
    "spain": "Madrid", "italy": "Rome", "portugal": "Lisbon",
    "netherlands": "Amsterdam", "belgium": "Brussels", "morocco": "Casablanca",
    "senegal": "Dakar", "nigeria": "Lagos", "egypt": "Cairo",
    "japan": "Tokyo", "south korea": "Seoul", "australia": "Sydney",
}



def _strip_accents(text: str) -> str:
    """Lowercase + strip diacritics so 'Alavés' matches 'alaves'."""
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )


def _normalize_team_name_for_lookup(name: str) -> str:
    """Strip common club-name noise so the TEAM_HOME_CITY keys match."""
    s = _strip_accents(name)
    # common suffixes and qualifiers
    for suffix in (
        " fc", " cf", " afc", " ssc", " ssc ", " sc", " bc", " sv",
        " ac", " acf", " as", " ogc", " losc", " rc ", " stade",
        " olympique", " club", " deportivo", " sporting",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


def _get_venue_city(home_team_name: str) -> str | None:
    """Look up a team's home city. Tries:
    1) exact lower match,
    2) diacritic-stripped match,
    3) normalised suffix-stripped match,
    4) substring containment (either direction).
    Returns the city string, or None if no match."""
    if not home_team_name:
        return None
    name_lower = home_team_name.lower().strip()
    if name_lower in TEAM_HOME_CITY:
        return TEAM_HOME_CITY[name_lower]
    name_norm = _normalize_team_name_for_lookup(home_team_name)
    if name_norm in TEAM_HOME_CITY:
        return TEAM_HOME_CITY[name_norm]
    for key, city in TEAM_HOME_CITY.items():
        if key in name_lower or name_lower in key:
            return city
        key_norm = _normalize_team_name_for_lookup(key)
        if key_norm in name_norm or name_norm in key_norm:
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
7. Report recent form / head-to-head / standings using ONLY the real
   figures provided to you under "recent_form", "head_to_head", and
   "standings" in the raw data. These come from a real stats API, not
   from you. Copy/summarize them faithfully. If a field arrives as an
   empty object {{}}, that means the data genuinely was not available —
   report it as 'UNKNOWN', do NOT invent a plausible-looking form string,
   H2H record, or league position to fill the gap. A form field you did
   not compute from the supplied data is a hallucination, full stop.

IMPORTANT OUTPUT RULES:
- Respond with ONLY the JSON object. No markdown code fences, no commentary, no explanation before or after.
- data_completeness must be a float between 0.0 and 1.0 reflecting how much real data you actually received
  (team names, odds present, injury reports present, weather present, recent_form/head_to_head/standings
  present and non-empty). Do NOT penalize completeness for lineups being unconfirmed — that's expected at
  this stage, not missing data. If odds/injuries/weather/form/H2H/standings are genuinely missing/UNKNOWN,
  data_completeness should be LOW (e.g. 0.3-0.5); lineup status has no bearing on this score.
- Never hallucinate injury, lineup, form, head-to-head, or standings data — if unknown, state 'UNKNOWN'."""


FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

ENABLE_REAL_FORM_DATA = os.environ.get("ENABLE_REAL_FORM_DATA", "true").strip().lower() != "false"

FOOTBALL_DATA_MIN_INTERVAL_SECONDS = _float_env("FOOTBALL_DATA_MIN_INTERVAL_SECONDS", 6.5)
_fd_last_request_time = 0.0

FOOTBALL_DATA_MAX_RETRIES = 2
FOOTBALL_DATA_DEFAULT_RETRY_WAIT_SECONDS = 30.0


async def _fd_pace() -> None:
    global _fd_last_request_time
    now = time.time()
    wait = FOOTBALL_DATA_MIN_INTERVAL_SECONDS - (now - _fd_last_request_time)
    if wait > 0:
        await asyncio.sleep(wait)
    _fd_last_request_time = time.time()


async def _fd_get(http: httpx.AsyncClient, url: str, params: dict | None = None) -> dict:
    last_exc: Exception | None = None
    for attempt in range(FOOTBALL_DATA_MAX_RETRIES + 1):
        await _fd_pace()
        resp = await http.get(
            url,
            headers={"X-Auth-Token": os.environ.get("FOOTBALL_DATA_KEY", "")},
            params=params,
        )
        if resp.status_code == 429:
            if attempt >= FOOTBALL_DATA_MAX_RETRIES:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else FOOTBALL_DATA_DEFAULT_RETRY_WAIT_SECONDS
            except ValueError:
                wait = FOOTBALL_DATA_DEFAULT_RETRY_WAIT_SECONDS
            logger.warning(
                f"[SCOUT] football-data.org 429 on {url} "
                f"(attempt {attempt + 1}/{FOOTBALL_DATA_MAX_RETRIES}) — retrying in {wait:.0f}s"
            )
            await asyncio.sleep(wait)
            continue
        try:
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            break
    if last_exc:
        raise last_exc
    return {}


async def fetch_standings(http: httpx.AsyncClient, competition_code: str) -> dict[int, dict]:
    try:
        data = await _fd_get(http, f"{FOOTBALL_DATA_BASE}/competitions/{competition_code}/standings")
    except Exception as e:
        logger.warning(f"[SCOUT] Standings fetch failed for {competition_code}: {e}")
        return {}

    table_map: dict[int, dict] = {}
    for group in data.get("standings", []):
        if group.get("type") != "TOTAL":
            continue
        for row in group.get("table", []):
            team_id = row.get("team", {}).get("id")
            if team_id is None:
                continue
            table_map[team_id] = {
                "position": row.get("position"),
                "points": row.get("points"),
                "played": row.get("playedGames"),
                "goal_diff": row.get("goalDifference"),
            }
    return table_map


async def fetch_team_recent_form(http: httpx.AsyncClient, team_id: int) -> dict:
    """Fetch team's recent form (W/L/D string).

    Bug fix (Phase 2): original version queried with `limit=5` and had
    no cross-season fallback. Early in the season (matchdays 1-2) teams
    have only 1-2 FINISHED matches, so the form string was just 'W' or 'L'
    and the Analyst flagged 'limited recent form data'. Newly-promoted
    teams have zero. Now: query current season with limit=10; if fewer
    than 3 results come back, fall back to the previous season.
    """
    matches: list = []
    try:
        data = await _fd_get(
            http,
            f"{FOOTBALL_DATA_BASE}/teams/{team_id}/matches",
            params={"status": "FINISHED", "limit": 10},
        )
        matches = list(data.get("matches") or [])
    except Exception as e:
        logger.warning(f"[SCOUT] Recent-form fetch failed for team {team_id}: {e}")
        return {}

    if len(matches) < 3:
        try:
            from datetime import date as _d
            last_season = _d.today().year - 1
            data2 = await _fd_get(
                http,
                f"{FOOTBALL_DATA_BASE}/teams/{team_id}/matches",
                params={"status": "FINISHED", "limit": 10, "season": last_season},
            )
            extra = list(data2.get("matches") or [])
            seen = {m.get("id") for m in matches}
            added = 0
            for mm in extra:
                if mm.get("id") not in seen:
                    matches.append(mm)
                    seen.add(mm.get("id"))
                    added += 1
            if added:
                logger.info(
                    f"[SCOUT] Form fallback for team {team_id}: added "
                    f"{added} matches from season {last_season} "
                    f"(had {len(matches) - added} from current season)."
                )
        except Exception as e:
            logger.warning(f"[SCOUT] Form season-fallback failed for team {team_id}: {e}")

    if not matches:
        return {}

    results = []
    goals_for = 0
    goals_against = 0
    for m in matches[:5]:
        full_time = m.get("score", {}).get("fullTime", {})
        home_goals, away_goals = full_time.get("home"), full_time.get("away")
        if home_goals is None or away_goals is None:
            continue
        is_home = m.get("homeTeam", {}).get("id") == team_id
        gf, ga = (home_goals, away_goals) if is_home else (away_goals, home_goals)
        goals_for += gf
        goals_against += ga
        results.append("W" if gf > ga else "L" if gf < ga else "D")

    if not results:
        return {}
    return {
        "results": "".join(results),
        "matches_considered": len(results),
        "goals_for": goals_for,
        "goals_against": goals_against,
    }


async def fetch_head_to_head(http: httpx.AsyncClient, fixture_id: int) -> dict:
    try:
        data = await _fd_get(
            http,
            f"{FOOTBALL_DATA_BASE}/matches/{fixture_id}/head2head",
            params={"limit": 10},
        )
    except Exception as e:
        logger.warning(f"[SCOUT] Head-to-head fetch failed for fixture {fixture_id}: {e}")
        return {}

    agg = data.get("aggregates", {})
    if not agg:
        return {}
    home = agg.get("homeTeam", {})
    away = agg.get("awayTeam", {})
    return {
        "matches_played": agg.get("numberOfMatches"),
        "home_wins": home.get("wins"),
        "draws": home.get("draws"),
        "away_wins": away.get("wins"),
    }


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


def _strip_accents(text: str) -> str:
    """Lowercase + strip diacritics (München -> munchen)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )


def _normalize_team_name(name: str) -> str:
    """Normalise team names for cross-provider matching.

    Bug fix (Phase 2): the original only stripped trailing suffixes and
    never stripped diacritics, so 'FC Bayern München' (football-data.org)
    never matched 'Bayern Munich' (The Odds API) in _find_match_odds.
    Bayern's odds then silently fell into the 'without odds' bucket
    (only 4/6 fixtures got odds on matchday 1). Fix: strip diacritics and
    leading prefixes like 'FC ', 'SSC ', 'US ', 'AS ', 'AC ' that football-
    data.org includes but The Odds API omits.
    """
    s = _strip_accents(name.strip())
    for prefix in (
        "fc ", "sc ", "ssc ", "us ", "as ", "ac ", "afc ", "bc ", "cf ", "sv ",
        "vfb ", "vfl ", "fsv ", "tsg ", "1. fc ", "if ", "fk ", "ik ", "bk ",
        "ogc ", "losc ", "rc ", "og ", "acf ", "ud ", "cd ", "sd ",
        "internazionale ",
    ):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    for suffix in (
        " fc", " cf", " afc", " cd", " sd", " ac", " bc", " sc", " as",
        " acf", " fcc", " ogc", " afc", " olympique", " club",
        " deportivo", " sporting",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# Aliases for known football-data.org <-> The Odds API name mismatches.
# After both sides are diacritic/prefix-stripped via _normalize_team_name,
# these canonical names are compared. Add an entry when you spot a new
# real-world provider naming mismatch; do NOT add broad fuzzy matching
# (Levenshtein / token-overlap) - the simple substring match below is
# intentional and kept simple to avoid false positives (e.g. "milan"
# inside "inter milan").
_TEAM_NAME_ALIASES = {
    # Bundesliga 2025/26 - promoted/relegated
    "kaiserslautern": "1 fc kaiserslautern",
    "magdeburg": "1 fc magdeburg",
    "nurnberg": "1 fc nurnberg",
    "darmstadt": "sv darmstadt 98",
    "heidenheim": "1 fc heidenheim",
    "hamburg": "hamburger sv",
    # Bundesliga vs Odds API naming differences
    "bayern munich": "fc bayern munchen",
    "fc bayern munchen": "fc bayern munchen",
    "bayer leverkusen": "bayer 04 leverkusen",
    "bayer 04 leverkusen": "bayer 04 leverkusen",
    # Serie A - Inter Milan vs The Odds API's 'FC Internazionale'
    "inter": "fc internazionale",
    "inter milan": "fc internazionale",
    "fc internazionale": "fc internazionale",
    "internazionale": "fc internazionale",
    "milan": "ac milan",
    "ac milan": "ac milan",
    "juventus": "juventus fc",
    "juventus fc": "juventus fc",
    "as roma": "as roma",
    "lazio": "ss lazio",
    "ss lazio": "ss lazio",
    "napoli": "ssc napoli",
    "ssc napoli": "ssc napoli",
    "venezia": "venezia fc",
    "venezia fc": "venezia fc",
    "parma": "parma calcio 1913",
    "parma calcio 1913": "parma calcio 1913",
    # Ligue 1 - short vs long forms
    "brest": "stade brestois 29",
    "stade brestois 29": "stade brestois 29",
    "stade brestois": "stade brestois 29",
    "lille": "losc lille",
    "losc lille": "losc lille",
    "rennes": "stade rennais fc",
    "stade rennais": "stade rennais fc",
    "marseille": "olympique de marseille",
    "olympique de marseille": "olympique de marseille",
    "lyon": "olympique lyonnais",
    "olympique lyonnais": "olympique lyonnais",
    "strasbourg": "rc strasbourg alsace",
    "le havre": "le havre ac",
    "le havre ac": "le havre ac",
    "lens": "rc lens",
    "rc lens": "rc lens",
    # PL - FPL bootstrap uses short nicknames
    "wolves": "wolverhampton wanderers",
    "wolverhampton wanderers": "wolverhampton wanderers",
    "spurs": "tottenham hotspur",
    "tottenham hotspur": "tottenham hotspur",
    "nott'm forest": "nottingham forest",
    "nottingham forest": "nottingham forest",
    "man utd": "manchester united",
    "manchester united": "manchester united",
    "man city": "manchester city",
    "manchester city": "manchester city",
    # Spain - diacritic + suffix variations
    "alaves": "deportivo alaves",
    "deportivo alaves": "deportivo alaves",
    "malaga": "malaga cf",
    "malaga cf": "malaga cf",
    "betis": "real betis",
    "real betis": "real betis",
    "racing santander": "racing club de santander",
    "racing": "racing club de santander",
    # 1. FC Koln vs FC Koln vs Koln
    "koln": "1 fc koln",
    "fc koln": "1 fc koln",
    "1 fc koln": "1 fc koln",
}


def _find_match_odds(odds_events: list[dict], home_name: str, away_name: str) -> dict:
    """Find the odds event matching a fixture.

    Bug fix (Phase 2): the previous version used raw substring containment
    with the *old* normalisation (no diacritics, no leading-prefix strip),
    which is why 'FC Bayern München' never matched 'Bayern Munich' and
    Bayern's odds silently fell off. New approach:
      1. Strip diacritics + leading prefixes via _normalize_team_name.
      2. Resolve provider-specific aliases via _TEAM_NAME_ALIASES.
      3. Match by exact equality OR substring WITH a length-ratio guard
         (shorter must be >= 50% of the longer) to prevent single-word
         false positives like 'milan' matching inside 'inter milan'.
    """
    home_canon = _TEAM_NAME_ALIASES.get(_normalize_team_name(home_name), _normalize_team_name(home_name))
    away_canon = _TEAM_NAME_ALIASES.get(_normalize_team_name(away_name), _normalize_team_name(away_name))
    for event in odds_events:
        eh = _TEAM_NAME_ALIASES.get(
            _normalize_team_name(event.get("home_team", "")),
            _normalize_team_name(event.get("home_team", "")),
        )
        ea = _TEAM_NAME_ALIASES.get(
            _normalize_team_name(event.get("away_team", "")),
            _normalize_team_name(event.get("away_team", "")),
        )
        # home match: equal OR substring with ratio guard (>=50%)
        if not _match_strong(home_canon, eh):
            continue
        if not _match_strong(away_canon, ea):
            continue
        return event
    return {}


def _match_strong(a: str, b: str) -> bool:
    """Match a and b by equality or substring containment with a 50% length-
    ratio guard. Prevents 'milan' matching inside 'inter milan' (5/11 = 0.45
    < 0.5) while still matching 'bayern munchen' inside itself."""
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 0.5 * len(long):
        return False
    return short in long


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
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        text = text.strip()
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


def analyze_with_gemini(raw_data: dict) -> dict:
    # A RuntimeError here (Gemini rate-limited across the whole fallback chain,
    # with a wait too long to block on — see GEMINI_MAX_SINGLE_WAIT_SECONDS)
    # previously propagated straight out of this function, through
    # _deep_analyze, and out of the batch loop in run() — crashing the
    # entire Scout phase and discarding every fixture already analyzed
    # earlier in the same run, each of which cost several real
    # football-data.org calls to assemble. Converting it to {} here instead
    # routes it through the exact same "couldn't get anything usable from
    # the LLM" path _deep_analyze already has for a parse failure (see the
    # skeleton-building fallback there) — so one fixture Gemini can't serve
    # right now costs just that fixture, not the whole batch.
    try:
        text = _gemini_chat(
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
  "form": {{"home": str, "away": str}},
  "form_source": "real|UNKNOWN",
  "head_to_head": {{"record": str, "matches_played": int}},
  "standings_gap": {{"home_position": int, "away_position": int, "points_gap": int}},
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
    except RuntimeError as e:
        logger.error(f"[SCOUT] analyze_with_gemini: Gemini unavailable ({e}) — returning empty result.")
        return {}
    result = _extract_json(text)

    if isinstance(result, list):
        logger.warning("[SCOUT] LLM returned a list instead of a dict — unwrapping first element.")
        result = result[0] if result and isinstance(result[0], dict) else {}

    if not isinstance(result, dict):
        logger.error(
            f"[SCOUT] Failed to parse LLM response as JSON object "
            f"(finish_reason={get_last_finish_reason()}): {text[:200]}"
        )
        return {}

    return result


MIN_QUALIFYING_MATCHES = 2
QUALIFYING_COMPLETENESS = CLEANER_THRESHOLD_ENV
HARD_CAP_MATCHES = 18


async def _deep_analyze(
    fixture: dict,
    odds_event: dict,
    epl_injuries: dict,
    fd_http: httpx.AsyncClient,
    standings_cache: dict[str, dict],
    team_form_cache: dict[int, dict],
) -> dict:
    fixture_id = fixture["id"]
    home_name = fixture["homeTeam"]["name"]
    away_name = fixture["awayTeam"]["name"]
    competition_code = fixture.get("_competition_code")
    home_id = fixture.get("homeTeam", {}).get("id")
    away_id = fixture.get("awayTeam", {}).get("id")

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

    if ENABLE_REAL_FORM_DATA:
        if competition_code and competition_code not in standings_cache:
            standings_cache[competition_code] = await fetch_standings(fd_http, competition_code)
        standings = standings_cache.get(competition_code, {}) if competition_code else {}
        home_standing = standings.get(home_id, {}) if home_id is not None else {}
        away_standing = standings.get(away_id, {}) if away_id is not None else {}

        if home_id is not None and home_id not in team_form_cache:
            team_form_cache[home_id] = await fetch_team_recent_form(fd_http, home_id)
        if away_id is not None and away_id not in team_form_cache:
            team_form_cache[away_id] = await fetch_team_recent_form(fd_http, away_id)
        home_form = team_form_cache.get(home_id, {}) if home_id is not None else {}
        away_form = team_form_cache.get(away_id, {}) if away_id is not None else {}

        head_to_head = await fetch_head_to_head(fd_http, fixture_id)
    else:
        home_standing = away_standing = home_form = away_form = head_to_head = {}

    raw = {
        "fixture": fixture,
        "odds": odds_snapshot,
        "home_injuries": home_injuries,
        "away_injuries": away_injuries,
        "weather": weather,
        "venue_city": venue_city,
        "recent_form": {"home": home_form, "away": away_form},
        "head_to_head": head_to_head,
        "standings": {"home": home_standing, "away": away_standing},
    }

    structured = analyze_with_gemini(raw)

    if not structured:
        logger.warning(f"[SCOUT] analyze_with_gemini returned empty result for {home_name} vs {away_name} — using skeleton.")
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
            score += 0.3
        if home_injuries or away_injuries:
            score += 0.2
        if odds_snapshot:
            score += 0.2
        if weather.get("temp_c") is not None:
            score += 0.1
        if home_form or away_form:
            score += 0.1
        if head_to_head:
            score += 0.1
        structured["data_completeness"] = round(score, 2)

    # Rescue clause (Phase 2): when football-data.org gave us rich standings +
    # odds + weather, the LLM is allowed to report a conservative
    # completeness (early-season form/H2H often come back UNKNOWN).
    # Take the max of the LLM's number and our deterministic count so a
    # well-stocked match is never artificially deflated (e.g. Bayern vs
    # Stuttgart: LLM reported 0.40 because form/H2H were UNKNOWN, but
    # local data clearly merits >= 0.65).
    local_count = 0
    if odds_snapshot:
        local_count += 1
    if home_standing or away_standing:
        local_count += 1
    if home_form or away_form:
        local_count += 1
    if head_to_head:
        local_count += 1
    if weather.get("temp_c") is not None:
        local_count += 1
    if local_count >= 4:
        floor = 0.65
        if (home_standing and away_standing
                and home_standing.get("position") and away_standing.get("position")):
            floor = max(floor, 0.75)
        if (head_to_head and head_to_head.get("matches_played")
                and home_form and away_form
                and home_form.get("results") and away_form.get("results")):
            floor = max(floor, 0.85)
        if structured["data_completeness"] < floor:
            logger.info(
                f"[SCOUT] LLM reported completeness "
                f"{structured['data_completeness']} but local data covers "
                f"{local_count}/5 evidence categories - bumping to {floor}."
            )
            structured["data_completeness"] = floor

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

    standings_cache: dict[str, dict] = {}
    team_form_cache: dict[int, dict] = {}

    async with httpx.AsyncClient(timeout=20) as fd_http:
        while cursor < len(prioritized) and len(results) < HARD_CAP_MATCHES:
            batch_size = MAX_MATCHES_PER_DAY if is_first_batch else FALLBACK_BATCH_SIZE
            remaining_budget = HARD_CAP_MATCHES - len(results)
            batch_size = min(batch_size, remaining_budget)

            batch = prioritized[cursor:cursor + batch_size]
            cursor += batch_size
            is_first_batch = False

            logger.info(f"[SCOUT] Analyzing batch of {len(batch)} fixture(s) (processed so far: {len(results)})...")

            for fixture, odds_event in batch:
                structured = await _deep_analyze(
                    fixture, odds_event, epl_injuries, fd_http, standings_cache, team_form_cache
                )
                results.append(structured)

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
