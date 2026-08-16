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

# football-data.org competition codes for the top 5 European leagues + UCL.
# https://docs.football-data.org/general/v4/competitions.html
# This is the DEFAULT set used whenever LEAGUE_IDS is not set in the
# environment — i.e. normal daily production runs.
DEFAULT_LEAGUE_IDS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "CL": "UEFA Champions League",
}

# Friendly names for the OTHER competition codes covered by
# football-data.org's free tier, used only for nicer log output when
# testing with a LEAGUE_IDS override below. NOTE: unlike the old
# API-Football version, this is now a closed list — the free tier only
# has these 12 competitions, so an override outside this set will 404.
KNOWN_LEAGUE_NAMES = {
    **DEFAULT_LEAGUE_IDS,
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "ELC": "Championship (England)",
    "BSA": "Campeonato Brasileiro Série A",
    "WC": "FIFA World Cup",
    "EC": "European Championship",
}

# Maps a football-data.org competition code to the sport key The Odds API
# uses for that same competition. Only competitions in this dict will get
# odds fetched — anything else (e.g. an ELC/BSA override) simply won't
# have odds available, and degrades gracefully like any other missing field.
ODDS_SPORT_KEYS = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "CL": "soccer_uefa_champs_league",
}

# A handful of club names that are spelled differently between
# football-data.org (usually the formal/legal name, e.g. "Newcastle
# United FC") and the Fantasy Premier League API (usually the short
# common name, e.g. "Newcastle"). Only needed for injury matching, since
# that's the only place we cross-reference two providers by name instead
# of by a shared ID. Add to this if a new mismatch turns up in testing.
FPL_NAME_ALIASES = {
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham Hotspur",
    "Man Utd": "Manchester United",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Leeds": "Leeds United",
    "Wolves": "Wolverhampton Wanderers",
}


def _load_league_ids() -> dict[str, str]:
    """
    Reads the LEAGUE_IDS env var (comma-separated football-data.org
    competition codes, e.g. "PL,PD,DED") if set, otherwise falls back to
    the normal top-5 + UCL default. This is meant for one-off testing
    runs — e.g. to confirm the pipeline works end-to-end on a day when
    top-5+UCL happens to have no fixtures — without editing this file
    back and forth. For normal daily runs, just leave LEAGUE_IDS unset.

    NOTE: codes must be from football-data.org's free-tier competition
    list (see KNOWN_LEAGUE_NAMES) — unlike the old API-Football numeric
    IDs, these aren't arbitrary; an unsupported code will 404 at fetch time.
    """
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
    """
    Reads a float-valued env var safely. Treats both "unset" AND
    "set but empty string" (e.g. a workflow env: block with a blank
    value, or an unpopulated repo variable/secret) as "use the default"
    instead of crashing with ValueError: could not convert string to
    float: ''. Also guards against a non-numeric value being pasted in
    by mistake — logs a warning and falls back rather than crashing the
    whole pipeline over one bad env var.
    """
    raw = os.environ.get(name, "")
    raw = raw.strip() if raw else ""
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[SCOUT] Env var {name}='{raw}' is not a valid float — using default {default}.")
        return default


# Minimum data_completeness for a match to count as "qualifying" during
# Scout's own batch-sizing decisions (see MIN_QUALIFYING_MATCHES below).
# Overridable via CLEANER_THRESHOLD so a testing run that loosens the
# Cleaner's bar (in pipeline.py) doesn't leave Scout still pulling extra
# fallback batches against the old, stricter default.
CLEANER_THRESHOLD_ENV = _float_env("CLEANER_THRESHOLD", 0.5)

# Size of the FIRST analysis batch. Each fully-analyzed match costs:
# 1 weather call (OpenWeather) + 1 Groq call for Scout structuring
# (injuries and odds are now fetched once per COMPETITION up front, not
# per match — see run()). Kept at 12 (not equal to HARD_CAP_MATCHES) so
# the fallback batch below actually has room to run if needed.
MAX_MATCHES_PER_DAY = 12

# Size of each FALLBACK batch, pulled in only if the first batch didn't
# yield enough qualifying matches. Smaller than the first batch to keep
# total token usage predictable.
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
    """
    Fetch fixtures for the target date, filtered to top leagues, via
    football-data.org. Each returned match dict is tagged with
    "_competition_code" so downstream code (run(), _deep_analyze) knows
    which competition it belongs to without re-deriving it.

    NOTE: no season-year parameter needed here — football-data.org
    filters by date range directly, which is what eliminated the whole
    class of "season=2026 has no data yet" bug we had with API-Football.
    """
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
            # football-data.org free tier is 10 req/min — 6 sequential
            # calls comfortably clears that even without this, but this
            # keeps it safe if more competitions get added later.
            await asyncio.sleep(1)

    return all_matches


async def fetch_league_odds(sport_key: str) -> list[dict]:
    """
    Fetch odds for EVERY upcoming fixture in one competition, in a
    single call — this is the free-tier-friendly shape of The Odds API:
    you pay per (market x region), not per fixture. Returns [] on any
    failure (missing/invalid key, competition out of season, etc.) so a
    problem with one competition's odds never blocks the whole run.
    """
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
    """
    Strips common club-name suffixes and normalizes '&' vs 'and' so names
    from different providers ("Arsenal FC" vs "Arsenal", "Brighton &
    Hove Albion FC" vs "Brighton and Hove Albion") compare equal. The
    '&'/'and' case was found via a real live test against The Odds API,
    not a guess — football-data.org uses '&', The Odds API spells it out.
    Not exhaustive — this is fuzzy matching, not an ID join, so an
    unusual club name could still occasionally fail to match. If that
    happens in testing, add the specific case to FPL_NAME_ALIASES (for
    injuries) or extend this function (for odds).
    """
    name = name.strip()
    for suffix in (" FC", " CF", " AFC", " CD", " SD", " AC"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = name.replace("&", "and")
    name = re.sub(r"\s+", " ", name)
    return name.strip().lower()


def _find_match_odds(odds_events: list[dict], home_name: str, away_name: str) -> dict:
    """Matches one football-data.org fixture to its Odds API event by team name."""
    home_norm = _normalize_team_name(home_name)
    away_norm = _normalize_team_name(away_name)
    for event in odds_events:
        eh = _normalize_team_name(event.get("home_team", ""))
        ea = _normalize_team_name(event.get("away_team", ""))
        if (eh in home_norm or home_norm in eh) and (ea in away_norm or away_norm in ea):
            return event
    return {}


def _extract_odds_snapshot(odds_event: dict) -> dict:
    """
    Best-effort extraction of home/draw/away/over25 odds from The Odds
    API's event structure. Picks whichever bookmaker appears first in
    the response (free-tier bookmaker availability varies by
    event/region, so there's no fixed "always use this book" choice
    the way the old API-Football version pinned to Bet365).

    btts_yes is intentionally left unset — this provider's free
    bookmaker set doesn't reliably carry a BTTS market. Same as any
    other genuinely-missing field, the LLM marks it UNKNOWN downstream.
    """
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
    """
    Fetches current player availability for ALL Premier League clubs in
    ONE call to the official Fantasy Premier League API (free, no key,
    no rate limit in practice). Returns a dict keyed by FPL's own team
    name, mapping to a list of unavailable players.

    KNOWN LIMITATION: this only covers the Premier League. La Liga,
    Bundesliga, Serie A, Ligue 1, and UCL fixtures will get an empty
    injuries list — Scout's existing completeness scoring already
    handles missing injury data gracefully, so this doesn't break
    anything, it just means those matches score lower on completeness
    than they would with a real injury feed. No free equivalent exists
    for the other leagues as of this writing (see prior research).
    """
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
        if player.get("status") == "a":  # available — nothing to report
            continue
        team_name = teams_by_id.get(player.get("team"))
        if not team_name:
            continue
        injuries_by_team.setdefault(team_name, []).append(
            {
                "player": player.get("web_name"),
                "status": player.get("status"),  # i=injured, s=suspended, d=doubtful, u=unavailable
                "news": player.get("news") or "No details provided",
                "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
            }
        )
    return injuries_by_team


def _lookup_epl_injuries(injuries_by_team: dict, fd_team_name: str) -> list[dict]:
    """Matches a football-data.org team name to FPL's injuries dict by name."""
    target = _normalize_team_name(fd_team_name)
    for fpl_name, injuries in injuries_by_team.items():
        candidate = _normalize_team_name(FPL_NAME_ALIASES.get(fpl_name, fpl_name))
        if candidate == target or candidate in target or target in candidate:
            return injuries
    return []


async def fetch_weather(venue_city: str | None) -> dict:
    """
    Fetch weather forecast for match venue. Returns blank dict if no
    city known.

    NOTE: football-data.org's match objects don't reliably include a
    venue city the way API-Football's did, so venue_city will often be
    None now — this function already degrades gracefully in that case,
    same as before, so nothing breaks. A team->home-city lookup table
    would fix this properly; treating that as a separate follow-up
    rather than bundling it into this provider swap.
    """
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


# Minimum matches that must reach a usable completeness score before Scout
# stops pulling in more candidates. Mirrors the Cleaner's own bar (see
# pipeline.py CLEANER_THRESHOLD) — both now read from the same
# CLEANER_THRESHOLD env var (default 0.5), so Scout never stops short of
# what the Cleaner would have accepted anyway, even during a testing run
# that loosens the threshold.
MIN_QUALIFYING_MATCHES = 2
QUALIFYING_COMPLETENESS = CLEANER_THRESHOLD_ENV

# Absolute ceiling on how many matches Scout will ever fully analyze in one
# run (first batch + fallback batch combined). MUST be greater than
# MAX_MATCHES_PER_DAY or the fallback can never actually trigger — see
# budget note in run() for why 18 is the safe ceiling given Groq's
# 100,000 tokens/day free limit.
HARD_CAP_MATCHES = 18


async def _deep_analyze(fixture: dict, odds_event: dict, epl_injuries: dict) -> dict:
    """
    Run the full injuries+weather+LLM analysis for one fixture. Odds and
    injuries are passed in already-fetched (one call per competition,
    made once in run()) rather than fetched here per-match — this is the
    same "don't re-fetch per fixture" principle the old API-Football
    version used, just applied at the competition level instead of the
    fixture level, since that's how the new providers are shaped.
    """
    fixture_id = fixture["id"]
    home_name = fixture["homeTeam"]["name"]
    away_name = fixture["awayTeam"]["name"]
    competition_code = fixture.get("_competition_code")
    venue_city = None  # see fetch_weather() docstring for why

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

    # Fallback completeness score, odds weighted lower (often post late)
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

    Strategy:
      1. Fetch fixtures for the target date across configured competitions
         (football-data.org, one call per competition).
      2. Fetch odds for each competition PRESENT in today's fixtures, in
         ONE call per competition (The Odds API) — not per fixture.
      3. Fetch Premier League injuries in ONE call (FPL), if PL fixtures
         are present today. Other leagues get no injury data (see
         fetch_epl_injuries docstring).
      4. Match each fixture to its odds by team name, prioritize
         odds-matched fixtures first, then run full deep analysis
         (injuries+weather+LLM) on the best-prioritized batch of
         MAX_MATCHES_PER_DAY. If fewer than MIN_QUALIFYING_MATCHES end up
         with usable completeness, automatically pulls in ONE additional
         fallback batch of FALLBACK_BATCH_SIZE from the remaining pool —
         up to HARD_CAP_MATCHES total — rather than silently giving up.

    Budget check (worst case, both batches fully used = HARD_CAP_MATCHES=18):
      football-data.org (free tier: 10 req/min, competitions accessible
      are the 12-competition free set):
        - 6 calls: fixture list, one per top competition, ~1/sec spaced
      The Odds API (free tier: 500 credits/month):
        - up to 6 calls: one per competition PRESENT today, at
          2 credits each (h2h + totals markets, 1 region) = up to 12
          credits/day ≈ 360/month — comfortably under the 500 cap, even
          allowing room for occasional re-runs.
      Fantasy Premier League API (free, no key, no practical rate limit):
        - 1 call, only if PL fixtures are present today.
      Groq (free tier: 100,000 tokens/day):
        - ~2,000 tokens/match for Scout structuring x 18 = ~36,000
        - ~1,350 tokens/match for Analyst x 18 = ~24,300
        - ~1,050 tokens/match for Risk x 18 = ~18,900
        - ~8,650 tokens flat for Portfolio+Auditor+Decision+Publisher
        Total: ~87,850 tokens — under 100,000, but with less margin.
        NOTE: this is the budget for ONE scheduled daily run. Manually
        re-triggering the workflow multiple times on the same day (e.g.
        for testing) can still exhaust the daily Groq quota — this was
        observed during development testing, not a bug in the logic.
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

    # Match each fixture to its odds event (if any), and prioritize
    # odds-matched fixtures first — same reasoning as before: they're
    # far more likely to yield a complete analysis.
    scanned = []
    for fixture in fixtures:
        events = odds_by_league.get(fixture["_competition_code"], [])
        matched = _find_match_odds(events, fixture["homeTeam"]["name"], fixture["awayTeam"]["name"])
        scanned.append((fixture, matched))

    with_odds = [(f, o) for f, o in scanned if o]
    without_odds = [(f, o) for f, o in scanned if not o]
    prioritized = with_odds + without_odds

    logger.info(
        f"[SCOUT] {len(with_odds)}/{len(scanned)} fixtures matched to odds. Prioritizing those first."
    )

    # ── Deep analysis — first batch, then fallback if needed ──
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
            await asyncio.sleep(2.5)  # Groq free tier: confirmed 30 req/min cap for
            # llama-3.3-70b-versatile — 2.5s clears the ~2s minimum spacing with margin

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
                f"[SCOUT] Only {qualifying} qualifying match(es), but no more fixtures or budget "
                f"remaining to pull further batches."
            )

    logger.info(f"[SCOUT] Collected {len(results)} match intelligence packages total.")
    return results
