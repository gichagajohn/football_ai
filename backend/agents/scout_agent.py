"""
SCOUT AGENT — Football Pulse AI (GitHub Actions edition)
Pulls fixtures, odds, form, injuries, lineups, weather, travel, market movement.

Filters to top European leagues for better data quality/completeness.

Uses a two-pass strategy: a cheap odds-only pre-scan across ALL fixtures
found (to prioritize which ones are likely to yield complete data), then
full deep analysis (odds+injuries+weather+LLM) on the best batch first —
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

# API-Football league IDs for the top 5 European leagues + Champions League.
# https://www.api-football.com/documentation-v3#tag/Leagues
# This is the DEFAULT set used whenever LEAGUE_IDS is not set in the
# environment — i.e. normal daily production runs.
DEFAULT_LEAGUE_IDS = {
    39: "Premier League",
    140: "La Liga",
    78: "Bundesliga",
    135: "Serie A",
    61: "Ligue 1",
    2: "UEFA Champions League",
}

# Friendly names for other common API-Football league IDs, used only for
# nicer log output when testing with a LEAGUE_IDS override below. Not
# exhaustive — any ID not listed here just logs as "League <id>", which is
# harmless, it's purely cosmetic.
KNOWN_LEAGUE_NAMES = {
    **DEFAULT_LEAGUE_IDS,
    3: "UEFA Europa League",
    848: "UEFA Europa Conference League",
    88: "Eredivisie",
    94: "Primeira Liga",
    203: "Süper Lig",
    144: "Belgian Pro League",
    40: "Championship (England)",
    253: "MLS",
    71: "Brasileirão",
    128: "Liga Profesional (Argentina)",
}


def _load_league_ids() -> dict[int, str]:
    """
    Reads the LEAGUE_IDS env var (comma-separated API-Football league IDs,
    e.g. "39,140,88,203") if set, otherwise falls back to the normal
    top-5 + UCL default. This is meant for one-off testing runs — e.g.
    to confirm the pipeline works end-to-end on a day when top-5+UCL
    happens to have no fixtures — without editing this file back and
    forth. For normal daily runs, just leave LEAGUE_IDS unset.
    """
    override = os.environ.get("LEAGUE_IDS", "").strip()
    if not override:
        return DEFAULT_LEAGUE_IDS

    ids: list[int] = []
    for part in override.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            logger.warning(f"[SCOUT] Ignoring invalid league id in LEAGUE_IDS: '{part}'")

    if not ids:
        logger.warning("[SCOUT] LEAGUE_IDS was set but contained no valid ids — falling back to default top-5+UCL.")
        return DEFAULT_LEAGUE_IDS

    result = {i: KNOWN_LEAGUE_NAMES.get(i, f"League {i}") for i in ids}
    logger.info(f"[SCOUT] LEAGUE_IDS override active — using {len(result)} league(s): {list(result.values())}")
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
# 1 injury pair + 1 weather call (API-Football/OpenWeather) + 1 Groq call
# for Scout structuring. Kept at 12 (not equal to HARD_CAP_MATCHES) so the
# fallback batch below actually has room to run if needed.
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


async def quick_odds_check(fixture_id: int) -> dict:
    """
    Cheap pre-scan: fetches odds once and returns them directly, so the
    deep-analysis pass can reuse this result instead of fetching odds
    again. Returns {} if no odds are posted yet for this fixture.
    """
    async with httpx.AsyncClient(timeout=10) as http:
        try:
            resp = await http.get(
                "https://v3.football.api-sports.io/odds",
                headers={"x-apisports-key": os.environ.get("API_FOOTBALL_KEY", "")},
                params={"fixture": fixture_id, "bookmaker": 6},
            )
            resp.raise_for_status()
            odds_data = resp.json().get("response", [])
            return odds_data[0] if odds_data else {}
        except Exception:
            return {}


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

# Cap on how many fixtures get the cheap odds pre-scan. On a huge day
# (50-60+ fixtures across all top leagues) scanning literally all of them
# would itself burn most of the daily API-Football budget before deep
# analysis even starts.
PRESCAN_CAP = 30


async def _deep_analyze(fixture: dict, prefetched_odds: dict) -> dict:
    """
    Run the full injuries+weather+LLM analysis for one fixture. Odds are
    passed in from the pre-scan pass (quick_odds_check) rather than
    re-fetched here, to avoid burning a second API-Football call per match.
    """
    fixture_id = fixture["fixture"]["id"]
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    venue_city = fixture["fixture"].get("venue", {}).get("city") or None

    odds = prefetched_odds
    home_injuries, away_injuries, weather = await asyncio.gather(
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
    if not structured.get("odds_snapshot") and odds:
        structured["odds_snapshot"] = _extract_odds_snapshot(odds)

    logger.info(
        f"[SCOUT] ✓ {structured.get('home_team', '?')} vs {structured.get('away_team', '?')} "
        f"({structured.get('league', '?')}) — completeness={structured.get('data_completeness')}"
    )
    return structured


async def run(target_date: date | None = None) -> list[dict]:
    """
    Main scout agent entrypoint. Returns list of structured match intelligence.

    Two-pass strategy:
      1. Cheap pre-scan: fetch odds for up to PRESCAN_CAP fixtures found
         (no injuries/weather/LLM calls yet). Fixtures WITH odds are
         prioritized, since they're far more likely to yield a complete
         analysis. Odds are cached here and reused in pass 2 — never
         fetched twice for the same fixture.
      2. Full deep analysis (injuries+weather+LLM, reusing pre-scanned
         odds) runs on the best-prioritized batch of MAX_MATCHES_PER_DAY
         first. If fewer than MIN_QUALIFYING_MATCHES end up with usable
         completeness, automatically pulls in ONE additional fallback
         batch of FALLBACK_BATCH_SIZE from the remaining pool — up to
         HARD_CAP_MATCHES total — rather than silently giving up.

    Budget check (worst case, both batches fully used = HARD_CAP_MATCHES=18):
      API-Football (free tier: 100 calls/day):
        - 6 calls: fixture list, one per top league
        - up to 30 calls: pre-scan (PRESCAN_CAP), odds only
        - up to 36 calls: deep analysis (18 matches x 2 injury calls;
          odds reused from pre-scan, not re-fetched)
        Total: 6 + 30 + 36 = 72 calls — safely under 100.
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

    prescan_pool = fixtures[:PRESCAN_CAP]
    if len(fixtures) > PRESCAN_CAP:
        logger.info(
            f"[SCOUT] {len(fixtures)} fixtures found — capping pre-scan to first "
            f"{PRESCAN_CAP} to stay within API budget."
        )

    logger.info(f"[SCOUT] Running quick odds pre-scan on {len(prescan_pool)} fixture(s)...")

    # ── Pass 1: cheap odds pre-scan (cached for reuse in pass 2) ────
    odds_cache: dict[int, dict] = {}
    scanned = []
    for fixture in prescan_pool:
        fixture_id = fixture["fixture"]["id"]
        odds = await quick_odds_check(fixture_id)
        odds_cache[fixture_id] = odds
        scanned.append((fixture, bool(odds)))
        await asyncio.sleep(0.3)  # light throttle, this is a cheap call

    with_odds = [f for f, has_odds in scanned if has_odds]
    without_odds = [f for f, has_odds in scanned if not has_odds]
    prioritized = with_odds + without_odds  # odds-available fixtures go first

    logger.info(
        f"[SCOUT] Pre-scan complete: {len(with_odds)}/{len(scanned)} fixtures have odds posted. "
        f"Prioritizing those first."
    )

    # ── Pass 2: full deep analysis — first batch, then fallback if needed ──
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

        for fixture in batch:
            fixture_id = fixture["fixture"]["id"]
            prefetched_odds = odds_cache.get(fixture_id, {})
            structured = await _deep_analyze(fixture, prefetched_odds)
            results.append(structured)
            await asyncio.sleep(1)  # ease rate limits on injuries/weather/Groq

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
