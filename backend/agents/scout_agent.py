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
from datetime import date
from typing import Any

import httpx
from groq import Groq, NotFoundError, RateLimitError

logger = logging.getLogger(__name__)
# max_retries=0: we do our own TPM-aware pacing + retry (see _groq_chat below)
# instead of relying on the SDK's blind exponential backoff.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"), max_retries=0)

# Model fallback chain: on a daily-quota 429 (RPD/TPD) we switch to the next
# model rather than sleeping out a multi-minute daily cooldown, since each
# model on Groq carries its own independent daily budget.
#
# NOTE: Groq deprecates/retires models on short notice (llama-3.1-8b-instant
# and llama-3.3-70b-versatile were both shut down Aug 16, 2026). If a model
# here starts 404ing, check https://console.groq.com/docs/models for the
# current lineup and update this list — the code below treats a 404 as
# "unusable, skip it" rather than crashing, but a stale chain still means
# fewer real fallback options in practice.
GROQ_MODEL_FALLBACK_CHAIN = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
]
_current_model_index = 0

# ---------------------------------------------------------------------------
# Reasoning control.
#
# All three models in the chain are *reasoning* models. Left on default
# settings they write out a full chain-of-thought as plain text before (or
# instead of) answering, and that reasoning counts as ordinary output
# tokens — it burns through the TPM budget fast, regularly blows past
# max_tokens (triggering the truncation-retry path over and over), and on
# qwen specifically can consume the entire response and hit
# finish_reason="stop" having never emitted the requested JSON at all,
# which _extract_json then correctly fails to parse.
#
# All three models support reasoning_effort (per console.groq.com/docs/
# reasoning and the qwen3.6-27b model card, checked 2026-08-22), but the
# valid values differ by family:
#   - gpt-oss (20b/120b): "low" | "medium" | "high" — reasoning is always on
#     to some degree and can't be fully disabled, so "low" is the floor.
#   - qwen3.6-27b: "none" | "default" — "none" is genuine non-thinking mode
#     per Groq's own model card ("use non-thinking mode (reasoning_effort=
#     'none') for efficient, general-purpose dialogue").
#
# Scout's job here (turn raw fixture data into structured JSON) is bounded
# structured-output work, not open-ended reasoning, so "minimal" is used
# for every call.
_REASONING_EFFORT_BY_FAMILY: dict[str, dict[str, str]] = {
    "gpt-oss": {"minimal": "low", "default": "medium"},
    "qwen": {"minimal": "none", "default": "default"},
}


def _reasoning_extra_body(model: str, desired: str) -> dict[str, str]:
    """Build the extra_body payload controlling reasoning for `model`.

    Returns {} for any model not in _REASONING_EFFORT_BY_FAMILY (i.e. a
    future non-reasoning model added to the chain) so we never send a
    reasoning param a model doesn't understand.
    """
    for family, effort_map in _REASONING_EFFORT_BY_FAMILY.items():
        if family in model:
            effort = effort_map.get(desired, effort_map["minimal"])
            return {"reasoning_effort": effort, "reasoning_format": "hidden"}
    return {}


# Set by _groq_chat right before returning, so callers can enrich their own
# parse-error logs with *why* the content was empty without changing
# _groq_chat's return type everywhere it's called.
_last_finish_reason: str | None = None


def _current_model() -> str:
    return GROQ_MODEL_FALLBACK_CHAIN[_current_model_index]


def _switch_to_next_model() -> bool:
    global _current_model_index
    if _current_model_index >= len(GROQ_MODEL_FALLBACK_CHAIN) - 1:
        return False
    _current_model_index += 1
    logger.warning(
        f"[FALLBACK] Switching to {_current_model()}"
    )
    return True


def _is_daily_limit_error(err: RateLimitError) -> bool:
    msg = str(err).lower()
    return "per day" in msg or "tpd" in msg or "rpd" in msg


# ---------------------------------------------------------------------------
# Rate limit configuration
#
# Verified against https://console.groq.com/docs/rate-limits (checked
# 2026-08-22). All three models currently in GROQ_MODEL_FALLBACK_CHAIN share
# the same free-tier limits:
#
#     RPM=30   RPD=1,000   TPM=8,000   TPD=200,000
#
# IMPORTANT: this file previously assumed a flat 18,000 TPM budget
# (GROQ_TPM_LIMIT env default) for every model — more than double the real
# 8,000 TPM cap. That mismatch is exactly why the daily run was getting
# 429'd on almost every call even with pacing in place: the pacer thought
# it had headroom it didn't actually have.
#
# Two things Groq's response headers do and don't tell you:
#   - x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens: TPM, and it's
#     authoritative — we read these off every response and prefer them over
#     our own estimate whenever they're fresh.
#   - RPM is NOT exposed in headers at all (only RPD is, confusingly under
#     the name x-ratelimit-remaining-requests). So RPM has to be paced
#     locally from a rolling request-timestamp log; there's no way to read
#     the real-time figure from Groq directly.
#
# If Groq changes these limits, or the fallback chain gains a model with
# different limits, update MODEL_LIMITS accordingly — check the current
# table at console.groq.com/docs/rate-limits.
MODEL_LIMITS: dict[str, dict[str, int]] = {
    "openai/gpt-oss-20b": {"rpm": 30, "tpm": 8000},
    "openai/gpt-oss-120b": {"rpm": 30, "tpm": 8000},
    "qwen/qwen3.6-27b": {"rpm": 30, "tpm": 8000},
}
_DEFAULT_MODEL_LIMITS = {"rpm": 30, "tpm": 8000}

# Safety margin: pace against this fraction of the documented cap, not the
# full cap, so normal jitter/estimation error doesn't still land us on a 429.
GROQ_SAFETY_MARGIN = 0.85

GROQ_TPM_WINDOW_SECONDS = 60
GROQ_RPM_WINDOW_SECONDS = 60

# Local retries *per model* before we fall forward to the next model in the
# chain (if any remain). Previously this was a flat 4 retries with no
# forward-fallback on plain rate-limit exhaustion — only on errors that
# explicitly said "per day" — which meant a model stuck in a tight
# per-minute rate-limit loop could exhaust all retries and crash the whole
# scout run even while other models in the chain sat completely unused.
GROQ_MAX_LOCAL_RETRIES = 4

# Ceiling for the truncation retry in _groq_chat. Reasoning models can run
# out of max_tokens two different ways: (1) spend the entire budget on
# hidden chain-of-thought and return empty content, or (2) spend it
# mid-answer and return a cut-off, non-empty-but-invalid partial response.
# Both show up as finish_reason="length" — content emptiness is not a
# reliable signal on its own, so the retry keys off finish_reason alone.
GROQ_TRUNCATION_RETRY_CEILING = 6000


# ---------------------------------------------------------------------------
# Per-model rate limit state, keyed by model name so switching back and
# forth in the chain doesn't lose pacing history for a given model.
class _ModelRateState:
    __slots__ = (
        "request_times",
        "token_log",
        "header_remaining_tokens",
        "header_reset_tokens_at",
    )

    def __init__(self) -> None:
        self.request_times: list[float] = []
        self.token_log: list[tuple[float, int]] = []
        self.header_remaining_tokens: int | None = None
        self.header_reset_tokens_at: float | None = None  # absolute time.time()


_model_states: dict[str, _ModelRateState] = {}


def _state_for(model: str) -> _ModelRateState:
    if model not in _model_states:
        _model_states[model] = _ModelRateState()
    return _model_states[model]


def _limits_for(model: str) -> dict[str, int]:
    return MODEL_LIMITS.get(model, _DEFAULT_MODEL_LIMITS)


def _parse_reset_duration(value: str | None) -> float:
    """Parse Groq's reset-time header format into seconds.

    Observed formats: "7.66s", "2m59.56s", "120ms", "1.2s".
    Returns 0.0 if the value is missing or unparseable.
    """
    if not value:
        return 0.0
    value = value.strip()
    if value.endswith("ms"):
        try:
            return float(value[:-2]) / 1000.0
        except ValueError:
            return 0.0
    m = re.match(r"^(?:(\d+)m)?(?:([\d.]+)s)?$", value)
    if not m or (m.group(1) is None and m.group(2) is None):
        return 0.0
    minutes = float(m.group(1)) if m.group(1) else 0.0
    seconds = float(m.group(2)) if m.group(2) else 0.0
    return minutes * 60 + seconds


def _record_headers(model: str, headers: Any) -> None:
    """Pull the authoritative TPM figures Groq hands back on every response
    (success or 429) so the next _pace_before_call for this model can use
    real numbers instead of only a local estimate."""
    if headers is None:
        return
    state = _state_for(model)
    try:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = headers.get("x-ratelimit-reset-tokens")
    except AttributeError:
        return
    if remaining is None:
        return
    try:
        state.header_remaining_tokens = int(remaining)
    except (TypeError, ValueError):
        return
    reset_seconds = _parse_reset_duration(reset)
    state.header_reset_tokens_at = time.time() + reset_seconds


def _prune(log: list, cutoff: float, key=lambda item: item) -> None:
    while log and key(log[0]) < cutoff:
        log.pop(0)


def _pace_before_call(model: str, estimated_tokens: int) -> None:
    """Sleep only if firing now would push us over this model's RPM or TPM
    budget. Prefers Groq's own header-reported TPM figures when fresh;
    falls back to a local rolling-window estimate otherwise (e.g. on the
    first call to a model, before we've seen any headers for it yet).
    """
    now = time.time()
    state = _state_for(model)
    limits = _limits_for(model)
    rpm_cap = max(1, int(limits["rpm"] * GROQ_SAFETY_MARGIN))
    tpm_cap = max(1, int(limits["tpm"] * GROQ_SAFETY_MARGIN))

    # --- RPM pacing (local-only; Groq doesn't expose this in headers) ---
    _prune(state.request_times, now - GROQ_RPM_WINDOW_SECONDS)
    if len(state.request_times) >= rpm_cap:
        oldest = state.request_times[0]
        wait = (oldest + GROQ_RPM_WINDOW_SECONDS) - now
        if wait > 0:
            logger.info(
                f"[RATE LIMIT] {model}: {len(state.request_times)}/{rpm_cap} "
                f"requests in the last {GROQ_RPM_WINDOW_SECONDS}s — pacing {wait:.1f}s"
            )
            time.sleep(wait)
            now = time.time()

    # --- TPM pacing: prefer Groq's own header figures when fresh ---
    if (
        state.header_remaining_tokens is not None
        and state.header_reset_tokens_at is not None
        and now < state.header_reset_tokens_at
    ):
        if state.header_remaining_tokens < estimated_tokens:
            wait = state.header_reset_tokens_at - now
            if wait > 0:
                logger.info(
                    f"[RATE LIMIT] {model}: Groq reports only "
                    f"{state.header_remaining_tokens} tokens left this window "
                    f"(need ~{estimated_tokens}) — pacing {wait:.1f}s"
                )
                time.sleep(wait)
        return

    # Fallback: no fresh header reading yet for this model — estimate from
    # our own local log against the documented (safety-margined) TPM cap.
    _prune(state.token_log, now - GROQ_TPM_WINDOW_SECONDS, key=lambda item: item[0])
    used = sum(tokens for _, tokens in state.token_log)
    if used + estimated_tokens <= tpm_cap:
        return
    oldest_ts = state.token_log[0][0]
    wait = (oldest_ts + GROQ_TPM_WINDOW_SECONDS) - now
    if wait > 0:
        logger.info(
            f"[RATE LIMIT] {model}: {used}+{estimated_tokens} tokens would "
            f"exceed the ~{tpm_cap} TPM budget — pacing {wait:.1f}s"
        )
        time.sleep(wait)


def _retry_after_seconds(err: RateLimitError) -> float | None:
    try:
        header = err.response.headers.get("retry-after")
        return float(header) if header is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _groq_chat(
    *,
    max_tokens: int,
    messages: list[dict],
    reasoning_effort: str = "minimal",
) -> str:
    """Single choke point for every Groq call in Scout: paces requests
    against the active model's real RPM/TPM budget (preferring Groq's own
    response headers over local estimates once we have them), falls forward
    to the next model in the chain on a daily-quota 429 OR on exhausting
    local retries for a per-minute limit, and otherwise backs off using the
    server's Retry-After header instead of failing the whole run.

    Also guards against reasoning models running out of max_tokens before
    finishing — signaled by finish_reason="length" — by bumping max_tokens
    (up to GROQ_TRUNCATION_RETRY_CEILING) and retrying the same call.
    """
    global _last_finish_reason
    prompt_chars = sum(len(m.get("content", "")) for m in messages)

    attempt = 0
    while True:
        model = _current_model()
        estimated_tokens = (prompt_chars // 4) + max_tokens
        _pace_before_call(model, estimated_tokens)

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        # groq==0.11.0's typed create() signature predates gpt-oss/qwen3.6
        # reasoning support — extra_body merges reasoning_effort/format
        # straight into the raw JSON request instead.
        extra_body = _reasoning_extra_body(model, reasoning_effort)
        if extra_body:
            kwargs["extra_body"] = extra_body

        try:
            try:
                raw = client.chat.completions.with_raw_response.create(**kwargs)
                _record_headers(model, getattr(raw, "headers", None))
                response = raw.parse()
            except AttributeError:
                response = client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            _record_headers(model, getattr(getattr(e, "response", None), "headers", None))
            if _is_daily_limit_error(e) and _switch_to_next_model():
                attempt = 0
                continue  # new model, new budget — retry now, no sleep needed

            attempt += 1
            if attempt >= GROQ_MAX_LOCAL_RETRIES:
                # Exhausted local retries on this model for a per-minute
                # limit. Fall forward to the next model in the chain before
                # giving up, same as we already do for daily-quota errors.
                if _switch_to_next_model():
                    logger.warning(
                        f"[FALLBACK] {model} exhausted {GROQ_MAX_LOCAL_RETRIES} "
                        f"local retries on a per-minute rate limit — "
                        f"falling forward to {_current_model()}"
                    )
                    attempt = 0
                    continue
                raise RuntimeError(
                    f"Groq API: still rate-limited on {model} after "
                    f"{GROQ_MAX_LOCAL_RETRIES} local retries, and no fallback "
                    f"models remain in the chain."
                ) from e

            wait = _retry_after_seconds(e) or (2 ** attempt) * 5
            logger.warning(
                f"[RATE LIMIT] Groq 429 on {model} "
                f"(attempt {attempt}/{GROQ_MAX_LOCAL_RETRIES}) — sleeping {wait:.1f}s"
            )
            time.sleep(wait)
            continue
        except NotFoundError as e:
            logger.error(f"[FALLBACK] {model} unavailable ({e}) — trying next model")
            if _switch_to_next_model():
                attempt = 0
                continue
            raise RuntimeError(
                f"Groq API: {model} unavailable and no fallback models remain"
            ) from e

        usage = getattr(response, "usage", None)
        actual_tokens = getattr(usage, "total_tokens", None) or estimated_tokens
        state = _state_for(model)
        now = time.time()
        state.request_times.append(now)
        state.token_log.append((now, actual_tokens))

        content = response.choices[0].message.content
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        _last_finish_reason = finish_reason

        if finish_reason == "length" and max_tokens < GROQ_TRUNCATION_RETRY_CEILING:
            bumped = min(max_tokens * 2, GROQ_TRUNCATION_RETRY_CEILING)
            preview = (content or "")[:80]
            logger.warning(
                f"[TRUNCATED] {model} hit its {max_tokens}-token limit "
                f"before finishing (finish_reason=length, content={'empty' if not content else f'partial: {preview!r}'}) "
                f"— retrying with max_tokens={bumped}"
            )
            max_tokens = bumped
            continue

        return content or ""

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
    if name_lower in TEAM_HOME_CITY:
        return TEAM_HOME_CITY[name_lower]
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
    # Defensive backstop: reasoning_format="hidden" should already keep
    # <think> blocks out of `content` entirely, but if a model ever leaks
    # one anyway, strip it rather than let it defeat every parse strategy
    # below. A dangling unclosed <think> (model ran out of budget mid-
    # thought) is stripped too by treating everything from the opening tag
    # onward as reasoning noise.
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
        logger.error(
            f"[SCOUT] Failed to parse LLM response as JSON object "
            f"(finish_reason={_last_finish_reason}): {text[:200]}"
        )
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
