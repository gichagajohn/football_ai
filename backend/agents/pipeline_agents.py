"""
ANALYST / RISK / PORTFOLIO / AUDITOR / DECISION / PUBLISHER AGENTS — Football Pulse AI
"""

import json
import logging
import os
import re
import time
from typing import Any

from groq import Groq, NotFoundError, RateLimitError

logger = logging.getLogger(__name__)
# max_retries=0: we do our own pacing + retry below so the two mechanisms
# don't stack (SDK backoff + our backoff was making failures take even longer).
client = Groq(api_key=os.environ.get("GROQ_API_KEY"), max_retries=0)

# Model fallback chain: if the *active* model hits its daily quota (RPD/TPD)
# OR exhausts its local per-minute retries (RPM/TPM), we switch to the next
# model instead of failing the whole run. Each model on Groq has its own
# independent RPM/TPM/RPD/TPD budget, so this unblocks the run immediately.
# Ordered by preference; only falls forward, never back.
#
# NOTE: Groq deprecates/retires models on short notice (llama-3.1-8b-instant
# and llama-3.3-70b-versatile were both shut down Aug 16, 2026). If a model
# in this chain starts 404ing, it's a good sign this list needs an update —
# check https://console.groq.com/docs/models for the current lineup. The
# code below already treats a 404 as "unusable, skip it" rather than crashing,
# but a stale chain still means fewer real fallback options in practice.
GROQ_MODEL_FALLBACK_CHAIN = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
]
_current_model_index = 0

# Models in this chain that support the reasoning_effort param (gpt-oss family
# only — qwen3.6-27b on Groq does not accept it and will error if we pass it).
_REASONING_EFFORT_CAPABLE_SUBSTR = "gpt-oss"

# Set by _groq_chat right before returning, so callers can enrich their own
# parse-error logs with *why* the content was empty (truncated mid-reasoning
# vs. some other cause) without changing _groq_chat's return type everywhere
# it's called.
_last_finish_reason: str | None = None


def _current_model() -> str:
    return GROQ_MODEL_FALLBACK_CHAIN[_current_model_index]


def _switch_to_next_model() -> bool:
    """Advance to the next fallback model. Returns False if none remain."""
    global _current_model_index
    if _current_model_index >= len(GROQ_MODEL_FALLBACK_CHAIN) - 1:
        return False
    _current_model_index += 1
    logger.warning(
        f"[FALLBACK] Switching to {_current_model()}"
    )
    return True


def _is_daily_limit_error(err: RateLimitError) -> bool:
    """Groq's error body says e.g. '...on tokens per day (TPD): Limit...' —
    that's a quota that won't refill for hours, unlike a per-minute TPM/RPM
    limit which our pacer already avoids. Sleeping it out isn't worth it when
    a fallback model with its own separate daily budget is available."""
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
# IMPORTANT: earlier versions of this file assumed a flat 18,000 TPM budget
# for every model — more than double the real 8,000 TPM cap. That mismatch
# is why the daily run was getting 429'd on almost every call even with
# pacing in place: the pacer thought it had headroom it didn't actually have.
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

# Local retries *per model* before we give up on that model and fall forward
# to the next one in the chain (if any remain). Previously this was a flat
# 4 retries with no forward-fallback on plain rate-limit exhaustion — only
# on errors that explicitly said "per day" — which meant a model stuck in a
# tight per-minute rate-limit loop could exhaust all retries and crash the
# whole pipeline even while other models in the chain sat completely unused.
GROQ_MAX_LOCAL_RETRIES = 4

# Ceiling for the truncation retry in _groq_chat. gpt-oss models can run out
# of max_tokens two different ways: (1) spend the entire budget on hidden
# chain-of-thought and return empty content, or (2) spend it mid-answer and
# return a cut-off, non-empty-but-invalid partial response (e.g. an
# unterminated JSON string). Both show up as finish_reason="length" — content
# emptiness is not a reliable signal on its own, so the retry keys off
# finish_reason alone. This caps how far we'll bump max_tokens chasing it.
GROQ_TRUNCATION_RETRY_CEILING = 6000

JSON_RULES = """

IMPORTANT OUTPUT RULES:
- Respond with ONLY the JSON object. No markdown code fences, no commentary, no explanation before or after.
- The response must be valid JSON that can be parsed directly with json.loads()."""

ANALYST_PROMPT = """You are the ANALYST AGENT for Football Pulse AI.
You receive clean match data and must estimate probabilities for each market.

Your analysis must be grounded in:
- Recent form (last 5 results, weighted recency)
- Head-to-head record (last 10 matches)
- xG data (if available)
- Home/away performance splits
- Injuries to key players (striker out = lower over 2.5 probability)
- League position and points gap

Probabilities must sum to 1.0 for mutually exclusive markets (1X2).
All values between 0.0 and 1.0.""" + JSON_RULES

RISK_PROMPT = """You are the RISK AGENT for Football Pulse AI.
Your job is to REJECT dangerous selections.

IMPORTANT CONTEXT: This assessment happens roughly 24-31 hours BEFORE kickoff
(the pipeline runs once daily, the morning before matchday). Official lineups
are almost never confirmed this far in advance — they typically come out
about 1 hour before kickoff. Therefore "lineup not yet confirmed" at this
stage is NORMAL and EXPECTED, not a sign of danger. Do not reject a match
for lacking lineup confirmation alone.

HARD REJECT RULES (any one = instant reject):
1. Adjusted confidence < 0.70
2. Both goalkeepers injured/suspended (confirmed injury, not lineup absence)
3. Odds moved against our pick by >15% from open
4. Weather: wind > 60 km/h or heavy snowfall forecast
5. Cup/playoff match with strong rotation signals (e.g. team already through,
   dead-rubber group match, manager has publicly stated rotation intent)
6. Team with 3+ key attackers injured/suspended (affects goals markets)

SOFT FLAGS (reduce confidence, may still pass):
- Lineup not yet confirmed (expected at this stage — note it, don't reject for it)
- Odds moved against pick by 8-15%
- One key player injured
- Minor weather concerns
- Long travel (>800km away)
- Schedule congestion (3rd game in 8 days)

Return JSON: {"approved": bool, "risk_level": "Low|Medium|High", "flags": [...], "rejection_reason": str|null}""" + JSON_RULES

PORTFOLIO_PROMPT = """You are the PORTFOLIO AGENT for Football Pulse AI.
Construct the optimal daily prediction ticket by SELECTING matches and markets only.

You do NOT calculate odds yourself — odds will be looked up from the match data
by the system after you choose. Focus purely on WHICH match + market combinations
to select.

TARGET: combined odds of approximately 10.0 (acceptable range: 8.0-13.0).

IMPORTANT — Double Chance markets typically have odds between 1.05 and 1.35.
Multiplying 3-4 Double Chance picks together usually only reaches 1.3-2.5,
NOT 10.0. To reach the ~10.0 target you will typically need a MIX:
- 1-2 safer picks (Double Chance, Draw No Bet, odds ~1.1-1.4), AND
- 2-3 higher-odds picks (BTTS, Over 2.5, or even an outright win for a
  team that is favoured but not overwhelmingly so, odds ~1.5-3.0)

Estimate the odds magnitude roughly yourself when selecting (you can see
home_win/draw/away_win/btts_yes/over25 in each match's odds_snapshot) so
your final selection set multiplies to roughly 8-13. The system will
compute the EXACT final odds — your job is just to pick a sensible
combination that's likely to land in range.

RULES:
- Selections: 2-5 only
- Prefer matches where model_confidence is highest
- If you cannot construct a combination likely to reach 8.0+, output decision NO_BET

MARKET PREFERENCE (use these exact market keys):
- double_chance_home / double_chance_away
- draw_no_bet_home / draw_no_bet_away
- btts_yes
- over25 (Over 2.5 Goals — only use this key, not over15/under45 which cannot
  be priced from available odds data)
- home_win / away_win (outright — only when confidence >= 0.70, and prefer
  the side that is favoured but still offers odds > 1.4)

FORBIDDEN:
- Correct score markets
- First goalscorer
- Asian handicap
- Combining two volatile markets from the same match
- over15 and under45 (cannot be priced — do not select these)

For each selection, output the fixture_id and the exact market key (must match one of:
home_win, draw, away_win, btts_yes, over25, double_chance_home,
double_chance_away, draw_no_bet_home, draw_no_bet_away).

Output JSON: {"selections": [{"fixture_id": int, "market": str, "rationale": str}], "portfolio_confidence": float, "rationale": str}
If you cannot build a combination likely to reach 8.0+, output: {"decision": "NO_BET", "reason": str}""" + JSON_RULES

AUDITOR_PROMPT = """You are the AUDITOR AGENT for Football Pulse AI.
Act as the devil's advocate. Your job is to CHALLENGE every selection.

For each selection, ask:
1. What is the most likely way this loses?
2. What assumptions are we making that could be wrong?
3. What piece of news from the last 48 hours could break this?
4. Is the market already pricing in what we think is an edge?
5. Historical base rate: how often does this market win at these odds?

Be genuinely critical. If you find a serious flaw, flag it.
Adjust the confidence DOWN where warranted.

Output JSON: {"adjusted_selections": [...], "overall_confidence_adjustment": float, "critical_flags": [...], "auditor_verdict": "APPROVE|REVISE|REJECT"}""" + JSON_RULES

DECISION_PROMPT = """You are the DECISION AGENT for Football Pulse AI.
You receive the final audited ticket and make the publish/no-bet call.

Note: combined_odds has already been computed deterministically from real
bookmaker odds — you do not need to recalculate it, only check it falls in range.

PUBLISH IF:
- Overall confidence >= 0.70 (70%)
- Combined odds within 8.0-13.0
- At least 2 selections passed auditor review
- No HARD REJECT flags active
- Auditor verdict is APPROVE or REVISE (with acceptable adjustments)

NO BET IF:
- Any condition above fails
- Gut-check: does this ticket look like disciplined value or desperate volume?

Output JSON: {"decision": "PUBLISH|NO_BET", "reason": str, "final_confidence": float}""" + JSON_RULES

PUBLISHER_PROMPT = """You are the PUBLISHER AGENT for Football Pulse AI.
Format the final ticket for human-readable release as PLAIN TEXT (not JSON).

Each selection in the portfolio has these fields: home_team, away_team, league,
market, odds (real decimal odds, already validated), rationale.

Output in exactly this format — no additions, no removals:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {date}  |  🕗 08:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Confidence: {confidence}%
⚠️  Overall Risk: {risk_level}

For each selection, include a block like:
MATCH: {home_team} vs {away_team} ({league})
Market: {market}
Odds: {odds}
Reason: {short reason based on rationale}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Combined Odds: {combined_odds}
💡 Expected Risk: {risk_level}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  DISCLAIMER
This is a probabilistic model output.
No prediction is guaranteed.
Bet only what you can afford to lose.
Discipline over volume. Always.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use the EXACT odds and combined_odds values provided — do not recalculate or
invent numbers. Respond with ONLY the formatted ticket text. No commentary
before or after."""


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


# ---------------------------------------------------------------------------
# Per-model rate limit state.
#
# Keyed by model name (not just "the active model") so that switching back
# and forth in the chain — or simply having already made calls on a model
# earlier in the run — doesn't lose pacing history for that model.
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
        # Authoritative figures parsed from Groq's own response headers.
        # None until we've seen at least one response for this model.
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
    Returns 0.0 if the value is missing or unparseable (caller should treat
    that as "no usable signal", not "reset is instant").
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
    (success or 429) and stash them so the next _pace_before_call for this
    model can use real numbers instead of our local estimate."""
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
    budget. Prefers Groq's own header-reported TPM figures when we have a
    fresh reading for this model; falls back to a local rolling-window
    estimate otherwise (e.g. on the very first call to a model, before we've
    seen any headers for it yet).

    RPM has no equivalent header from Groq at all, so it's always paced from
    our own local request-timestamp log against the documented per-model cap.
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
    reasoning_effort: str | None = None,
) -> str:
    """Single choke point for every Groq call: paces requests against the
    active model's real RPM/TPM budget (per console.groq.com/docs/rate-limits,
    preferring Groq's own response headers over local estimates once we have
    them), falls forward to the next model in the chain on a daily-quota 429
    OR on exhausting local retries for a per-minute limit, and otherwise
    backs off using the server's Retry-After header instead of failing the
    whole pipeline run.

    Also guards against gpt-oss models running out of max_tokens before
    finishing their answer — signaled by finish_reason="length" — which can
    surface as either empty content (whole budget spent on hidden
    chain-of-thought) or non-empty-but-truncated content (e.g. a JSON object
    cut off mid-string). Both are real 200 OK responses that look like parse
    errors downstream, so we key off finish_reason itself rather than
    content emptiness, and bump max_tokens (up to GROQ_TRUNCATION_RETRY_CEILING)
    and retry the same call before giving up.
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
        if reasoning_effort and _REASONING_EFFORT_CAPABLE_SUBSTR in model:
            # groq==0.11.0's typed create() signature predates gpt-oss support
            # and has no `reasoning_effort` parameter — passing it as a direct
            # kwarg raises TypeError. extra_body merges it straight into the
            # raw JSON request instead, which this SDK version does support.
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

        try:
            # with_raw_response gives us access to Groq's rate-limit headers
            # (x-ratelimit-remaining-tokens / x-ratelimit-reset-tokens) so
            # _pace_before_call can use real numbers instead of only a local
            # estimate. Falls back to the plain call if this SDK version
            # doesn't support it, so a SDK upgrade/downgrade can't crash the
            # pipeline outright — it just loses header-based calibration.
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
                # limit. Previously this raised immediately even when other
                # models in the chain were untouched — now we fall forward
                # to them first, same as we already do for daily-quota
                # errors, and only raise once the whole chain is exhausted.
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
            # Model doesn't exist / no access — e.g. deprecated or renamed.
            # This isn't a transient rate limit, so retrying the same model
            # is pointless; skip straight to the next one in the chain.
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

        # Key off finish_reason alone, not content emptiness — a truncated
        # call can come back with SOME content (a cut-off JSON string) just
        # as easily as none (whole budget spent on hidden reasoning), and
        # both are equally unusable to the caller.
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


def _derive_odds(market: str, odds_snapshot: dict) -> float | None:
    try:
        home = odds_snapshot.get("home_win")
        draw = odds_snapshot.get("draw")
        away = odds_snapshot.get("away_win")

        if market == "home_win":
            return _valid(home)
        if market == "away_win":
            return _valid(away)
        if market == "draw":
            return _valid(draw)
        if market == "btts_yes":
            return _valid(odds_snapshot.get("btts_yes"))
        if market == "over25":
            return _valid(odds_snapshot.get("over25"))
        if market == "double_chance_home" and home and draw:
            implied = (1 / home) + (1 / draw)
            return _valid(round(1 / implied, 3)) if implied > 0 else None
        if market == "double_chance_away" and away and draw:
            implied = (1 / away) + (1 / draw)
            return _valid(round(1 / implied, 3)) if implied > 0 else None
        if market == "draw_no_bet_home" and home and draw:
            p_home = 1 / home
            p_draw = 1 / draw
            p_away = 1 / away if away else max(0.01, 1 - p_home - p_draw)
            adj_p_home = p_home / (p_home + p_away) if (p_home + p_away) > 0 else None
            return _valid(round(1 / adj_p_home, 3)) if adj_p_home else None
        if market == "draw_no_bet_away" and away and draw:
            p_home = 1 / home if home else 0
            p_draw = 1 / draw
            p_away = 1 / away
            adj_p_away = p_away / (p_home + p_away) if (p_home + p_away) > 0 else None
            return _valid(round(1 / adj_p_away, 3)) if adj_p_away else None
        return None
    except (TypeError, ZeroDivisionError):
        return None


def _valid(odds: float | None) -> float | None:
    if odds is None:
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds < 1.01:
        return None
    return odds


def run_analyst(clean_matches: list[dict]) -> list[dict]:
    probabilities = []
    for i, match in enumerate(clean_matches):
        text = _groq_chat(
            max_tokens=1500,
            messages=[
                {"role": "system", "content": ANALYST_PROMPT},
                {
                    "role": "user",
                    "content": f"""Estimate probabilities for this match:
{json.dumps(match, indent=2)}

Return JSON:
{{
  "fixture_id": int,
  "home_team": str,
  "away_team": str,
  "markets": {{
    "home_win": float,
    "draw": float,
    "away_win": float,
    "btts_yes": float,
    "over15": float,
    "over25": float,
    "under45": float,
    "double_chance_home": float,
    "double_chance_away": float,
    "draw_no_bet_home": float,
    "draw_no_bet_away": float
  }},
  "key_factors": [str],
  "model_confidence": float
}}"""
                }
            ]
        )
        data = _extract_json(text)
        if data:
            data.setdefault("fixture_id", match.get("fixture_id"))
            data.setdefault("home_team", match.get("home_team"))
            data.setdefault("away_team", match.get("away_team"))
            data["odds_snapshot"] = match.get("odds_snapshot", {})
            data["league"] = match.get("league")
            probabilities.append(data)
            logger.info(
                f"[ANALYST] {data.get('home_team')} vs {data.get('away_team')} "
                f"— model_confidence={data.get('model_confidence')}"
            )
        else:
            logger.error(
                f"Analyst parse error for {match.get('home_team')} vs {match.get('away_team')} "
                f"(finish_reason={_last_finish_reason}): {text[:200]}"
            )
    return probabilities


def run_risk_filter(probabilities: list[dict], intelligence: list[dict]) -> list[dict]:
    safe = []
    for i, prob in enumerate(probabilities):
        intel = next((m for m in intelligence if m.get("fixture_id") == prob.get("fixture_id")), {})
        text = _groq_chat(
            max_tokens=600,
            messages=[
                {"role": "system", "content": RISK_PROMPT},
                {
                    "role": "user",
                    "content": f"Evaluate risk:\nProbabilities: {json.dumps(prob)}\nIntelligence: {json.dumps(intel)}"
                }
            ]
        )
        risk_data = _extract_json(text)
        if not risk_data:
            logger.error(
                f"Risk parse error for {prob.get('home_team')} vs {prob.get('away_team')} "
                f"(finish_reason={_last_finish_reason}): {text[:200]}"
            )
        elif risk_data.get("approved"):
            prob["risk_assessment"] = risk_data
            safe.append(prob)
            logger.info(
                f"[RISK] Approved: {prob.get('home_team')} vs {prob.get('away_team')} "
                f"— risk={risk_data.get('risk_level')}"
            )
        else:
            logger.info(
                f"[RISK] Rejected: {prob.get('home_team')} vs {prob.get('away_team')} "
                f"— model_confidence={prob.get('model_confidence')} "
                f"— {risk_data.get('rejection_reason')}"
            )
    return safe


def run_portfolio(safe_matches: list[dict]) -> dict:
    if len(safe_matches) < 2:
        return {"decision": "NO_BET", "reason": "Insufficient safe candidates after risk filtering."}

    match_lookup = {m.get("fixture_id"): m for m in safe_matches}

    # This is the most reasoning-heavy call in the pipeline — it has to weigh
    # combinations across every safe candidate to hit an 8.0-13.0 combined-odds
    # target. gpt-oss models can burn an entire small max_tokens budget on
    # hidden chain-of-thought for a task like this and return empty content
    # (finish_reason="length") even on a 200 OK. Giving it more room and a
    # capped reasoning_effort makes that much less likely; _groq_chat's
    # empty-content retry is the backstop if it still happens.
    text = _groq_chat(
        max_tokens=3000,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": PORTFOLIO_PROMPT},
            {
                "role": "user",
                "content": f"Select matches and markets from these safe candidates:\n{json.dumps(safe_matches, indent=2)}"
            }
        ]
    )
    data = _extract_json(text)
    if not data:
        logger.error(
            f"Portfolio parse error (finish_reason={_last_finish_reason}, "
            f"content_len={len(text)}): {text[:200]}"
        )
        reason = (
            "Portfolio construction failed: model output was cut off before "
            "completing valid JSON, even after a retry with more tokens "
            "(finish_reason=length)."
            if _last_finish_reason == "length"
            else "Portfolio construction failed: model returned unparseable content."
        )
        return {"decision": "NO_BET", "reason": reason}

    if data.get("decision") == "NO_BET":
        return data

    raw_selections = data.get("selections", [])
    if len(raw_selections) < 2:
        return {"decision": "NO_BET", "reason": "Portfolio agent selected fewer than 2 matches."}

    final_selections = []
    skipped = []

    for sel in raw_selections:
        fixture_id = sel.get("fixture_id")
        market = sel.get("market")
        match = match_lookup.get(fixture_id)
        if not match:
            skipped.append(f"fixture {fixture_id} not found in safe matches")
            continue
        odds_snapshot = match.get("odds_snapshot", {}) or {}
        odds = _derive_odds(market, odds_snapshot)
        if odds is None:
            skipped.append(
                f"{match.get('home_team')} vs {match.get('away_team')} "
                f"({market}): could not derive valid odds from available data"
            )
            continue
        final_selections.append({
            "fixture_id": fixture_id,
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "league": match.get("league"),
            "market": market,
            "odds": odds,
            "rationale": sel.get("rationale", ""),
        })

    if skipped:
        logger.info(f"[PORTFOLIO] Skipped selections: {skipped}")

    if len(final_selections) < 2:
        return {"decision": "NO_BET", "reason": f"Fewer than 2 selections had derivable odds. Skipped: {skipped}"}

    final_selections = final_selections[:5]
    combined_odds = 1.0
    for s in final_selections:
        combined_odds *= s["odds"]

    return {
        "selections": final_selections,
        "combined_odds": round(combined_odds, 2),
        "portfolio_confidence": data.get("portfolio_confidence"),
        "rationale": data.get("rationale", ""),
        "risk_level": data.get("risk_level", "Medium"),
    }


def run_auditor(portfolio: dict) -> dict:
    if portfolio.get("decision") == "NO_BET":
        return {"auditor_verdict": "REJECT", "critical_flags": ["No portfolio to audit — already NO_BET."]}

    text = _groq_chat(
        max_tokens=1500,
        messages=[
            {"role": "system", "content": AUDITOR_PROMPT},
            {
                "role": "user",
                "content": f"Challenge this ticket:\n{json.dumps(portfolio, indent=2)}"
            }
        ]
    )
    data = _extract_json(text)
    if not data:
        logger.error(f"Auditor parse error (finish_reason={_last_finish_reason}): {text[:200]}")
        return {"auditor_verdict": "REJECT", "critical_flags": ["Auditor system error — could not parse response."]}
    return data


def run_decision(audited: dict, portfolio: dict) -> dict:
    if portfolio.get("decision") == "NO_BET":
        return {"decision": "NO_BET", "reason": portfolio.get("reason", "No valid portfolio constructed."), "final_confidence": 0.0}

    if audited.get("auditor_verdict") == "REJECT":
        return {"decision": "NO_BET", "reason": f"Auditor rejected: {audited.get('critical_flags')}", "final_confidence": 0.0}

    text = _groq_chat(
        max_tokens=400,
        messages=[
            {"role": "system", "content": DECISION_PROMPT},
            {
                "role": "user",
                "content": f"Portfolio: {json.dumps(portfolio)}\nAudit: {json.dumps(audited)}\nMake the final decision."
            }
        ]
    )
    data = _extract_json(text)
    if not data:
        logger.error(f"Decision parse error (finish_reason={_last_finish_reason}): {text[:200]}")
        return {"decision": "NO_BET", "reason": "Decision agent error — defaulting safe.", "final_confidence": 0.0}

    final_confidence = data.get("final_confidence")
    combined_odds = portfolio.get("combined_odds")

    if final_confidence is not None and final_confidence < 0.70:
        if data.get("decision") == "PUBLISH":
            logger.warning(
                f"[DECISION] Overriding LLM's PUBLISH -> NO_BET: "
                f"final_confidence={final_confidence} < 0.70"
            )
        data["decision"] = "NO_BET"
        if "below" not in str(data.get("reason", "")).lower():
            data["reason"] = (
                f"Overridden to NO_BET: final_confidence "
                f"({final_confidence}) is below the 0.70 publish threshold. "
                f"Original reasoning: {data.get('reason', '')}"
            )

    if combined_odds is not None and not (8.0 <= combined_odds <= 13.0):
        if data.get("decision") == "PUBLISH":
            logger.warning(
                f"[DECISION] Overriding LLM's PUBLISH -> NO_BET: "
                f"combined_odds={combined_odds} outside 8.0-13.0 range"
            )
        data["decision"] = "NO_BET"
        data["reason"] = (
            f"Overridden to NO_BET: combined_odds ({combined_odds}) is outside "
            f"the required 8.0-13.0 range. Original reasoning: {data.get('reason', '')}"
        )

    return data


def run_publisher(portfolio: dict, decision: dict, audited: dict, target_date: str) -> str:
    if decision.get("decision") == "NO_BET":
        return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {target_date}  |  🕗 08:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 NO BET TODAY

Reason: {decision.get('reason', 'Insufficient edge detected.')}

The system found no selections meeting
the 70%+ confidence threshold today.

Discipline over volume. We wait.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    return _groq_chat(
        max_tokens=1200,
        messages=[
            {"role": "system", "content": PUBLISHER_PROMPT},
            {
                "role": "user",
                "content": f"""Format the final ticket.
Date: {target_date}
Portfolio: {json.dumps(portfolio)}
Decision: {json.dumps(decision)}
Audited: {json.dumps(audited)}"""
            }
        ]
    )
