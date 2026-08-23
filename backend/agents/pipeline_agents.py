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

# ---------------------------------------------------------------------------
# Reasoning control.
#
# Every model in the fallback chain is a *reasoning* model, and left on
# default settings all three will write out a full chain-of-thought before
# (or instead of) answering. That reasoning counts as ordinary output
# tokens, so it: burns through the TPM budget fast, regularly blows past
# max_tokens (triggering the truncation-retry path over and over), and on
# qwen specifically has been observed to consume the *entire* response and
# hit finish_reason="stop" having never emitted the requested JSON at all —
# which _extract_json then correctly fails to parse.
#
# All three models in this chain support reasoning_effort (confirmed at
# console.groq.com/docs/reasoning and the qwen3.6-27b model card, checked
# 2026-08-22), but the valid values — and what they actually MEAN — differ
# by family in a way that matters a lot for a token-budgeted pipeline:
#   - gpt-oss (20b/120b): "low" | "medium" | "high" — three genuinely
#     bounded steps on the same scale. Reasoning is always on to some
#     degree and can't be fully disabled, so "low" is the floor, but "medium"
#     is a modest, predictable bump from "low", not a different mode.
#   - qwen3.6-27b: "none" | "default" — NOT bounded steps on a scale. Per
#     Groq's own model card, "default" is literally "thinking mode... for
#     complex reasoning, math, and coding" — genuinely open-ended
#     chain-of-thought with no built-in ceiling, as opposed to "none"
#     ("non-thinking mode... for efficient, general-purpose dialogue").
#     This is a real, observed failure mode on this model family: reports
#     of default/unbounded thinking mode burning 20,000+ reasoning tokens
#     on trivial requests are common. In this pipeline it showed up as
#     Portfolio hitting finish_reason="length" with completely EMPTY
#     content at both max_tokens=4000 and, after the truncation retry,
#     6000 — the model was spending the entire budget on invisible
#     reasoning before writing a single character of JSON, so raising
#     max_tokens further doesn't help; there's no ceiling to reach.
#
# Every task in this file (Analyst/Risk/Portfolio/Auditor/Decision/Publisher)
# is a bounded structured-output task, not open-ended problem solving. For
# gpt-oss, Portfolio still opts into the bounded "medium" step for a bit more
# room to weigh combinations (see reasoning_effort="default" on that call).
# For qwen specifically, "default" is NOT used anywhere in this file, even
# for Portfolio — its two-mode "none"/"default" split doesn't have a safe
# middle ground the way gpt-oss's three-step scale does, so every call maps
# to "none" on qwen regardless of which desired level the caller asked for.
#
# reasoning_format="hidden" is set unconditionally alongside reasoning_effort
# (per the qwen model card's own recommendation: "set reasoning_format to
# hidden to return only the final answer") as a second line of defense — if
# a model still reasons a little under "low"/"none", the reasoning tokens
# don't leak into `content` and break JSON parsing.
_REASONING_EFFORT_BY_FAMILY: dict[str, dict[str, str]] = {
    "gpt-oss": {"minimal": "low", "default": "medium"},
    "qwen": {"minimal": "none", "default": "none"},
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

# Hard ceiling on any SINGLE sleep this module will ever perform for a 429.
#
# _retry_after_seconds() trusts Groq's Retry-After header verbatim, with no
# upper bound. That's fine for an ordinary per-minute (TPM/RPM) wait — those
# are always <= ~60s since that's the window length — but a real production
# run hit sleeping 1428.0s (~24 minutes) in a single call, on the LAST model
# in the fallback chain (qwen/qwen3.6-27b), when it hit a genuinely
# daily-quota-flavored 429 with nowhere left to fall forward to.
# _is_daily_limit_error(e) and _switch_to_next_model() correctly identifies
# "this is a daily quota, not a per-minute one" for models earlier in the
# chain — but when the LAST model hits it, _switch_to_next_model() returns
# False (nothing left), so that whole branch's condition short-circuits to
# False and execution falls through to the plain per-minute retry path
# below, which blindly obeys whatever Retry-After said. A 24-minute blocking
# sleep inside a single Python call is long enough to blow through most CI
# job timeouts outright — and that's exactly what happened: the run was
# killed mid-sleep ("Error: The operation was canceled"), which is *worse*
# than simply failing this one call, because it also discards every match
# already successfully processed earlier in the same loop.
#
# Any wait longer than this is treated as "not going to resolve within a
# time-boxed pipeline run" regardless of which header reported it or why:
# fall forward to a fresh model immediately if one remains (a working model
# right now beats waiting out an unknown-length quota), and if none remain,
# give up on this one call via RuntimeError rather than blocking — the
# caller (run_analyst / run_risk_filter) is responsible for catching that
# and skipping just this item instead of losing the whole batch.
GROQ_MAX_SINGLE_WAIT_SECONDS = float(os.environ.get("GROQ_MAX_SINGLE_WAIT_SECONDS", "90"))

JSON_RULES = """

IMPORTANT OUTPUT RULES:
- Respond with ONLY the JSON object. No markdown code fences, no commentary, no explanation before or after.
- The response must be valid JSON that can be parsed directly with json.loads()."""

# ---------------------------------------------------------------------------
# Confidence threshold + edge margin — single source of truth.
#
# CONFIDENCE_THRESHOLD was a flat 0.70 baked into 6 separate places (3 prompt
# strings + 3 code checks) — lowered here to a value the real, non-hallucinated
# Analyst confidences can actually clear (observed distribution after fixing
# the form/H2H/standings data gap: 0.50-0.66 across an 11-match batch, none
# reaching 0.70). Tune via env without touching code or prompts.
#
# But a flat confidence floor is the wrong tool on its own regardless of
# where it's set: the standard "value betting" rule (see e.g. Wheatcroft
# 2020's level-stakes strategy, and the Kelly criterion literature more
# generally) is to only bet when your estimated probability p̂ exceeds the
# market's ODDS-IMPLIED probability (1/decimal_odds) — not any fixed
# absolute number. A confidence of 0.60 is bad value on a double-chance pick
# priced at 1.15 (implied ~87%) and good value on a BTTS pick priced at 2.20
# (implied ~45%); one flat threshold can't distinguish those. Bookmakers also
# bake in a margin (the "overround") on top of fair odds, so beating the
# implied probability by ZERO isn't enough — MIN_EDGE_MARGIN requires
# beating it by a buffer before a selection counts as real value.
#
# CONFIDENCE_THRESHOLD still gates the overall match (Risk/Decision) as a
# floor on "is this match understood well enough at all", while
# MIN_EDGE_MARGIN gates the specific MARKET Portfolio picks against real
# odds (see run_portfolio) — they check different things and both matter.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))
MIN_EDGE_MARGIN = float(os.environ.get("MIN_EDGE_MARGIN", "0.03"))

# ---------------------------------------------------------------------------
# Combined-odds band — was a flat 8.0-13.0 hardcoded in 6 separate places
# (2 prompt strings + 4 code checks/messages). Widened per request: 8-13 was
# rejecting entire days' worth of tickets (e.g. 2026-08-23: only 3 matches
# cleared risk filtering, and no 2-5 combination of them could land in that
# narrow a band even though each individual selection still had to pass its
# own MIN_EDGE_MARGIN value check). The value/edge requirement per selection
# is unchanged and unaffected by this — this band only controls how many
# selections get combined together, not whether any given selection is
# worth taking.
COMBINED_ODDS_MIN = float(os.environ.get("COMBINED_ODDS_MIN", "0.0"))
COMBINED_ODDS_MAX = float(os.environ.get("COMBINED_ODDS_MAX", "20.0"))
COMBINED_ODDS_TARGET = (COMBINED_ODDS_MIN + COMBINED_ODDS_MAX) / 2

ANALYST_PROMPT = """You are the ANALYST AGENT for Football Pulse AI.
You receive clean match data and must estimate probabilities for each market.

The match data includes real fields fetched from a stats API:
- recent_form.home / recent_form.away: {"results": "WLDWW" (most recent
  first), "goals_for", "goals_against"} from each team's real last 5
  finished matches, or {} if genuinely unavailable
- head_to_head: {"matches_played", "home_wins", "draws", "away_wins"} from
  the real last-10-meetings record, or {} if unavailable
- standings.home / standings.away: {"position", "points", "goal_diff"} from
  the real current league table, or {} if unavailable

These are the ONLY source of "recent form", "head-to-head record", and
"league position" — you do not have any other source for them, and you must
never invent a plausible-looking form string, W-D-L record, or standings gap
that isn't present in the data. If a field is {} (empty), that specific
input is UNKNOWN — treat it as missing information, not as neutral/average
form, and do not silently substitute a guess.

Before scoring, you MUST fill in "supporting_stats" (see schema below)
restating the ACTUAL values you were given for each factor — copy them, do
not paraphrase or round them. This is required even when a factor is
UNKNOWN; write "UNKNOWN" for that key rather than omitting it. A
"supporting_stats" entry that doesn't match what was in the raw data is
treated as a hallucination.

Then compute model_confidence as an explicit calculation, not a single
freeform number:
1. Start from a base of 0.50.
2. Adjust up/down based ONLY on the supporting_stats you just cited —
   e.g. clear form advantage, H2H dominance, standings gap, confirmed
   injuries to key players. State each adjustment and its direction/size
   in "confidence_calculation" (a short list of strings like
   "+0.08 home team 4W-1D in last 5 vs away 1W-3L-1D").
3. If two or more of recent_form/head_to_head/standings came back UNKNOWN
   for this match, cap your final model_confidence at 0.60 — insufficient
   real data to justify higher confidence than that, regardless of odds or
   general impression of the teams.
4. Sum the adjustments onto the base for the final model_confidence.

Do NOT default to a "safe-sounding" round number (0.65, 0.70, 0.75) out of
habit. Two different matches with genuinely different supporting_stats
should produce genuinely different confidence values — if you notice you're
about to output the same confidence you gave a previous match in this batch
for materially different underlying stats, that's a sign you're anchoring
rather than computing; go back and adjust based on the actual numbers.

"confidence_calculation" must NEVER be an empty list, even when your final
answer stays at the 0.50 base — in that case write the reason explicitly,
e.g. "no adjustment — all of recent_form/head_to_head/standings were
UNKNOWN" or "no adjustment — home and away stats were evenly balanced". A
final confidence that differs from 0.50 with no entries explaining why is
not acceptable output.

Your analysis must be grounded in:
- Recent form (last 5 results, weighted recency) — from recent_form, never invented
- Head-to-head record (last 10 matches) — from head_to_head, never invented
- xG data (if available)
- Home/away performance splits
- Injuries to key players (striker out = lower over 2.5 probability)
- League position and points gap — from standings, never invented

Probabilities must sum to 1.0 for mutually exclusive markets (1X2).
All values between 0.0 and 1.0.""" + JSON_RULES

RISK_PROMPT = f"""You are the RISK AGENT for Football Pulse AI.
Your job is to REJECT dangerous selections.

IMPORTANT CONTEXT: This assessment happens roughly 24-31 hours BEFORE kickoff
(the pipeline runs once daily, the morning before matchday). Official lineups
are almost never confirmed this far in advance — they typically come out
about 1 hour before kickoff. Therefore "lineup not yet confirmed" at this
stage is NORMAL and EXPECTED, not a sign of danger. Do not reject a match
for lacking lineup confirmation alone.

HARD REJECT RULES (any one = instant reject):
1. Adjusted confidence < {CONFIDENCE_THRESHOLD:.2f}
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

Return JSON: {{"approved": bool, "risk_level": "Low|Medium|High", "flags": [...], "rejection_reason": str|null}}""" + JSON_RULES

PORTFOLIO_PROMPT = f"""You are the PORTFOLIO AGENT for Football Pulse AI.
Construct the optimal daily prediction ticket by SELECTING matches and markets only.

You do NOT calculate odds yourself — odds will be looked up from the match data
by the system after you choose. Focus purely on WHICH match + market combinations
to select.

TARGET: combined odds of approximately {COMBINED_ODDS_TARGET:.1f} (acceptable range: {COMBINED_ODDS_MIN:.1f}-{COMBINED_ODDS_MAX:.1f}).

IMPORTANT — Double Chance markets typically have odds between 1.05 and 1.35.
Multiplying 3-4 Double Chance picks together usually only reaches 1.3-2.5,
which may not be enough on its own. To reach the target you will typically
need a MIX:
- 1-2 safer picks (Double Chance, Draw No Bet, odds ~1.1-1.4), AND
- 2-3 higher-odds picks (BTTS, Over 2.5, or even an outright win for a
  team that is favoured but not overwhelmingly so, odds ~1.5-3.0)

IMPORTANT — beyond hitting the odds target, the system will independently
verify each selection has real "edge": your estimated probability for that
exact market must exceed the market's odds-implied probability (1/odds) by
at least a margin, or the system will drop it regardless of your rationale.
So don't pick a market just because it's convenient for the combined-odds
math — pick markets where your own markets{{}} probability for that outcome
is meaningfully higher than what the odds imply. A selection you can't
justify against the actual odds will be silently dropped downstream.

Estimate the odds magnitude roughly yourself when selecting (you can see
home_win/draw/away_win/btts_yes/over25 in each match's odds_snapshot) so
your final selection set multiplies to roughly 8-13. The system will
compute the EXACT final odds — your job is just to pick a sensible
combination that's likely to land in range.

RULES:
- Selections: 2-5 only
- Prefer matches where model_confidence is highest
- If you cannot construct any combination that stays within the acceptable range while each pick maintains real edge over the market, output decision NO_BET

MARKET PREFERENCE (use these exact market keys):
- double_chance_home / double_chance_away
- draw_no_bet_home / draw_no_bet_away
- btts_yes
- over25 (Over 2.5 Goals — only use this key, not over15/under45 which cannot
  be priced from available odds data)
- home_win / away_win (outright — only when confidence >= {CONFIDENCE_THRESHOLD:.2f}, and prefer
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

Output JSON: {{"selections": [{{"fixture_id": int, "market": str, "rationale": str}}], "portfolio_confidence": float, "rationale": str}}
If you cannot build such a combination, output: {{"decision": "NO_BET", "reason": str}}""" + JSON_RULES

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

DECISION_PROMPT = f"""You are the DECISION AGENT for Football Pulse AI.
You receive the final audited ticket and make the publish/no-bet call.

Note: combined_odds has already been computed deterministically from real
bookmaker odds — you do not need to recalculate it, only check it falls in range.

PUBLISH IF:
- Overall confidence >= {CONFIDENCE_THRESHOLD:.2f} ({CONFIDENCE_THRESHOLD * 100:.0f}%)
- Combined odds within {COMBINED_ODDS_MIN:.1f}-{COMBINED_ODDS_MAX:.1f}
- At least 2 selections passed auditor review
- No HARD REJECT flags active
- Auditor verdict is APPROVE or REVISE (with acceptable adjustments)

NO BET IF:
- Any condition above fails
- Gut-check: does this ticket look like disciplined value or desperate volume?

Output JSON: {{"decision": "PUBLISH|NO_BET", "reason": str, "final_confidence": float}}""" + JSON_RULES

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
    # Defensive backstop: reasoning_format="hidden" should already keep
    # <think> blocks out of `content` entirely, but if a model ever leaks
    # one anyway (or reasoning_format is dropped in a future edit), strip it
    # rather than let it defeat every parse strategy below. A dangling
    # unclosed <think> (model ran out of budget mid-thought) is stripped too
    # by treating everything from the opening tag onward as reasoning noise.
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
    # Guard: if the log is empty (e.g. right after a model switch, or the
    # very first call this run), there's nothing to "wait out" — a large
    # batched call can legitimately need more tokens than the whole TPM cap
    # on its own. Blindly indexing state.token_log[0] here would raise
    # IndexError; instead just proceed and let the normal 429/retry/fallback
    # path in _groq_chat handle it if the estimate turns out to be too
    # optimistic.
    if not state.token_log:
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
    truncation_ceiling: int = GROQ_TRUNCATION_RETRY_CEILING,
) -> str:
    """Single choke point for every Groq call: paces requests against the
    active model's real RPM/TPM budget (per console.groq.com/docs/rate-limits,
    preferring Groq's own response headers over local estimates once we have
    them), falls forward to the next model in the chain on a daily-quota 429
    OR on exhausting local retries for a per-minute limit, and otherwise
    backs off using the server's Retry-After header instead of failing the
    whole pipeline run.

    `reasoning_effort` is "minimal" by default ("low" for gpt-oss, "none" for
    qwen — see _REASONING_EFFORT_BY_FAMILY) because every task in this file
    is bounded structured-output generation, not open-ended problem solving.
    Pass "default" for a call that genuinely benefits from more reasoning
    room (currently just Portfolio's combination search) — note this only
    changes gpt-oss's behavior (to its bounded "medium" step); qwen maps
    "default" to "none" too, since qwen's "default" is unbounded thinking
    mode, not a bounded step (see _REASONING_EFFORT_BY_FAMILY for why).

    `truncation_ceiling` overrides GROQ_TRUNCATION_RETRY_CEILING for calls
    whose expected output is larger than a single match's worth (e.g. a
    batched Analyst/Risk call covering several matches in one request) — the
    default is still right for the common single-item case.

    Also guards against reasoning models running out of max_tokens before
    finishing their answer — signaled by finish_reason="length" — which can
    surface as either empty content (whole budget spent on hidden
    chain-of-thought) or non-empty-but-truncated content (e.g. a JSON object
    cut off mid-string). Both are real 200 OK responses that look like parse
    errors downstream, so we key off finish_reason itself rather than
    content emptiness, and bump max_tokens (up to truncation_ceiling)
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
        # groq==0.11.0's typed create() signature predates gpt-oss/qwen3.6
        # reasoning support and has no `reasoning_effort` or `reasoning_format`
        # parameters — passing them as direct kwargs raises TypeError.
        # extra_body merges them straight into the raw JSON request instead,
        # which this SDK version does support.
        extra_body = _reasoning_extra_body(model, reasoning_effort)
        if extra_body:
            kwargs["extra_body"] = extra_body

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

            wait = _retry_after_seconds(e) or (2 ** attempt) * 5

            # Never block this long inside a time-boxed pipeline run — see
            # GROQ_MAX_SINGLE_WAIT_SECONDS above for the production incident
            # this guards against (a 1428s/~24min sleep that got the whole
            # CI job killed). This check runs BEFORE the attempt-count
            # threshold below on purpose: the dangerous case is a huge wait
            # on the very FIRST attempt (attempt=1, as happened in
            # production), which the old attempt-count gate never caught
            # because it only fell forward after GROQ_MAX_LOCAL_RETRIES had
            # already been reached.
            if wait > GROQ_MAX_SINGLE_WAIT_SECONDS:
                logger.warning(
                    f"[RATE LIMIT] {model} reports a {wait:.0f}s wait — too long "
                    f"to block on inside this pipeline"
                )
                if _switch_to_next_model():
                    attempt = 0
                    continue
                # No fallback left and the wait is long enough that
                # sleeping any capped fraction of it would almost certainly
                # just hit another 429 immediately after — fail this one
                # call now instead. The caller is expected to catch this
                # and skip just this item rather than losing everything
                # already processed in the same run.
                raise RuntimeError(
                    f"Groq API: {model} reports a {wait:.0f}s rate-limit wait "
                    f"with no fallback model remaining in the chain — giving "
                    f"up on this call rather than blocking the whole run."
                ) from e

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
        if finish_reason == "length" and max_tokens < truncation_ceiling:
            bumped = min(max_tokens * 2, truncation_ceiling)
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


# ---------------------------------------------------------------------------
# Analyst / Risk batching.
#
# Root cause of the 2026-08-23 NO_BET: it wasn't a code bug at all — every
# fix up to this point worked exactly as designed (no crash, no long block,
# no lost work). The account genuinely ran out of Groq quota. The tell is
# the WAIT VALUES once things went bad: 967s, 898s, 1565s, 1233s — all far
# longer than any TPM window (max 60s), which is the signature of an
# RPD/TPD (per-DAY) limit, not a per-minute one. By the time Analyst even
# started, both gpt-oss models were already dead on arrival (429 within
# milliseconds, no meaningful wait attempted), because Scout's own 12 calls
# earlier in the SAME run had already spent a chunk of both models' daily
# budgets — leaving qwen to carry Analyst's 12 calls + Risk's calls alone,
# and it ran out too.
#
# The actual fix has to be fewer real Groq requests per run, not more retry
# logic — retries and fallback only redistribute load across models with
# the same combined daily ceiling, they don't create more quota. Batching N
# matches into a single Analyst (or Risk) call cuts the request count by
# ~N, which is the only lever inside this file that reduces real API load
# rather than just handling failures better. (Scout's 12 unbatched calls
# are the same category of cost and a good next target, but that's a
# separate file — see scout_agent.py.)
ANALYST_BATCH_SIZE = int(os.environ.get("ANALYST_BATCH_SIZE", "4"))
RISK_BATCH_SIZE = int(os.environ.get("RISK_BATCH_SIZE", "4"))

# Empirically, a single match's Analyst output (supporting_stats +
# confidence_calculation, which can run long when the model has to explain
# multiple UNKNOWN inputs) has been observed at roughly 400-900 tokens.
# Budget generously per match and let the batch's total scale with it,
# rather than reusing a single-match constant that would starve larger
# batches. The existing truncation-retry path (now driven by
# truncation_ceiling) is still the backstop if a particular batch runs
# unusually long — this just sets a first-attempt size that should cover
# the common case without needing that retry at all.
ANALYST_TOKENS_PER_MATCH = 900
ANALYST_TRUNCATION_CEILING = int(os.environ.get("ANALYST_TRUNCATION_CEILING", "12000"))

RISK_TOKENS_PER_MATCH = 350
RISK_TRUNCATION_CEILING = int(os.environ.get("RISK_TRUNCATION_CEILING", "6000"))


def _chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), max(1, size))]


def run_analyst(clean_matches: list[dict]) -> list[dict]:
    probabilities = []

    for batch in _chunk(clean_matches, ANALYST_BATCH_SIZE):
        match_lookup = {m.get("fixture_id"): m for m in batch}
        max_tokens = min(ANALYST_TOKENS_PER_MATCH * len(batch) + 500, ANALYST_TRUNCATION_CEILING)

        # Same isolation principle as before, just at batch granularity now:
        # a batch Groq genuinely can't serve right now (rate-limited across
        # the whole fallback chain) costs us that batch's matches, not every
        # match already collected in `probabilities` from earlier batches in
        # this same loop.
        try:
            text = _groq_chat(
                max_tokens=max_tokens,
                truncation_ceiling=ANALYST_TRUNCATION_CEILING,
                messages=[
                    {"role": "system", "content": ANALYST_PROMPT},
                    {
                        "role": "user",
                        "content": f"""Estimate probabilities for EACH of the following {len(batch)} matches.
Apply the full ANALYST AGENT instructions above to EVERY match independently
— each one needs its own supporting_stats and its own itemized
confidence_calculation, computed only from that match's own data. Do not let
one match's numbers influence another's.

MATCHES:
{json.dumps(batch, indent=2)}

Return a single JSON object of the form {{"analyses": [ ... ]}} containing
exactly {len(batch)} elements, ONE PER MATCH ABOVE, in the same order they
were given. Each element must have this shape:
{{
  "fixture_id": int,
  "home_team": str,
  "away_team": str,
  "supporting_stats": {{
    "home_form": str,
    "away_form": str,
    "head_to_head": str,
    "standings_gap": str,
    "key_injuries": str
  }},
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
  "confidence_calculation": [str],
  "model_confidence": float
}}"""
                    }
                ]
            )
        except RuntimeError as e:
            names = [f"{m.get('home_team')} vs {m.get('away_team')}" for m in batch]
            logger.error(
                f"[ANALYST] Giving up on batch of {len(batch)} matches ({names}) "
                f"— Groq unavailable: {e}"
            )
            continue

        data = _extract_json(text)
        analyses = data.get("analyses") if isinstance(data, dict) else None
        if not isinstance(analyses, list) or not analyses:
            logger.error(
                f"Analyst batch parse error for {len(batch)} matches "
                f"(finish_reason={_last_finish_reason}): {text[:300]}"
            )
            continue

        # Two-pass mapping. A single forward pass that checks "is exactly
        # one match unclaimed so far?" undercounts: a later item in this
        # SAME response hasn't had the chance to claim its own valid
        # fixture_id yet, so an early garbage item can see more
        # "unclaimed" matches than will actually remain once the whole
        # response is processed. Pass 1 here establishes the true final set
        # of validly-claimed fixture_ids across the whole response before
        # any recovery attempt; pass 2 only recovers a bad/missing
        # fixture_id when there's genuinely exactly one leftover match it
        # could be — and consumes it from the pool so a second bad item in
        # the same batch can't also claim it.
        claimed_ids = {
            item.get("fixture_id") for item in analyses
            if isinstance(item, dict) and item.get("fixture_id") in match_lookup
        }
        unclaimed = [m for m in batch if m.get("fixture_id") not in claimed_ids]

        seen_fixture_ids = set()
        for item in analyses:
            if not isinstance(item, dict):
                continue
            fixture_id = item.get("fixture_id")
            match = match_lookup.get(fixture_id)
            if match is None:
                # Model returned a fixture_id that doesn't match anything we
                # sent, or omitted it — recover positionally only if
                # exactly one genuinely-unclaimed match remains for this
                # whole batch, since guessing wrong would silently
                # attribute numbers to the wrong fixture.
                if len(unclaimed) == 1:
                    match = unclaimed.pop(0)
                    fixture_id = match.get("fixture_id")
                else:
                    logger.error(
                        f"[ANALYST] Batch item has unrecognized/missing "
                        f"fixture_id={fixture_id!r} and {len(unclaimed)} "
                        f"unclaimed matches remain in this batch of "
                        f"{len(batch)} — dropping this item rather than "
                        f"guessing which match it belongs to."
                    )
                    continue
            seen_fixture_ids.add(fixture_id)

            item["fixture_id"] = fixture_id
            item.setdefault("home_team", match.get("home_team"))
            item.setdefault("away_team", match.get("away_team"))
            item["odds_snapshot"] = match.get("odds_snapshot", {})
            item["league"] = match.get("league")
            probabilities.append(item)

            # Prompt compliance check — the model is instructed to never
            # return an empty confidence_calculation, but instructions
            # aren't guarantees (this exact gap showed up in production:
            # Espanyol vs Real Madrid logged model_confidence=0.6 with
            # calc: []). Flag it loudly rather than silently trusting an
            # unexplained confidence value.
            calc = item.get("confidence_calculation")
            if not calc:
                logger.warning(
                    f"[ANALYST] {item.get('home_team')} vs {item.get('away_team')} "
                    f"returned model_confidence={item.get('model_confidence')} with an "
                    f"EMPTY confidence_calculation — prompt compliance gap, this "
                    f"confidence value is unexplained and should not be trusted as-is."
                )

            logger.info(
                f"[ANALYST] {item.get('home_team')} vs {item.get('away_team')} "
                f"— model_confidence={item.get('model_confidence')} "
                f"(calc: {item.get('confidence_calculation')})"
            )

        missing = [m for m in batch if m.get("fixture_id") not in seen_fixture_ids]
        if missing:
            # Built as a plain list, not inline in the f-string: nesting an
            # f-string using the same quote character inside another
            # f-string's expression (PEP 701) only parses on Python 3.12+.
            # The CI runtime here is 3.11.16 (see the earlier traceback's
            # /opt/hostedtoolcache/Python/3.11.16/x64/ path), which would
            # raise SyntaxError on that construct.
            missing_names = [f"{m.get('home_team')} vs {m.get('away_team')}" for m in missing]
            logger.warning(
                f"[ANALYST] Batch returned {len(analyses)} analyses but "
                f"{len(missing)}/{len(batch)} matches in the batch got no "
                f"result: {missing_names}"
            )

    # Cheap automated check for the exact anchoring pattern observed earlier:
    # many genuinely different matches all landing on the same confidence.
    # This doesn't fix a bad batch, but it surfaces the signal in logs
    # immediately instead of requiring someone to eyeball the run afterward.
    if len(probabilities) >= 3:
        from collections import Counter
        counts = Counter(
            round(p["model_confidence"], 2)
            for p in probabilities
            if isinstance(p.get("model_confidence"), (int, float))
        )
        if counts:
            common_value, common_count = counts.most_common(1)[0]
            if common_count / len(probabilities) >= 0.5:
                logger.warning(
                    f"[ANALYST] {common_count}/{len(probabilities)} matches share the same "
                    f"model_confidence ({common_value}) this run — possible anchoring rather "
                    f"than genuine differentiation. Check each match's supporting_stats/"
                    f"confidence_calculation to confirm they're actually distinct."
                )

    return probabilities


def run_risk_filter(probabilities: list[dict], intelligence: list[dict]) -> list[dict]:
    safe = []
    intel_by_fixture = {m.get("fixture_id"): m for m in intelligence}

    for batch in _chunk(probabilities, RISK_BATCH_SIZE):
        prob_lookup = {p.get("fixture_id"): p for p in batch}
        max_tokens = min(RISK_TOKENS_PER_MATCH * len(batch) + 300, RISK_TRUNCATION_CEILING)

        payload = [
            {
                "fixture_id": prob.get("fixture_id"),
                "probabilities": prob,
                "intelligence": intel_by_fixture.get(prob.get("fixture_id"), {}),
            }
            for prob in batch
        ]

        # Same isolation principle as run_analyst: a batch Groq genuinely
        # can't serve right now costs us that batch's matches, not every
        # match already approved into `safe` from earlier batches.
        try:
            text = _groq_chat(
                max_tokens=max_tokens,
                truncation_ceiling=RISK_TRUNCATION_CEILING,
                messages=[
                    {"role": "system", "content": RISK_PROMPT},
                    {
                        "role": "user",
                        "content": f"""Evaluate risk for EACH of the following {len(batch)} matches
independently, applying the full RISK AGENT instructions above to every one
— one match's flags or rejection must never influence another's assessment.

MATCHES:
{json.dumps(payload, indent=2)}

Return a single JSON object of the form {{"assessments": [ ... ]}} containing
exactly {len(batch)} elements, ONE PER MATCH ABOVE, in the same order they
were given. Each element must have this shape:
{{
  "fixture_id": int,
  "approved": bool,
  "risk_level": "Low|Medium|High",
  "flags": [str],
  "rejection_reason": str|null
}}"""
                    }
                ]
            )
        except RuntimeError as e:
            names = [f"{p.get('home_team')} vs {p.get('away_team')}" for p in batch]
            logger.error(
                f"[RISK] Giving up on batch of {len(batch)} matches ({names}) "
                f"— Groq unavailable: {e}"
            )
            continue

        data = _extract_json(text)
        assessments = data.get("assessments") if isinstance(data, dict) else None
        if not isinstance(assessments, list) or not assessments:
            logger.error(
                f"Risk batch parse error for {len(batch)} matches "
                f"(finish_reason={_last_finish_reason}): {text[:300]}"
            )
            continue

        # Same two-pass mapping as run_analyst above — see the comment
        # there for why a single forward pass undercounts "unclaimed".
        claimed_ids = {
            item.get("fixture_id") for item in assessments
            if isinstance(item, dict) and item.get("fixture_id") in prob_lookup
        }
        unclaimed = [p for p in batch if p.get("fixture_id") not in claimed_ids]

        seen_fixture_ids = set()
        for item in assessments:
            if not isinstance(item, dict):
                continue
            fixture_id = item.get("fixture_id")
            prob = prob_lookup.get(fixture_id)
            if prob is None:
                if len(unclaimed) == 1:
                    prob = unclaimed.pop(0)
                    fixture_id = prob.get("fixture_id")
                else:
                    logger.error(
                        f"[RISK] Batch item has unrecognized/missing "
                        f"fixture_id={fixture_id!r} and {len(unclaimed)} "
                        f"unclaimed matches remain in this batch of "
                        f"{len(batch)} — dropping this item rather than "
                        f"guessing which match it belongs to."
                    )
                    continue
            seen_fixture_ids.add(fixture_id)

            if item.get("approved"):
                prob["risk_assessment"] = item
                safe.append(prob)
                logger.info(
                    f"[RISK] Approved: {prob.get('home_team')} vs {prob.get('away_team')} "
                    f"— risk={item.get('risk_level')}"
                )
            else:
                logger.info(
                    f"[RISK] Rejected: {prob.get('home_team')} vs {prob.get('away_team')} "
                    f"— model_confidence={prob.get('model_confidence')} "
                    f"— {item.get('rejection_reason')}"
                )

        missing = [p for p in batch if p.get("fixture_id") not in seen_fixture_ids]
        if missing:
            # Same Python-3.11-compatibility fix as run_analyst above.
            missing_names = [f"{p.get('home_team')} vs {p.get('away_team')}" for p in missing]
            logger.warning(
                f"[RISK] Batch returned {len(assessments)} assessments but "
                f"{len(missing)}/{len(batch)} matches in the batch got no "
                f"result: {missing_names}"
            )

    return safe


def run_portfolio(safe_matches: list[dict]) -> dict:
    if len(safe_matches) < 2:
        return {"decision": "NO_BET", "reason": "Insufficient safe candidates after risk filtering."}

    match_lookup = {m.get("fixture_id"): m for m in safe_matches}

    # This is the most reasoning-heavy call in the pipeline — it has to weigh
    # combinations across every safe candidate to hit the target combined-odds
    # target, so unlike every other call in this file it opts into "default"
    # reasoning effort rather than "minimal" (maps to gpt-oss "medium" — a
    # bounded step. On qwen it now maps to "none", same as "minimal", because
    # qwen's "default" is unbounded thinking mode, not a bounded step — see
    # the _REASONING_EFFORT_BY_FAMILY comment for the full explanation and
    # the production failure this caused: finish_reason="length" with
    # completely EMPTY content at both max_tokens=4000 and 6000, because the
    # model was spending the whole budget on invisible reasoning with no
    # ceiling to hit).
    #
    # NOTE ON max_tokens=4000 (was 3000): the last run hit finish_reason=
    # "length" with EMPTY content at 3000 (whole budget spent before any
    # visible output), forcing a retry at 6000. 4000 gives gpt-oss's
    # "medium" reasoning effort more breathing room to actually reach
    # visible output on the first attempt for the common case (2-5
    # selections), while the existing truncation retry still climbs to the
    # 6000 ceiling if a particular candidate set needs it. (This budget is
    # irrelevant for qwen now that it runs in "none"/non-thinking mode.)
    try:
        text = _groq_chat(
            max_tokens=4000,
            reasoning_effort="default",
            messages=[
                {"role": "system", "content": PORTFOLIO_PROMPT},
                {
                    "role": "user",
                    "content": f"Select matches and markets from these safe candidates:\n{json.dumps(safe_matches, indent=2)}"
                }
            ]
        )
    except RuntimeError as e:
        # Groq genuinely unavailable (rate-limited across the whole fallback
        # chain, with a wait too long to block on — see
        # GROQ_MAX_SINGLE_WAIT_SECONDS). Degrade to NO_BET, same as every
        # other failure mode here, instead of letting this crash the whole
        # daily_run and lose the Scout/Analyst/Risk work that already
        # succeeded upstream in this same run.
        logger.error(f"Portfolio agent unavailable: {e}")
        return {"decision": "NO_BET", "reason": f"Portfolio agent unavailable: {e}"}
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

        # Real edge check (see the CONFIDENCE_THRESHOLD/MIN_EDGE_MARGIN
        # comment above ANALYST_PROMPT for the reasoning): a flat confidence
        # floor doesn't tell you whether a bet is actually good value against
        # ITS specific odds. The Analyst already estimated a probability for
        # this exact market in "markets" — compare that to what the real
        # odds imply (1/odds) and require it to beat that by MIN_EDGE_MARGIN,
        # the standard "value betting" rule. A selection that fails this is
        # dropped here regardless of what Portfolio's rationale said, because
        # Portfolio only sees odds_snapshot's raw 5 fields, not every derived
        # market's true implied probability.
        model_prob = (match.get("markets") or {}).get(market)
        if model_prob is None:
            skipped.append(
                f"{match.get('home_team')} vs {match.get('away_team')} "
                f"({market}): no Analyst-estimated probability available for this market"
            )
            continue
        implied_prob = 1.0 / odds
        edge = model_prob - implied_prob
        if edge < MIN_EDGE_MARGIN:
            skipped.append(
                f"{match.get('home_team')} vs {match.get('away_team')} ({market}): "
                f"no real edge — model {model_prob:.2f} vs market-implied {implied_prob:.2f} "
                f"(edge {edge:+.2f} < required {MIN_EDGE_MARGIN:.2f})"
            )
            continue

        final_selections.append({
            "fixture_id": fixture_id,
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "league": match.get("league"),
            "market": market,
            "odds": odds,
            "model_prob": round(model_prob, 3),
            "implied_prob": round(implied_prob, 3),
            "edge": round(edge, 3),
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

    try:
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
    except RuntimeError as e:
        logger.error(f"Auditor agent unavailable: {e}")
        return {"auditor_verdict": "REJECT", "critical_flags": [f"Auditor agent unavailable: {e}"]}
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

    try:
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
    except RuntimeError as e:
        logger.error(f"Decision agent unavailable: {e}")
        return {"decision": "NO_BET", "reason": f"Decision agent unavailable: {e}", "final_confidence": 0.0}
    data = _extract_json(text)
    if not data:
        logger.error(f"Decision parse error (finish_reason={_last_finish_reason}): {text[:200]}")
        return {"decision": "NO_BET", "reason": "Decision agent error — defaulting safe.", "final_confidence": 0.0}

    final_confidence = data.get("final_confidence")
    combined_odds = portfolio.get("combined_odds")

    if final_confidence is not None and final_confidence < CONFIDENCE_THRESHOLD:
        if data.get("decision") == "PUBLISH":
            logger.warning(
                f"[DECISION] Overriding LLM's PUBLISH -> NO_BET: "
                f"final_confidence={final_confidence} < {CONFIDENCE_THRESHOLD:.2f}"
            )
        data["decision"] = "NO_BET"
        if "below" not in str(data.get("reason", "")).lower():
            data["reason"] = (
                f"Overridden to NO_BET: final_confidence "
                f"({final_confidence}) is below the {CONFIDENCE_THRESHOLD:.2f} publish threshold. "
                f"Original reasoning: {data.get('reason', '')}"
            )

    if combined_odds is not None and not (COMBINED_ODDS_MIN <= combined_odds <= COMBINED_ODDS_MAX):
        if data.get("decision") == "PUBLISH":
            logger.warning(
                f"[DECISION] Overriding LLM's PUBLISH -> NO_BET: "
                f"combined_odds={combined_odds} outside {COMBINED_ODDS_MIN:.1f}-{COMBINED_ODDS_MAX:.1f} range"
            )
        data["decision"] = "NO_BET"
        data["reason"] = (
            f"Overridden to NO_BET: combined_odds ({combined_odds}) is outside "
            f"the required {COMBINED_ODDS_MIN:.1f}-{COMBINED_ODDS_MAX:.1f} range. "
            f"Original reasoning: {data.get('reason', '')}"
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
the {CONFIDENCE_THRESHOLD * 100:.0f}%+ confidence threshold today.

Discipline over volume. We wait.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    try:
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
    except RuntimeError as e:
        # This is the very last step after every upstream agent has already
        # succeeded — a Publisher-only failure here shouldn't mean no email
        # goes out at all. Fall back to a plain-text summary built directly
        # from the data we already have, skipping only the LLM formatting.
        logger.error(f"Publisher agent unavailable: {e}")
        selections_text = "\n".join(
            f"MATCH: {s.get('home_team')} vs {s.get('away_team')} ({s.get('league')})\n"
            f"Market: {s.get('market')}\nOdds: {s.get('odds')}\nReason: {s.get('rationale', '')}"
            for s in portfolio.get("selections", [])
        )
        return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {target_date}  |  🕗 08:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Confidence: {decision.get('final_confidence', 'N/A')}
⚠️  Overall Risk: {portfolio.get('risk_level', 'N/A')}
(Publisher agent unavailable — this is a plain fallback summary, not the
usual formatted ticket: {e})

{selections_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Combined Odds: {portfolio.get('combined_odds', 'N/A')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
