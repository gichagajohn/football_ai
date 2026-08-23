"""
ANALYST / RISK / PORTFOLIO / AUDITOR / DECISION / PUBLISHER AGENTS — Football Pulse AI
"""

import json
import logging
import os
import re

from backend.gemini_client import gemini_chat as _gemini_chat, get_last_finish_reason

logger = logging.getLogger(__name__)

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
# Batching reduces the number of Gemini requests in a daily run while keeping
# failure isolation at batch granularity: one unavailable batch does not erase
# matches successfully processed in earlier batches.
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
        # a batch Gemini genuinely can't serve right now (rate-limited across
        # the whole fallback chain) costs us that batch's matches, not every
        # match already collected in `probabilities` from earlier batches in
        # this same loop.
        try:
            text = _gemini_chat(
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
                f"— Gemini unavailable: {e}"
            )
            continue

        data = _extract_json(text)
        analyses = data.get("analyses") if isinstance(data, dict) else None
        if not isinstance(analyses, list) or not analyses:
            logger.error(
                f"Analyst batch parse error for {len(batch)} matches "
                f"(finish_reason={get_last_finish_reason()}): {text[:300]}"
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

        # Same isolation principle as run_analyst: a batch Gemini genuinely
        # can't serve right now costs us that batch's matches, not every
        # match already approved into `safe` from earlier batches.
        try:
            text = _gemini_chat(
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
                f"— Gemini unavailable: {e}"
            )
            continue

        data = _extract_json(text)
        assessments = data.get("assessments") if isinstance(data, dict) else None
        if not isinstance(assessments, list) or not assessments:
            logger.error(
                f"Risk batch parse error for {len(batch)} matches "
                f"(finish_reason={get_last_finish_reason()}): {text[:300]}"
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

    # This is the most reasoning-heavy call in the pipeline; it still uses
    # bounded JSON output and the same shared Gemini client as every other
    # agent.
    try:
        text = _gemini_chat(
            max_tokens=4000,
            messages=[
                {"role": "system", "content": PORTFOLIO_PROMPT},
                {
                    "role": "user",
                    "content": f"Select matches and markets from these safe candidates:\n{json.dumps(safe_matches, indent=2)}"
                }
            ]
        )
    except RuntimeError as e:
        # Gemini genuinely unavailable (rate-limited across the whole fallback
        # chain, with a wait too long to block on — see
        # GEMINI_MAX_SINGLE_WAIT_SECONDS). Degrade to NO_BET, same as every
        # other failure mode here, instead of letting this crash the whole
        # daily_run and lose the Scout/Analyst/Risk work that already
        # succeeded upstream in this same run.
        logger.error(f"Portfolio agent unavailable: {e}")
        return {"decision": "NO_BET", "reason": f"Portfolio agent unavailable: {e}"}
    data = _extract_json(text)
    if not data:
        logger.error(
            f"Portfolio parse error (finish_reason={get_last_finish_reason()}, "
            f"content_len={len(text)}): {text[:200]}"
        )
        reason = (
            "Portfolio construction failed: model output was cut off before "
            "completing valid JSON, even after a retry with more tokens "
            "(finish_reason=length)."
            if get_last_finish_reason() == "length"
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
        text = _gemini_chat(
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
        logger.error(f"Auditor parse error (finish_reason={get_last_finish_reason()}): {text[:200]}")
        return {"auditor_verdict": "REJECT", "critical_flags": ["Auditor system error — could not parse response."]}
    return data


def run_decision(audited: dict, portfolio: dict) -> dict:
    if portfolio.get("decision") == "NO_BET":
        return {"decision": "NO_BET", "reason": portfolio.get("reason", "No valid portfolio constructed."), "final_confidence": 0.0}

    if audited.get("auditor_verdict") == "REJECT":
        return {"decision": "NO_BET", "reason": f"Auditor rejected: {audited.get('critical_flags')}", "final_confidence": 0.0}

    try:
        text = _gemini_chat(
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
        logger.error(f"Decision parse error (finish_reason={get_last_finish_reason()}): {text[:200]}")
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
        return _gemini_chat(
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
