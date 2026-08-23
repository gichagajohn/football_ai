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

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))
MIN_EDGE_MARGIN = float(os.environ.get("MIN_EDGE_MARGIN", "0.03"))

# Manual override. When enabled, a valid portfolio is published even if the
# Auditor or final Decision agent rejects it.
FORCE_PUBLISH = (
    os.environ.get("FORCE_PUBLISH", "false").strip().lower() == "true"
)

# ---------------------------------------------------------------------------
# Combined odds configuration.

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
   in "confidence_calculation".
3. If two or more of recent_form/head_to_head/standings came back UNKNOWN
   for this match, cap your final model_confidence at 0.60.
4. Sum the adjustments onto the base for the final model_confidence.

Do NOT default to a "safe-sounding" round number out of habit. Different
matches with different supporting_stats should produce different confidence
values.

"confidence_calculation" must NEVER be an empty list. If there is no
adjustment, explain why.

Your analysis must be grounded in:
- Recent form from recent_form
- Head-to-head record from head_to_head
- xG data if available
- Home/away performance splits
- Injuries to key players
- League position and points gap from standings

Probabilities must sum to 1.0 for mutually exclusive markets.
All values must be between 0.0 and 1.0.""" + JSON_RULES

RISK_PROMPT = f"""You are the RISK AGENT for Football Pulse AI.
Your job is to REJECT dangerous selections.

IMPORTANT CONTEXT: This assessment happens roughly 24-31 hours BEFORE kickoff.
Official lineups are usually not confirmed this far in advance. Do not reject
a match for lacking lineup confirmation alone.

HARD REJECT RULES:
1. Adjusted confidence < {CONFIDENCE_THRESHOLD:.2f}
2. Both goalkeepers injured/suspended
3. Odds moved against our pick by >15% from open
4. Weather: wind >60 km/h or heavy snowfall
5. Strong rotation signals in cup/playoff matches
6. Team with 3+ key attackers injured/suspended

SOFT FLAGS:
- Lineup not yet confirmed
- Odds moved against pick by 8-15%
- One key player injured
- Minor weather concerns
- Long travel over 800km
- Schedule congestion

Return JSON: {{"approved": bool, "risk_level": "Low|Medium|High",
"flags": [...], "rejection_reason": str|null}}""" + JSON_RULES

PORTFOLIO_PROMPT = f"""You are the PORTFOLIO AGENT for Football Pulse AI.
Construct the daily prediction ticket by selecting matches and markets only.

You do not calculate odds yourself. Odds will be validated by the system after
you choose the selections.

TARGET: combined odds approximately {COMBINED_ODDS_TARGET:.1f}
ACCEPTABLE RANGE: {COMBINED_ODDS_MIN:.1f}-{COMBINED_ODDS_MAX:.1f}

RULES:
- Select 2-5 matches
- Prefer higher-confidence matches
- Use exact fixture IDs
- Use exact market keys
- Do not combine two markets from the same match
- Do not select correct score or first goalscorer markets
- Use only markets that can be priced from the available odds

Allowed markets:
- home_win
- away_win
- draw
- btts_yes
- over25
- double_chance_home
- double_chance_away
- draw_no_bet_home
- draw_no_bet_away

Output JSON:
{{"selections": [{{"fixture_id": int, "market": str, "rationale": str}}],
"portfolio_confidence": float, "rationale": str}}

If no portfolio can be built, output:
{{"decision": "NO_BET", "reason": str}}""" + JSON_RULES

AUDITOR_PROMPT = """You are the AUDITOR AGENT for Football Pulse AI.
Act as the devil's advocate and challenge every selection.

For each selection, ask:
1. What is the most likely way this loses?
2. What assumptions could be wrong?
3. What recent news could break this?
4. Is the market already pricing in the expected outcome?
5. How often does this market win at these odds?

Be critical. Adjust confidence downward where warranted.

Output JSON:
{"adjusted_selections": [...],
"overall_confidence_adjustment": float,
"critical_flags": [...],
"auditor_verdict": "APPROVE|REVISE|REJECT"}""" + JSON_RULES

DECISION_PROMPT = f"""You are the DECISION AGENT for Football Pulse AI.
You receive the final audited ticket and make the publish/no-bet call.

PUBLISH IF:
- Overall confidence >= {CONFIDENCE_THRESHOLD:.2f}
- Combined odds within {COMBINED_ODDS_MIN:.1f}-{COMBINED_ODDS_MAX:.1f}
- At least 2 selections are present
- No hard reject flags are active
- Auditor verdict is APPROVE or REVISE

NO BET IF:
- Any condition above fails
- The ticket looks like desperate volume instead of disciplined value

Output JSON:
{{"decision": "PUBLISH|NO_BET",
"reason": str,
"final_confidence": float}}""" + JSON_RULES

PUBLISHER_PROMPT = """You are the PUBLISHER AGENT for Football Pulse AI.
Format the final ticket as plain text.

Output exactly this format:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {date}  |  🕗 08:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Confidence: {confidence}%
⚠️  Overall Risk: {risk_level}

For each selection:

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

Use the exact odds and combined_odds values provided.
Respond with only the formatted ticket text."""


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

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.MULTILINE,
    )

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
            adj_p_home = (
                p_home / (p_home + p_away)
                if (p_home + p_away) > 0
                else None
            )
            return _valid(round(1 / adj_p_home, 3)) if adj_p_home else None

        if market == "draw_no_bet_away" and away and draw:
            p_home = 1 / home if home else 0
            p_draw = 1 / draw
            p_away = 1 / away
            adj_p_away = (
                p_away / (p_home + p_away)
                if (p_home + p_away) > 0
                else None
            )
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


ANALYST_BATCH_SIZE = int(os.environ.get("ANALYST_BATCH_SIZE", "4"))
RISK_BATCH_SIZE = int(os.environ.get("RISK_BATCH_SIZE", "4"))

ANALYST_TOKENS_PER_MATCH = 900
ANALYST_TRUNCATION_CEILING = int(
    os.environ.get("ANALYST_TRUNCATION_CEILING", "12000")
)

RISK_TOKENS_PER_MATCH = 350
RISK_TRUNCATION_CEILING = int(
    os.environ.get("RISK_TRUNCATION_CEILING", "6000")
)


def _chunk(items: list, size: int) -> list[list]:
    return [
        items[i:i + size]
        for i in range(0, len(items), max(1, size))
    ]


def run_analyst(clean_matches: list[dict]) -> list[dict]:
    probabilities = []

    for batch in _chunk(clean_matches, ANALYST_BATCH_SIZE):
        match_lookup = {
            m.get("fixture_id"): m
            for m in batch
        }

        max_tokens = min(
            ANALYST_TOKENS_PER_MATCH * len(batch) + 500,
            ANALYST_TRUNCATION_CEILING,
        )

        try:
            text = _gemini_chat(
                max_tokens=max_tokens,
                truncation_ceiling=ANALYST_TRUNCATION_CEILING,
                messages=[
                    {
                        "role": "system",
                        "content": ANALYST_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"""Estimate probabilities for EACH of the following
{len(batch)} matches.

Apply the full ANALYST AGENT instructions above to EVERY match independently.

MATCHES:
{json.dumps(batch, indent=2)}

Return a single JSON object:
{{"analyses": [ ... ]}}

Return exactly {len(batch)} analysis objects, one for each match.
Each item must contain:
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
}}""",
                    },
                ],
            )

        except RuntimeError as e:
            names = [
                f"{m.get('home_team')} vs {m.get('away_team')}"
                for m in batch
            ]

            logger.error(
                f"[ANALYST] Giving up on batch of {len(batch)} matches "
                f"({names}) — Gemini unavailable: {e}"
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

        claimed_ids = {
            item.get("fixture_id")
            for item in analyses
            if isinstance(item, dict)
            and item.get("fixture_id") in match_lookup
        }

        unclaimed = [
            m for m in batch
            if m.get("fixture_id") not in claimed_ids
        ]

        seen_fixture_ids = set()

        for item in analyses:
            if not isinstance(item, dict):
                continue

            fixture_id = item.get("fixture_id")
            match = match_lookup.get(fixture_id)

            if match is None:
                if len(unclaimed) == 1:
                    match = unclaimed.pop(0)
                    fixture_id = match.get("fixture_id")
                else:
                    logger.error(
                        f"[ANALYST] Batch item has unrecognized/missing "
                        f"fixture_id={fixture_id!r}; dropping item."
                    )
                    continue

            seen_fixture_ids.add(fixture_id)

            item["fixture_id"] = fixture_id
            item.setdefault("home_team", match.get("home_team"))
            item.setdefault("away_team", match.get("away_team"))
            item["odds_snapshot"] = match.get("odds_snapshot", {})
            item["league"] = match.get("league")
            probabilities.append(item)

            if not item.get("confidence_calculation"):
                logger.warning(
                    f"[ANALYST] {item.get('home_team')} vs "
                    f"{item.get('away_team')} returned an empty "
                    f"confidence_calculation."
                )

            logger.info(
                f"[ANALYST] {item.get('home_team')} vs "
                f"{item.get('away_team')} — "
                f"model_confidence={item.get('model_confidence')} "
                f"(calc: {item.get('confidence_calculation')})"
            )

        missing = [
            m for m in batch
            if m.get("fixture_id") not in seen_fixture_ids
        ]

        if missing:
            logger.warning(
                f"[ANALYST] {len(missing)} matches received no result."
            )

    return probabilities


def run_risk_filter(
    probabilities: list[dict],
    intelligence: list[dict],
) -> list[dict]:
    safe = []
    intel_by_fixture = {
        m.get("fixture_id"): m
        for m in intelligence
    }

    for batch in _chunk(probabilities, RISK_BATCH_SIZE):
        prob_lookup = {
            p.get("fixture_id"): p
            for p in batch
        }

        max_tokens = min(
            RISK_TOKENS_PER_MATCH * len(batch) + 300,
            RISK_TRUNCATION_CEILING,
        )

        payload = [
            {
                "fixture_id": prob.get("fixture_id"),
                "probabilities": prob,
                "intelligence": intel_by_fixture.get(
                    prob.get("fixture_id"),
                    {},
                ),
            }
            for prob in batch
        ]

        try:
            text = _gemini_chat(
                max_tokens=max_tokens,
                truncation_ceiling=RISK_TRUNCATION_CEILING,
                messages=[
                    {
                        "role": "system",
                        "content": RISK_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": f"""Evaluate risk for EACH of the following
{len(batch)} matches independently.

MATCHES:
{json.dumps(payload, indent=2)}

Return:
{{"assessments": [ ... ]}}

Return exactly {len(batch)} assessments:
{{
  "fixture_id": int,
  "approved": bool,
  "risk_level": "Low|Medium|High",
  "flags": [str],
  "rejection_reason": str|null
}}""",
                    },
                ],
            )

        except RuntimeError as e:
            names = [
                f"{p.get('home_team')} vs {p.get('away_team')}"
                for p in batch
            ]

            logger.error(
                f"[RISK] Giving up on batch of {len(batch)} matches "
                f"({names}) — Gemini unavailable: {e}"
            )
            continue

        data = _extract_json(text)
        assessments = (
            data.get("assessments")
            if isinstance(data, dict)
            else None
        )

        if not isinstance(assessments, list) or not assessments:
            logger.error(
                f"Risk batch parse error for {len(batch)} matches "
                f"(finish_reason={get_last_finish_reason()}): "
                f"{text[:300]}"
            )
            continue

        claimed_ids = {
            item.get("fixture_id")
            for item in assessments
            if isinstance(item, dict)
            and item.get("fixture_id") in prob_lookup
        }

        unclaimed = [
            p for p in batch
            if p.get("fixture_id") not in claimed_ids
        ]

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
                        f"fixture_id={fixture_id!r}; dropping item."
                    )
                    continue

            seen_fixture_ids.add(fixture_id)

            if item.get("approved"):
                prob["risk_assessment"] = item
                safe.append(prob)

                logger.info(
                    f"[RISK] Approved: {prob.get('home_team')} vs "
                    f"{prob.get('away_team')} — "
                    f"risk={item.get('risk_level')}"
                )
            else:
                logger.info(
                    f"[RISK] Rejected: {prob.get('home_team')} vs "
                    f"{prob.get('away_team')} — "
                    f"model_confidence={prob.get('model_confidence')} — "
                    f"{item.get('rejection_reason')}"
                )

        missing = [
            p for p in batch
            if p.get("fixture_id") not in seen_fixture_ids
        ]

        if missing:
            logger.warning(
                f"[RISK] {len(missing)} matches received no result."
            )

    return safe


def run_portfolio(safe_matches: list[dict]) -> dict:
    if len(safe_matches) < 2:
        return {
            "decision": "NO_BET",
            "reason": "Insufficient safe candidates after risk filtering.",
        }

    match_lookup = {
        m.get("fixture_id"): m
        for m in safe_matches
    }

    try:
        text = _gemini_chat(
            max_tokens=4000,
            messages=[
                {
                    "role": "system",
                    "content": PORTFOLIO_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Select matches and markets from these safe candidates:\n"
                        f"{json.dumps(safe_matches, indent=2)}"
                    ),
                },
            ],
        )

    except RuntimeError as e:
        logger.error(f"Portfolio agent unavailable: {e}")
        return {
            "decision": "NO_BET",
            "reason": f"Portfolio agent unavailable: {e}",
        }

    data = _extract_json(text)

    if not data:
        logger.error(
            f"Portfolio parse error "
            f"(finish_reason={get_last_finish_reason()}, "
            f"content_len={len(text)}): {text[:200]}"
        )

        reason = (
            "Portfolio construction failed: model output was cut off "
            "before completing valid JSON."
            if get_last_finish_reason() == "length"
            else "Portfolio construction failed: model returned "
            "unparseable content."
        )

        return {
            "decision": "NO_BET",
            "reason": reason,
        }

    if data.get("decision") == "NO_BET":
        return data

    raw_selections = data.get("selections", [])

    if len(raw_selections) < 2:
        return {
            "decision": "NO_BET",
            "reason": "Portfolio agent selected fewer than 2 matches.",
        }

    final_selections = []
    skipped = []

    for sel in raw_selections:
        fixture_id = sel.get("fixture_id")
        market = sel.get("market")
        match = match_lookup.get(fixture_id)

        if not match:
            skipped.append(
                f"fixture {fixture_id} not found in safe matches"
            )
            continue

        odds_snapshot = match.get("odds_snapshot", {}) or {}
        odds = _derive_odds(market, odds_snapshot)

        if odds is None:
            skipped.append(
                f"{match.get('home_team')} vs "
                f"{match.get('away_team')} ({market}): "
                "could not derive valid odds"
            )
            continue

        model_prob = (match.get("markets") or {}).get(market)

        if model_prob is None:
            skipped.append(
                f"{match.get('home_team')} vs "
                f"{match.get('away_team')} ({market}): "
                "no model probability available"
            )
            continue

        implied_prob = 1.0 / odds
        edge = model_prob - implied_prob

        if edge < MIN_EDGE_MARGIN:
            skipped.append(
                f"{match.get('home_team')} vs "
                f"{match.get('away_team')} ({market}): "
                f"no real edge — model {model_prob:.2f} vs "
                f"market-implied {implied_prob:.2f} "
                f"(edge {edge:+.2f} < required "
                f"{MIN_EDGE_MARGIN:.2f})"
            )
            continue

        final_selections.append(
            {
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
            }
        )

    if skipped:
        logger.info(f"[PORTFOLIO] Skipped selections: {skipped}")

    if len(final_selections) < 2:
        return {
            "decision": "NO_BET",
            "reason": (
                "Fewer than 2 selections had derivable odds. "
                f"Skipped: {skipped}"
            ),
        }

    final_selections = final_selections[:5]

    combined_odds = 1.0
    for selection in final_selections:
        combined_odds *= selection["odds"]

    return {
        "selections": final_selections,
        "combined_odds": round(combined_odds, 2),
        "portfolio_confidence": data.get("portfolio_confidence"),
        "rationale": data.get("rationale", ""),
        "risk_level": data.get("risk_level", "Medium"),
    }


def run_auditor(portfolio: dict) -> dict:
    if portfolio.get("decision") == "NO_BET":
        return {
            "auditor_verdict": "REJECT",
            "critical_flags": [
                "No portfolio to audit — already NO_BET."
            ],
        }

    try:
        text = _gemini_chat(
            max_tokens=1500,
            messages=[
                {
                    "role": "system",
                    "content": AUDITOR_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Challenge this ticket:\n"
                        f"{json.dumps(portfolio, indent=2)}"
                    ),
                },
            ],
        )

    except RuntimeError as e:
        logger.error(f"Auditor agent unavailable: {e}")
        return {
            "auditor_verdict": "REJECT",
            "critical_flags": [
                f"Auditor agent unavailable: {e}"
            ],
        }

    data = _extract_json(text)

    if not data:
        logger.error(
            f"Auditor parse error "
            f"(finish_reason={get_last_finish_reason()}): "
            f"{text[:200]}"
        )

        return {
            "auditor_verdict": "REJECT",
            "critical_flags": [
                "Auditor system error — could not parse response."
            ],
        }

    return data


def run_decision(audited: dict, portfolio: dict) -> dict:
    if portfolio.get("decision") == "NO_BET":
        return {
            "decision": "NO_BET",
            "reason": portfolio.get(
                "reason",
                "No valid portfolio constructed.",
            ),
            "final_confidence": 0.0,
        }

    if FORCE_PUBLISH and portfolio.get("selections"):
        try:
            confidence = float(
                portfolio.get("portfolio_confidence")
            )
        except (TypeError, ValueError):
            confidence = CONFIDENCE_THRESHOLD

        return {
            "decision": "PUBLISH",
            "reason": (
                "FORCE_PUBLISH is enabled; publishing the validated "
                "portfolio despite Auditor/Decision warnings. "
                f"Auditor flags: "
                f"{audited.get('critical_flags', [])}"
            ),
            "final_confidence": confidence,
        }

    if audited.get("auditor_verdict") == "REJECT":
        return {
            "decision": "NO_BET",
            "reason": (
                f"Auditor rejected: "
                f"{audited.get('critical_flags')}"
            ),
            "final_confidence": 0.0,
        }

    try:
        text = _gemini_chat(
            max_tokens=400,
            messages=[
                {
                    "role": "system",
                    "content": DECISION_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Portfolio: {json.dumps(portfolio)}\n"
                        f"Audit: {json.dumps(audited)}\n"
                        "Make the final decision."
                    ),
                },
            ],
        )

    except RuntimeError as e:
        logger.error(f"Decision agent unavailable: {e}")
        return {
            "decision": "NO_BET",
            "reason": f"Decision agent unavailable: {e}",
            "final_confidence": 0.0,
        }

    data = _extract_json(text)

    if not data:
        logger.error(
            f"Decision parse error "
            f"(finish_reason={get_last_finish_reason()}): "
            f"{text[:200]}"
        )

        return {
            "decision": "NO_BET",
            "reason": "Decision agent error — defaulting safe.",
            "final_confidence": 0.0,
        }

    final_confidence = data.get("final_confidence")
    combined_odds = portfolio.get("combined_odds")

    if (
        final_confidence is not None
        and final_confidence < CONFIDENCE_THRESHOLD
    ):
        if data.get("decision") == "PUBLISH":
            logger.warning(
                "[DECISION] Overriding LLM PUBLISH to NO_BET: "
                f"final_confidence={final_confidence} < "
                f"{CONFIDENCE_THRESHOLD:.2f}"
            )

        data["decision"] = "NO_BET"

        if "below" not in str(data.get("reason", "")).lower():
            data["reason"] = (
                f"Overridden to NO_BET: final_confidence "
                f"({final_confidence}) is below the "
                f"{CONFIDENCE_THRESHOLD:.2f} threshold. "
                f"Original reasoning: {data.get('reason', '')}"
            )

    if (
        combined_odds is not None
        and not (
            COMBINED_ODDS_MIN
            <= combined_odds
            <= COMBINED_ODDS_MAX
        )
    ):
        if data.get("decision") == "PUBLISH":
            logger.warning(
                "[DECISION] Overriding LLM PUBLISH to NO_BET: "
                f"combined_odds={combined_odds} outside "
                f"{COMBINED_ODDS_MIN:.1f}-"
                f"{COMBINED_ODDS_MAX:.1f}"
            )

        data["decision"] = "NO_BET"
        data["reason"] = (
            f"Overridden to NO_BET: combined_odds "
            f"({combined_odds}) is outside the required "
            f"{COMBINED_ODDS_MIN:.1f}-"
            f"{COMBINED_ODDS_MAX:.1f} range. "
            f"Original reasoning: {data.get('reason', '')}"
        )

    return data


def run_publisher(
    portfolio: dict,
    decision: dict,
    audited: dict,
    target_date: str,
) -> str:
    if decision.get("decision") == "NO_BET":
        return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {target_date}  |  🕗 08:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 NO BET TODAY

Reason: {decision.get('reason', 'Insufficient edge detected.')}

The system found no selections meeting the
{CONFIDENCE_THRESHOLD * 100:.0f}%+ confidence threshold today.

Discipline over volume. We wait.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    try:
        return _gemini_chat(
            max_tokens=1200,
            messages=[
                {
                    "role": "system",
                    "content": PUBLISHER_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""Format the final ticket.

Date: {target_date}
Portfolio: {json.dumps(portfolio)}
Decision: {json.dumps(decision)}
Audited: {json.dumps(audited)}""",
                },
            ],
        )

    except RuntimeError as e:
        logger.error(f"Publisher agent unavailable: {e}")

        selections_text = "\n".join(
            f"MATCH: {s.get('home_team')} vs "
            f"{s.get('away_team')} ({s.get('league')})\n"
            f"Market: {s.get('market')}\n"
            f"Odds: {s.get('odds')}\n"
            f"Reason: {s.get('rationale', '')}"
            for s in portfolio.get("selections", [])
        )

        return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {target_date}  |  🕗 08:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Confidence: {decision.get('final_confidence', 'N/A')}
⚠️  Overall Risk: {portfolio.get('risk_level', 'N/A')}

Publisher agent unavailable. Plain fallback summary:

{selections_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Combined Odds: {portfolio.get('combined_odds', 'N/A')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
