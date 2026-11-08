"""
ANALYST / RISK / PORTFOLIO / AUDITOR / DECISION / PUBLISHER AGENTS — Football Pulse AI
"""

import json
import logging
import math
import os
import re

from backend.gemini_client import gemini_chat as _gemini_chat, get_last_finish_reason
from backend.models.dixon_coles import MODEL_VERSION
from backend.models.markets import blend_probabilities, normalized_implied, value_metrics
from backend.models.learning import calibrate_probability
from backend.models.portfolio_optimizer import (
    PortfolioLimits,
    optimize_portfolio,
    portfolio_utility as shared_portfolio_utility,
)
from backend.models.conservative_shadow import build_conservative_shadow

logger = logging.getLogger(__name__)

JSON_RULES = """

IMPORTANT OUTPUT RULES:
- Respond with ONLY the JSON object. No markdown code fences, no commentary, no explanation before or after.
- The response must be valid JSON that can be parsed directly with json.loads()."""

# ---------------------------------------------------------------------------
# Confidence threshold + edge margin — single source of truth.

CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.60"))
MIN_EDGE_MARGIN = float(os.environ.get("MIN_EDGE_MARGIN", "0.0"))

# Manual override. When enabled, a valid portfolio is published even if the
# Auditor or final Decision agent rejects it.
FORCE_PUBLISH = (
    os.environ.get("FORCE_PUBLISH", "false").strip().lower() == "true"
)

# ---------------------------------------------------------------------------
# Combined odds configuration.

COMBINED_ODDS_MIN = float(os.environ.get("COMBINED_ODDS_MIN", "1.0"))
COMBINED_ODDS_MAX = float(os.environ.get("COMBINED_ODDS_MAX", "0.0"))  # 0 = no fixed cap
MIN_SELECTIONS = max(1, int(os.environ.get("MIN_SELECTIONS", "1")))
MAX_SELECTIONS = max(0, int(os.environ.get("MAX_SELECTIONS", "0")))  # 0 = all qualifying fixtures
MIN_SELECTION_PROBABILITY = float(os.environ.get("MIN_SELECTION_PROBABILITY", "0.50"))
POISSON_MARKET_WEIGHT = float(os.environ.get("POISSON_MARKET_WEIGHT", "0.35"))
MIN_TEAM_HISTORY = int(os.environ.get("MIN_TEAM_HISTORY", "8"))
PORTFOLIO_MAX_STAKE = min(0.10, max(0.0, float(os.environ.get("PORTFOLIO_MAX_STAKE", "0.02"))))
PORTFOLIO_BEAM_WIDTH = max(20, int(os.environ.get("PORTFOLIO_BEAM_WIDTH", "200")))
SELECTION_LIMIT_TEXT = (
    f"{MIN_SELECTIONS} or more selections, with no fixed maximum"
    if MAX_SELECTIONS == 0 else f"{MIN_SELECTIONS}-{MAX_SELECTIONS} selections"
)
ODDS_LIMIT_TEXT = (
    f"combined odds >= {COMBINED_ODDS_MIN:.2f}, with no fixed maximum"
    if COMBINED_ODDS_MAX <= 0 else f"combined odds {COMBINED_ODDS_MIN:.2f}-{COMBINED_ODDS_MAX:.2f}"
)

SUPPORTED_MARKETS = (
    "home_win", "draw", "away_win", "btts_yes", "over25",
    "double_chance_home", "double_chance_away",
    "draw_no_bet_home", "draw_no_bet_away",
)
TOURNAMENT_COMPETITIONS = {"CL", "EL", "UCL", "DFB", "FAC", "CIT", "CDR"}

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
- {ODDS_LIMIT_TEXT}
- {SELECTION_LIMIT_TEXT}; no leg-count or odds target is imposed
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
📅 {date}  |  🕗 07:00 EAT
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
    """Return only a directly quoted, executable price for a supported market.

    The historical helper name is retained for internal compatibility, but no
    price is derived. In particular, 1X2 quotes cannot prove that a DNB or
    double-chance ticket was actually available at a calculated price.
    """
    if market not in SUPPORTED_MARKETS:
        return None
    return _valid(odds_snapshot.get(market))


def _valid(odds: float | None) -> float | None:
    if odds is None:
        return None

    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(odds) or odds < 1.01:
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


def _complete_market(odds: dict, keys: tuple[str, ...]) -> tuple[dict, float | None]:
    if any(odds.get(key) is None for key in keys):
        return {}, None
    return normalized_implied({key: odds.get(key) for key in keys})


def _blend_two_way(blended: dict, market: dict, yes_key: str, no_key: str) -> None:
    if yes_key not in market or no_key not in market:
        return
    weight = POISSON_MARKET_WEIGHT
    blended[yes_key] = (1 - weight) * blended[yes_key] + weight * market[yes_key]
    blended[no_key] = (1 - weight) * blended[no_key] + weight * market[no_key]
    total = blended[yes_key] + blended[no_key]
    blended[yes_key] /= total
    blended[no_key] /= total


def run_analyst(clean_matches: list[dict], poisson_context: dict) -> list[dict]:
    """Compute deterministic Dixon-Coles probabilities for usable fixtures."""
    probabilities = []
    models = poisson_context.get("models", {})

    for match in clean_matches:
        code = match.get("competition_code")
        model = models.get(code)
        home_id = match.get("home_team_id")
        away_id = match.get("away_team_id")
        label = f"{match.get('home_team')} vs {match.get('away_team')}"

        if model is None:
            logger.warning(f"[ANALYST] {label}: no fitted {code} history; skipping.")
            continue
        try:
            home_id, away_id = int(home_id), int(away_id)
            model_markets = model.probabilities(home_id, away_id)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"[ANALYST] {label}: unavailable team history ({exc}); skipping.")
            continue

        home_history = model.matches_per_team.get(home_id, 0)
        away_history = model.matches_per_team.get(away_id, 0)
        # Knockout tournaments have fewer matches per team than leagues. Use a
        # lower floor only for those competitions, with multi-season history.
        required_history = min(MIN_TEAM_HISTORY, 4) if code in TOURNAMENT_COMPETITIONS else MIN_TEAM_HISTORY
        if min(home_history, away_history) < required_history:
            logger.warning(
                f"[ANALYST] {label}: only {home_history}/{away_history} historical matches; "
                f"minimum is {required_history}."
            )
            continue

        odds = match.get("odds_snapshot", {}) or {}
        market_1x2, overround_1x2 = _complete_market(odds, ("home_win", "draw", "away_win"))
        market_totals, overround_totals = _complete_market(odds, ("over25", "under25"))
        market_btts, overround_btts = _complete_market(odds, ("btts_yes", "btts_no"))
        market_probabilities = {**market_1x2, **market_totals, **market_btts}
        if market_1x2:
            market_probabilities["double_chance_home"] = market_1x2["home_win"] + market_1x2["draw"]
            market_probabilities["double_chance_away"] = market_1x2["away_win"] + market_1x2["draw"]
            non_draw = max(market_1x2["home_win"] + market_1x2["away_win"], 1e-12)
            market_probabilities["draw_no_bet_home"] = market_1x2["home_win"] / non_draw
            market_probabilities["draw_no_bet_away"] = market_1x2["away_win"] / non_draw

        blended = blend_probabilities(model_markets, market_1x2, POISSON_MARKET_WEIGHT)
        _blend_two_way(blended, market_totals, "over25", "under25")
        _blend_two_way(blended, market_btts, "btts_yes", "btts_no")

        recent_form = match.get("recent_form") or {}
        standings = match.get("standings") or {}
        feature_snapshot = {
            "model_version": MODEL_VERSION,
            "competition_code": code,
            "training_matches": model.training_matches,
            "home_team_history": home_history,
            "away_team_history": away_history,
            "minimum_team_history": required_history,
            "time_decay_days": model.time_decay_days,
            "low_score_rho": round(model.rho, 6),
            "home_advantage_log": round(model.home_advantage, 6),
            "expected_home_goals": round(model_markets["expected_home_goals"], 4),
            "expected_away_goals": round(model_markets["expected_away_goals"], 4),
            "market_weight": POISSON_MARKET_WEIGHT,
            "market_1x2_overround": overround_1x2,
            "totals_overround": overround_totals,
            "btts_overround": overround_btts,
            "odds_source": match.get("odds_source"),
            "odds_bookmakers": match.get("odds_bookmakers", []),
            "odds_last_update": match.get("odds_last_update"),
            "price_sources": odds.get("_price_sources", {}),
            "probability_calibration": poisson_context.get("calibration", {"active": False}),
        }
        item = {
            **match,
            "fixture_id": match.get("fixture_id"),
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "league": match.get("league"),
            "markets": {key: round(float(value), 6) for key, value in blended.items()},
            "poisson_markets": {
                key: round(float(value), 6) for key, value in model_markets.items()
            },
            "market_probabilities": market_probabilities,
            "model_version": MODEL_VERSION,
            "probability_calibration": poisson_context.get("calibration", {"active": False}),
            "feature_snapshot": feature_snapshot,
            "model_confidence": round(max(blended[key] for key in ("home_win", "draw", "away_win")), 4),
            "supporting_stats": {
                "home_form": str((recent_form.get("home") or {}).get("results", "UNKNOWN")),
                "away_form": str((recent_form.get("away") or {}).get("results", "UNKNOWN")),
                "head_to_head": str(match.get("head_to_head") or "UNKNOWN"),
                "standings_gap": str(standings or "UNKNOWN"),
                "key_injuries": str(match.get("injuries") or "UNKNOWN"),
            },
            "key_factors": [
                f"Expected goals: {model_markets['expected_home_goals']:.2f}-{model_markets['expected_away_goals']:.2f}",
                f"Time-weighted history: {model.training_matches} league matches",
                f"Team history: {home_history} home / {away_history} away team matches",
                f"Market shrinkage weight: {POISSON_MARKET_WEIGHT:.0%} where complete prices exist",
            ],
            "confidence_calculation": [
                "Dixon-Coles attack/defence ratings fitted by time-weighted maximum likelihood.",
                "Low-score adjustment and normalized 0-0 through 10-10 score matrix applied.",
                "Confidence is the largest blended 1X2 probability, not a win guarantee.",
            ],
            "odds_snapshot": odds,
        }
        probabilities.append(item)
        logger.info(
            f"[ANALYST] {label}: xG {model_markets['expected_home_goals']:.2f}-"
            f"{model_markets['expected_away_goals']:.2f}; 1X2 "
            f"{blended['home_win']:.3f}/{blended['draw']:.3f}/{blended['away_win']:.3f}."
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


def _portfolio_utility(selections: list[dict]) -> tuple[float, float, float, float]:
    """Expected log growth with a capped Kelly stake for the whole ticket."""
    return shared_portfolio_utility(
        selections, probability_key="model_prob", max_stake=PORTFOLIO_MAX_STAKE
    )


def _portfolio_limits() -> PortfolioLimits:
    return PortfolioLimits(
        min_selections=MIN_SELECTIONS,
        max_selections=MAX_SELECTIONS,
        combined_odds_min=COMBINED_ODDS_MIN,
        combined_odds_max=COMBINED_ODDS_MAX,
        beam_width=PORTFOLIO_BEAM_WIDTH,
        max_stake=PORTFOLIO_MAX_STAKE,
    )


def run_portfolio(safe_matches: list[dict]) -> dict:
    """Build a risk-adjusted portfolio with no mandatory leg/odds ceiling."""
    candidates = []
    evaluations = []

    for match in safe_matches:
        odds_snapshot = match.get("odds_snapshot", {}) or {}
        model_markets = match.get("markets") or {}
        calibration = match.get("probability_calibration") or {"active": False}
        for market in SUPPORTED_MARKETS:
            raw_model_prob = model_markets.get(market)
            model_prob = (
                calibrate_probability(raw_model_prob, calibration)
                if raw_model_prob is not None else None
            )
            odds = _derive_odds(market, odds_snapshot)
            evaluation = {
                "fixture_id": match.get("fixture_id"),
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "league": match.get("league"),
                "kickoff_utc": match.get("kickoff_utc"),
                "market": market,
                "model_version": match.get("model_version", MODEL_VERSION),
                "calibration_version": calibration.get("version"),
                "raw_model_prob": raw_model_prob,
                "model_prob": model_prob,
                "bookmaker_odds": odds,
                "market_implied_prob": (match.get("market_probabilities") or {}).get(market),
                "odds_source": "bookmaker",
                "feature_snapshot": match.get("feature_snapshot", {}),
                "eligible": False,
                "selected": False,
            }
            if odds is None or model_prob is None:
                evaluation["rejection_reason"] = "Missing valid price or model probability."
                evaluations.append(evaluation)
                continue
            try:
                metrics = value_metrics(float(model_prob), odds)
            except ValueError as exc:
                evaluation["rejection_reason"] = str(exc)
                evaluations.append(evaluation)
                continue

            evaluation.update({
                "bookmaker_implied_prob": metrics["bookmaker_implied_prob"],
                "fair_odds": metrics["fair_odds"],
                "edge": metrics["edge"],
                "expected_value": metrics["expected_value"],
            })
            reasons = []
            if float(model_prob) < MIN_SELECTION_PROBABILITY:
                reasons.append(
                    f"Probability {model_prob:.3f} below {MIN_SELECTION_PROBABILITY:.3f}."
                )
            if metrics["expected_value"] < 0:
                reasons.append(f"Negative expected value {metrics['expected_value']:.3f}.")
            if metrics["edge"] < MIN_EDGE_MARGIN:
                reasons.append(
                    f"Edge {metrics['edge']:.3f} below {MIN_EDGE_MARGIN:.3f}."
                )
            evaluation["rejection_reason"] = " ".join(reasons) or None
            evaluation["eligible"] = not reasons
            evaluations.append(evaluation)
            if reasons:
                continue

            candidate = {
                "fixture_id": match.get("fixture_id"),
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "league": match.get("league"),
                "kickoff_utc": match.get("kickoff_utc"),
                "market": market,
                "odds": round(float(odds), 3),
                "raw_model_prob": float(raw_model_prob),
                "model_prob": float(model_prob),
                "implied_prob": metrics["bookmaker_implied_prob"],
                "market_implied_prob": evaluation["market_implied_prob"],
                "fair_odds": metrics["fair_odds"],
                "edge": metrics["edge"],
                "expected_value": metrics["expected_value"],
                "model_version": match.get("model_version", MODEL_VERSION),
                "calibration_version": calibration.get("version"),
                "feature_snapshot": match.get("feature_snapshot", {}),
                "odds_source": evaluation["odds_source"],
            }
            candidates.append(candidate)

    optimised = optimize_portfolio(
        candidates,
        _portfolio_limits(),
        probability_key="model_prob",
    )
    if optimised is None:
        return {
            "decision": "NO_BET",
            "reason": f"No positive-EV combination satisfies {SELECTION_LIMIT_TEXT} and {ODDS_LIMIT_TEXT}.",
            "candidate_evaluations": evaluations,
        }

    selections = optimised["selections"]
    combined_odds = optimised["combined_odds"]
    joint_probability = optimised["portfolio_confidence"]
    portfolio_ev = optimised["portfolio_expected_value"]
    selected_keys = {(s["fixture_id"], s["market"]) for s in selections}
    for evaluation in evaluations:
        evaluation["selected"] = (evaluation["fixture_id"], evaluation["market"]) in selected_keys

    for sel in selections:
        features = sel.get("feature_snapshot") or {}
        calibrated = (features.get("probability_calibration") or {}).get("active", False)
        sel["rationale"] = (
            f"{'Calibrated ' if calibrated else ''}Dixon-Coles probability {sel['model_prob']:.1%}; "
            f"fair odds {sel['fair_odds']:.2f}; price {sel['odds']:.2f}; EV {sel['expected_value']:+.1%}. "
            f"Expected goals {features.get('expected_home_goals', 'N/A')}-"
            f"{features.get('expected_away_goals', 'N/A')} from "
            f"{features.get('training_matches', 'N/A')} weighted league matches."
        )
        sel["implied_prob"] = round(sel["implied_prob"], 6)
        sel["fair_odds"] = round(sel["fair_odds"], 4)
        sel["edge"] = round(sel["edge"], 6)
        sel["expected_value"] = round(sel["expected_value"], 6)
        sel.pop("_single_utility", None)

    risk_level = "Low" if joint_probability >= 0.45 else "Medium" if joint_probability >= 0.20 else "High"
    return {
        "selections": selections,
        "combined_odds": round(combined_odds, 2),
        "portfolio_confidence": joint_probability,
        "portfolio_expected_value": portfolio_ev,
        "model_version": MODEL_VERSION,
        "rationale": (
            f"Risk-adjusted {len(selections)}-leg portfolio; no maximum leg count or odds target was imposed. "
            "Joint probability assumes independence between different fixtures and is an estimate, not a guarantee."
        ),
        "risk_level": risk_level,
        "candidate_evaluations": evaluations,
    }


def run_conservative_shadow(candidate_evaluations: list[dict]) -> dict:
    """Evaluate the pre-registered alternative without publishing it."""
    return build_conservative_shadow(candidate_evaluations, _portfolio_limits())


def _portfolio_for_llm(portfolio: dict) -> dict:
    """Keep rejected-candidate diagnostics and feature blobs out of LLM prompts."""
    compact = {key: value for key, value in portfolio.items() if key not in ("candidate_evaluations", "selections")}
    compact["selections"] = [
        {
            key: sel.get(key)
            for key in (
                "fixture_id", "home_team", "away_team", "league", "market", "odds",
                "model_prob", "implied_prob", "fair_odds", "expected_value", "odds_source", "rationale",
            )
        }
        for sel in portfolio.get("selections", [])
    ]
    return compact


def validate_portfolio(portfolio: dict) -> list[str]:
    """Hard mathematical/data gates. FORCE_PUBLISH never bypasses these."""
    errors = []
    selections = portfolio.get("selections") or []
    if len(selections) < MIN_SELECTIONS or (MAX_SELECTIONS > 0 and len(selections) > MAX_SELECTIONS):
        errors.append(f"Selection count {len(selections)} violates {SELECTION_LIMIT_TEXT}.")
        return errors
    fixture_ids = [sel.get("fixture_id") for sel in selections]
    if any(fixture_id is None for fixture_id in fixture_ids) or len(set(fixture_ids)) != len(fixture_ids):
        errors.append("Every selection needs a fixture ID and only one market per fixture is allowed.")
    try:
        combined = math.prod(float(sel.get("odds")) for sel in selections)
    except (TypeError, ValueError):
        return errors + ["Invalid decimal odds in portfolio."]
    if not math.isfinite(combined) or combined < COMBINED_ODDS_MIN:
        errors.append(f"Combined odds {combined} are invalid or below {COMBINED_ODDS_MIN:.2f}.")
    if COMBINED_ODDS_MAX > 0 and combined > COMBINED_ODDS_MAX + 1e-9:
        errors.append(f"Combined odds {combined:.4f} exceed configured maximum {COMBINED_ODDS_MAX:.2f}.")
    try:
        reported_combined = float(portfolio.get("combined_odds"))
        if abs(reported_combined - combined) > 0.011:
            errors.append(
                f"Reported combined odds {reported_combined} do not match selection product {combined:.4f}."
            )
    except (TypeError, ValueError):
        errors.append("Portfolio is missing valid reported combined odds.")
    for sel in selections:
        try:
            probability = float(sel.get("model_prob"))
            odds = float(sel.get("odds"))
            metrics = value_metrics(probability, odds)
        except (TypeError, ValueError):
            errors.append(f"Invalid probability/odds for fixture {sel.get('fixture_id')}.")
            continue
        if sel.get("market") not in SUPPORTED_MARKETS:
            errors.append(f"Unsupported market {sel.get('market')}.")
        if not str(sel.get("model_version", "")).startswith("dixon_coles_"):
            errors.append(f"Fixture {sel.get('fixture_id')} lacks Dixon-Coles model metadata.")
        features = sel.get("feature_snapshot") or {}
        if not features:
            errors.append(f"Fixture {sel.get('fixture_id')} lacks model feature metadata.")
        else:
            try:
                training_matches = int(features.get("training_matches", 0))
                home_history = int(features.get("home_team_history", 0))
                away_history = int(features.get("away_team_history", 0))
                required_history = int(features.get("minimum_team_history", MIN_TEAM_HISTORY))
                if training_matches < 24 or min(home_history, away_history) < required_history:
                    errors.append(f"Fixture {sel.get('fixture_id')} has insufficient model history.")
            except (TypeError, ValueError):
                errors.append(f"Fixture {sel.get('fixture_id')} has invalid model history metadata.")
        if sel.get("odds_source") not in ("bookmaker", "derived_from_bookmaker_1x2"):
            errors.append(f"Fixture {sel.get('fixture_id')} lacks a valid odds source.")
        if probability < MIN_SELECTION_PROBABILITY:
            errors.append(f"Fixture {sel.get('fixture_id')} probability below safety minimum.")
        if metrics["expected_value"] < -1e-9:
            errors.append(f"Fixture {sel.get('fixture_id')} has negative expected value.")
        if metrics["edge"] < MIN_EDGE_MARGIN - 1e-9:
            errors.append(f"Fixture {sel.get('fixture_id')} is below the minimum edge.")
    return errors

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
                        f"{json.dumps(_portfolio_for_llm(portfolio), indent=2)}"
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

    hard_errors = validate_portfolio(portfolio)
    if hard_errors:
        logger.error(f"[DECISION] Hard portfolio validation failed: {hard_errors}")
        return {
            "decision": "NO_BET",
            "reason": "Hard portfolio validation failed: " + " ".join(hard_errors),
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
                        f"Portfolio: {json.dumps(_portfolio_for_llm(portfolio))}\n"
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


    return data


def _plain_published_ticket(portfolio: dict, decision: dict, target_date: str, note: str = "") -> str:
    selections_text = "\n\n".join(
        f"MATCH: {s.get('home_team')} vs {s.get('away_team')} ({s.get('league')})\n"
        f"Market: {s.get('market')}\nOdds: {s.get('odds')}\n"
        f"Reason: {s.get('rationale', '')}"
        for s in portfolio.get("selections", [])
    )
    return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {target_date}  |  🕗 07:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Joint probability estimate: {decision.get('final_confidence', 'N/A')}
⚠️  Overall Risk: {portfolio.get('risk_level', 'N/A')}
{note}

{selections_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Combined Odds: {portfolio.get('combined_odds', 'N/A')}

⚠️  DISCLAIMER
This is a probabilistic model output. No prediction is guaranteed.
Bet only what you can afford to lose.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def run_publisher(
    portfolio: dict,
    decision: dict,
    audited: dict,
    target_date: str,
) -> str:
    if decision.get("decision") == "NO_BET":
        return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔵 FOOTBALL PULSE AI
📅 {target_date}  |  🕗 07:00 EAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚫 NO BET TODAY

Reason: {decision.get('reason', 'Insufficient edge detected.')}

The system found no selections meeting the
{CONFIDENCE_THRESHOLD * 100:.0f}%+ confidence threshold today.

Discipline over volume. We wait.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

    if len(portfolio.get("selections", [])) > 12:
        return _plain_published_ticket(
            portfolio, decision, target_date,
            "Full portfolio formatted deterministically so no selection is truncated.",
        )

    try:
        return _gemini_chat(
            max_tokens=min(8000, max(1200, 500 + 300 * len(portfolio.get("selections", [])))),
            messages=[
                {
                    "role": "system",
                    "content": PUBLISHER_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"""Format the final ticket.

Date: {target_date}
Portfolio: {json.dumps(_portfolio_for_llm(portfolio))}
Decision: {json.dumps(decision)}
Audited: {json.dumps(audited)}""",
                },
            ],
        )

    except RuntimeError as e:
        logger.error(f"Publisher agent unavailable: {e}")
        return _plain_published_ticket(
            portfolio, decision, target_date, "Publisher agent unavailable; using plain fallback."
        )
