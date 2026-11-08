"""Pre-registered conservative portfolio evaluated in shadow mode only."""

from __future__ import annotations

import math

from backend.models.portfolio_optimizer import PortfolioLimits, optimize_portfolio

SHADOW_STRATEGY = "robust_conservative_v1"
SHADOW_CONFIG = {
    "model_weight": 0.50,
    "market_weight": 0.50,
    "min_probability": 0.55,
    "min_expected_value": 0.03,
    "allowed_odds_source": "bookmaker",
}


def build_conservative_shadow(
    evaluations: list[dict], limits: PortfolioLimits
) -> dict:
    """Build a private alternative without changing the production portfolio.

    The probability anchor deliberately gives the no-vig market estimate as much
    weight as the model. Derived prices and missing baselines are rejected.
    """
    shadow_evaluations = []
    candidates = []
    model_weight = SHADOW_CONFIG["model_weight"]
    market_weight = SHADOW_CONFIG["market_weight"]

    for original in evaluations:
        item = dict(original)
        item["feature_snapshot"] = dict(original.get("feature_snapshot") or {})
        item.update({
            "strategy": SHADOW_STRATEGY,
            "selected": False,
            "shadow_selected": False,
            "eligible": False,
            "robust_model_prob": None,
            "robust_expected_value": None,
        })
        reasons = []
        try:
            model_probability = float(item["model_prob"])
            market_probability = float(item["market_implied_prob"])
            odds = float(item["bookmaker_odds"])
        except (KeyError, TypeError, ValueError):
            model_probability = market_probability = odds = 0.0
            reasons.append("Missing model, market baseline, or genuine price.")

        if item.get("odds_source") != SHADOW_CONFIG["allowed_odds_source"]:
            reasons.append("Shadow strategy requires a genuine bookmaker price, not a derived one.")
        if not item.get("kickoff_utc"):
            reasons.append("Shadow strategy requires kickoff_utc.")

        robust_probability = None
        robust_ev = None
        if not reasons or (model_probability and market_probability and odds):
            if (
                not math.isfinite(model_probability)
                or not math.isfinite(market_probability)
                or not math.isfinite(odds)
                or not 0 < model_probability < 1
                or not 0 < market_probability < 1
                or odds <= 1
            ):
                reasons.append("Invalid probabilities or decimal odds.")
            else:
                robust_probability = (
                    model_weight * model_probability + market_weight * market_probability
                )
                robust_probability = min(1 - 1e-6, max(1e-6, robust_probability))
                robust_ev = robust_probability * odds - 1.0
                item["robust_model_prob"] = robust_probability
                item["robust_expected_value"] = robust_ev

                if robust_probability < SHADOW_CONFIG["min_probability"]:
                    reasons.append(
                        f"Robust probability {robust_probability:.3f} below "
                        f"{SHADOW_CONFIG['min_probability']:.3f}."
                    )
                if robust_ev < SHADOW_CONFIG["min_expected_value"]:
                    reasons.append(
                        f"Robust EV {robust_ev:.3f} below "
                        f"{SHADOW_CONFIG['min_expected_value']:.3f}."
                    )

        item["rejection_reason"] = " ".join(reasons) or None
        item["eligible"] = not reasons
        shadow_evaluations.append(item)
        if reasons:
            continue

        candidates.append({
            "fixture_id": item.get("fixture_id"),
            "home_team": item.get("home_team"),
            "away_team": item.get("away_team"),
            "league": item.get("league"),
            "kickoff_utc": item.get("kickoff_utc"),
            "market": item.get("market"),
            "odds": round(odds, 3),
            "raw_model_prob": item.get("raw_model_prob"),
            "model_prob": model_probability,
            "market_implied_prob": market_probability,
            "robust_model_prob": robust_probability,
            "robust_expected_value": robust_ev,
            "model_version": item.get("model_version"),
            "calibration_version": item.get("calibration_version"),
            "feature_snapshot": item.get("feature_snapshot"),
            "odds_source": item.get("odds_source"),
        })

    optimised = optimize_portfolio(
        candidates, limits, probability_key="robust_model_prob"
    )
    selections = optimised["selections"] if optimised else []
    selected_keys = {(s["fixture_id"], s["market"]) for s in selections}
    for item in shadow_evaluations:
        item["shadow_selected"] = (item.get("fixture_id"), item.get("market")) in selected_keys

    for selection in selections:
        selection.pop("_single_utility", None)
        selection["rationale"] = (
            f"SHADOW ONLY {SHADOW_STRATEGY}: robust probability "
            f"{selection['robust_model_prob']:.1%} ({model_weight:.0%} model + "
            f"{market_weight:.0%} market); price {selection['odds']:.2f}; "
            f"robust EV {selection['robust_expected_value']:+.1%}. Not an official pick."
        )

    return {
        "decision": "SHADOW_ONLY" if selections else "NO_BET",
        "is_shadow": True,
        "strategy": SHADOW_STRATEGY,
        "config": dict(SHADOW_CONFIG),
        "selections": selections,
        "candidate_evaluations": shadow_evaluations,
        "combined_odds": optimised["combined_odds"] if optimised else None,
        "portfolio_confidence": optimised["portfolio_confidence"] if optimised else 0.0,
        "portfolio_expected_value": optimised["portfolio_expected_value"] if optimised else None,
        "reason": (
            "Experimental shadow evaluation only; not published or emailed as an official ticket. "
            "Parameters are pre-registered and must be assessed on later outcomes."
        ),
    }
