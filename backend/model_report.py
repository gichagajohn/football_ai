"""Calibration and flat-stake report from private logs + settled selections.

Run after applying the prediction-log migration:
    python -m backend.model_report
"""

from __future__ import annotations

import json
import logging
import math

from backend.db import supabase_client
from backend.models.evaluation import brier_score, flat_stake_roi, log_loss
from backend.models.portfolio_backtesting import settle_accumulator, summarize_bet_records

logger = logging.getLogger(__name__)


def summarize_predictions(logs: list[dict], selections: list[dict]) -> dict:
    selection_outcomes = {
        (str(row.get("ticket_date")), row.get("fixture_id"), row.get("market")): row.get("outcome")
        for row in selections
    }
    probabilities, outcomes, odds = [], [], []
    missing_outcomes = 0
    for row in logs:
        if not row.get("selected"):
            continue
        key = (str(row.get("ticket_date")), row.get("fixture_id"), row.get("market"))
        outcome = selection_outcomes.get(key)
        if outcome not in ("win", "loss"):
            missing_outcomes += 1
            continue
        try:
            probability = float(row["model_prob"])
            price = float(row["bookmaker_odds"])
        except (KeyError, TypeError, ValueError):
            missing_outcomes += 1
            continue
        if not (0 < probability < 1 and price > 1):
            missing_outcomes += 1
            continue
        probabilities.append(probability)
        outcomes.append(int(outcome == "win"))
        odds.append(price)

    buckets = []
    for lower in range(0, 10):
        low, high = lower / 10, (lower + 1) / 10
        indices = [
            i for i, probability in enumerate(probabilities)
            if low <= probability < high or (high == 1.0 and probability == 1.0)
        ]
        if not indices:
            continue
        buckets.append({
            "range": f"{low:.1f}-{high:.1f}",
            "count": len(indices),
            "mean_predicted_probability": sum(probabilities[i] for i in indices) / len(indices),
            "observed_win_rate": sum(outcomes[i] for i in indices) / len(indices),
        })

    count = len(outcomes)
    return {
        "settled_predictions": count,
        "wins": sum(outcomes),
        "losses": count - sum(outcomes),
        "pending_or_unmatched": missing_outcomes,
        "hit_rate": sum(outcomes) / count if count else None,
        "brier_score": brier_score(probabilities, outcomes) if count else None,
        "log_loss": log_loss(probabilities, outcomes) if count else None,
        "flat_stake_roi": flat_stake_roi(probabilities, odds, outcomes) if count else None,
        "calibration_buckets": buckets,
        "sample_warning": "Small samples are unstable; do not infer a guaranteed edge." if count < 100 else None,
    }


def _strategy_probability_summary(rows: list[dict], probability_key: str) -> dict:
    probabilities, outcomes = [], []
    for row in rows:
        if row.get("outcome") not in ("win", "loss"):
            continue
        try:
            probability = float(row.get(probability_key))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(probability) or not 0 < probability < 1:
            continue
        probabilities.append(probability)
        outcomes.append(int(row["outcome"] == "win"))
    if not outcomes:
        return {
            "predictions": 0, "brier_score": None, "log_loss": None,
            "mean_predicted_probability": None, "observed_win_rate": None,
        }
    return {
        "predictions": len(outcomes),
        "brier_score": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
        "mean_predicted_probability": sum(probabilities) / len(probabilities),
        "observed_win_rate": sum(outcomes) / len(outcomes),
    }


def summarize_strategy_predictions(logs: list[dict]) -> dict:
    """Compare settled production-optimizer and private shadow diagnostics.

    Production ``selected`` rows describe the optimizer portfolio, not whether
    the Auditor/Decision ultimately published it. Official results remain in
    ``summarize_predictions`` and the public ticket tables.
    """
    latest = {}
    for row in logs:
        strategy = row.get("strategy") or "production"
        key = (row.get("ticket_date"), row.get("fixture_id"), row.get("market"), strategy)
        latest[key] = row

    grouped: dict[str, list[dict]] = {}
    for row in latest.values():
        grouped.setdefault(row.get("strategy") or "production", []).append(row)

    report = {}
    for strategy, rows in sorted(grouped.items()):
        is_shadow = strategy != "production"
        selected_rows = [
            row for row in rows
            if bool(row.get("shadow_selected") if is_shadow else row.get("selected"))
        ]
        singles = []
        for index, row in enumerate(selected_rows):
            if row.get("outcome") not in ("win", "loss", "void"):
                continue
            try:
                price = float(row.get("bookmaker_odds"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price) or price <= 1:
                continue
            outcome = row["outcome"]
            gross_return = price if outcome == "win" else 1.0 if outcome == "void" else 0.0
            singles.append({
                "strategy": strategy,
                "date": str(row.get("ticket_date")),
                "fixture_id": row.get("fixture_id"),
                "fixture_ids": [row.get("fixture_id")],
                "league": row.get("league"),
                "market": row.get("market"),
                "odds": price,
                "closing_odds": row.get("closing_bookmaker_odds"),
                "outcome": outcome,
                "return": gross_return,
                "profit": gross_return - 1.0,
                "sequence": index,
            })

        by_date: dict[str, list[dict]] = {}
        for row in singles:
            by_date.setdefault(row["date"], []).append(row)
        accumulators = [
            settle_accumulator(day_rows, strategy=f"{strategy}_accumulator")
            for _, day_rows in sorted(by_date.items())
        ]
        accumulators = [row for row in accumulators if row]
        report[strategy] = {
            "scope": (
                "Private shadow experiment; never an official publication."
                if is_shadow else
                "Production optimizer output; not necessarily published after Auditor/Decision."
            ),
            "settled_candidate_rows": sum(
                row.get("outcome") in ("win", "loss", "void") for row in rows
            ),
            "pending_selected": sum(row.get("outcome") == "pending" for row in selected_rows),
            "candidate_model_probability": _strategy_probability_summary(rows, "model_prob"),
            "candidate_robust_probability": _strategy_probability_summary(rows, "robust_model_prob"),
            "candidate_market_probability": _strategy_probability_summary(rows, "market_implied_prob"),
            "selected_model_probability": _strategy_probability_summary(selected_rows, "model_prob"),
            "selected_robust_probability": _strategy_probability_summary(selected_rows, "robust_model_prob"),
            "singles": summarize_bet_records(singles),
            "accumulator": summarize_bet_records(accumulators),
        }
    return report


def main() -> None:
    logs = supabase_client.get_prediction_logs()
    selections = supabase_client.get_graded_selections()
    official = summarize_predictions(logs, selections)
    official["strategy_comparison"] = summarize_strategy_predictions(
        supabase_client.get_strategy_prediction_logs()
    )
    official["activation_note"] = (
        "Shadow results never activate a strategy; require a later holdout and explicit review."
    )
    print(json.dumps(official, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
