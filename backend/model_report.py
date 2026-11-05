"""Calibration and flat-stake report from private logs + settled selections.

Run after applying the prediction-log migration:
    python -m backend.model_report
"""

from __future__ import annotations

import json
import logging

from backend.db import supabase_client
from backend.models.evaluation import brier_score, flat_stake_roi, log_loss

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


def main() -> None:
    logs = supabase_client.get_prediction_logs()
    selections = supabase_client.get_graded_selections()
    print(json.dumps(summarize_predictions(logs, selections), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
