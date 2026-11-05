"""Guarded self-calibration from settled model outputs.

Team ratings already learn from every newly finished match. This module adds a
second feedback loop for published probabilities. It activates only after a
minimum sample and only when a time-based holdout beats the raw model log loss.
"""

from __future__ import annotations

from datetime import date, datetime
import logging
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from backend.models.evaluation import log_loss

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = "guarded_platt_v1"


def _date_value(value) -> date:
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def training_examples_from_rows(logs: list[dict], selections: list[dict]) -> list[dict]:
    outcomes = {
        (str(row.get("ticket_date")), str(row.get("fixture_id")), row.get("market")): row.get("outcome")
        for row in selections
    }
    examples = []
    for row in logs:
        if not row.get("selected"):
            continue
        key = (str(row.get("ticket_date")), str(row.get("fixture_id")), row.get("market"))
        outcome = outcomes.get(key)
        if outcome not in ("win", "loss"):
            continue
        raw_probability = row.get("raw_model_prob")
        if raw_probability is None:
            raw_probability = row.get("model_prob")
        try:
            probability = float(raw_probability)
            ticket_date = _date_value(row["ticket_date"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 < probability < 1:
            continue
        examples.append({
            "ticket_date": ticket_date,
            "probability": probability,
            "outcome": int(outcome == "win"),
        })
    examples.sort(key=lambda row: row["ticket_date"])
    return examples


def _fit_parameters(examples: list[dict], half_life_days: float) -> tuple[float, float]:
    if not examples:
        raise ValueError("No calibration examples")
    latest = max(row["ticket_date"] for row in examples)
    x = np.array([
        math.log(row["probability"] / (1 - row["probability"])) for row in examples
    ], dtype=float)
    y = np.array([row["outcome"] for row in examples], dtype=float)
    weights = np.array([
        0.5 ** (max(0, (latest - row["ticket_date"]).days) / max(1.0, half_life_days))
        for row in examples
    ], dtype=float)

    def objective(params):
        intercept, slope = params
        calibrated = np.clip(expit(intercept + slope * x), 1e-12, 1 - 1e-12)
        loss = -np.sum(weights * (y * np.log(calibrated) + (1 - y) * np.log(1 - calibrated)))
        # Shrink toward the identity map; prevents a noisy sample from
        # producing an extreme calibration curve.
        penalty = 2.0 * (intercept ** 2 + (slope - 1.0) ** 2)
        return float(loss + penalty)

    result = minimize(
        objective,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0), (0.20, 3.0)],
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Calibration fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def fit_guarded_calibrator(
    examples: list[dict],
    min_samples: int = 200,
    min_class_samples: int = 30,
    half_life_days: float = 240.0,
) -> dict:
    """Return an active calibrator only after a chronological holdout improves."""
    ordered = sorted(examples, key=lambda row: row["ticket_date"])
    count = len(ordered)
    base = {
        "active": False,
        "version": CALIBRATION_VERSION,
        "samples": count,
        "half_life_days": half_life_days,
    }
    wins = sum(row["outcome"] for row in ordered)
    losses = count - wins
    if count < min_samples or min(wins, losses) < min_class_samples:
        base["reason"] = (
            f"Need at least {min_samples} settled outputs and {min_class_samples} per class; "
            f"have {count} ({wins} wins/{losses} losses)."
        )
        return base

    split = int(count * 0.70)
    train, test = ordered[:split], ordered[split:]
    if min(sum(row["outcome"] for row in train), len(train) - sum(row["outcome"] for row in train)) < 10:
        base["reason"] = "Chronological training segment lacks both outcome classes."
        return base
    if len(set(row["outcome"] for row in test)) < 2:
        base["reason"] = "Chronological holdout lacks both outcome classes."
        return base

    try:
        train_intercept, train_slope = _fit_parameters(train, half_life_days)
        raw_probabilities = [row["probability"] for row in test]
        outcomes = [row["outcome"] for row in test]
        calibrated = [
            float(expit(train_intercept + train_slope * math.log(p / (1 - p))))
            for p in raw_probabilities
        ]
        raw_loss = log_loss(raw_probabilities, outcomes)
        calibrated_loss = log_loss(calibrated, outcomes)
        improvement = raw_loss - calibrated_loss
        base.update({
            "holdout_samples": len(test),
            "holdout_raw_log_loss": raw_loss,
            "holdout_calibrated_log_loss": calibrated_loss,
            "holdout_log_loss_improvement": improvement,
        })
        # Require a material improvement, not a lucky seventh decimal.
        if improvement <= 0.001:
            base["reason"] = "Calibration did not materially improve the later holdout; staying off."
            return base
        intercept, slope = _fit_parameters(ordered, half_life_days)
    except (ValueError, TypeError) as exc:
        base["reason"] = str(exc)
        return base

    base.update({
        "active": True,
        "intercept": intercept,
        "slope": slope,
        "trained_through": max(row["ticket_date"] for row in ordered).isoformat(),
        "reason": "Activated after improving the chronological holdout log loss.",
    })
    return base


def calibrate_probability(probability: float, calibration: dict | None) -> float:
    p = min(1 - 1e-9, max(1e-9, float(probability)))
    if not calibration or not calibration.get("active"):
        return p
    logit = math.log(p / (1 - p))
    calibrated = float(expit(calibration["intercept"] + calibration["slope"] * logit))
    return min(1 - 1e-6, max(1e-6, calibrated))


def load_probability_calibrator() -> dict:
    """Load settled outputs without ever writing credentials or outcomes here."""
    min_samples = int(os.environ.get("CALIBRATION_MIN_SAMPLES", "200"))
    half_life = float(os.environ.get("CALIBRATION_HALF_LIFE_DAYS", "240"))
    try:
        from backend.db import supabase_client
        logs = supabase_client.get_prediction_logs()
        selections = supabase_client.get_graded_selections()
        examples = training_examples_from_rows(logs, selections)
        calibration = fit_guarded_calibrator(
            examples, min_samples=min_samples, half_life_days=half_life
        )
        logger.info(f"[LEARNING] {calibration.get('reason')}")
        return calibration
    except Exception as exc:
        logger.warning(f"[LEARNING] Calibration unavailable; using raw probabilities: {exc}")
        return {
            "active": False,
            "version": CALIBRATION_VERSION,
            "samples": 0,
            "reason": f"Calibration load failed safely: {exc}",
        }
