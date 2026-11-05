from datetime import date, timedelta

import numpy as np
import pytest
from scipy.special import expit

from backend.models.learning import (
    calibrate_probability,
    fit_guarded_calibrator,
    training_examples_from_rows,
)


def synthetic_examples(count=320):
    rng = np.random.default_rng(20261105)
    examples = []
    start = date(2025, 1, 1)
    for index in range(count):
        probability = float(rng.uniform(0.52, 0.90))
        logit = np.log(probability / (1 - probability))
        true_probability = float(expit(-0.45 + 1.45 * logit))
        examples.append({
            "ticket_date": start + timedelta(days=index),
            "probability": probability,
            "outcome": int(rng.random() < true_probability),
        })
    return examples


def test_training_join_prefers_raw_probability_and_settled_outcomes():
    logs = [{
        "ticket_date": "2026-11-01", "fixture_id": 1, "market": "home_win", "selected": True,
        "raw_model_prob": 0.62, "model_prob": 0.67,
    }]
    selections = [{"ticket_date": "2026-11-01", "fixture_id": 1, "market": "home_win", "outcome": "win"}]
    examples = training_examples_from_rows(logs, selections)
    assert examples == [{"ticket_date": date(2026, 11, 1), "probability": 0.62, "outcome": 1}]


def test_calibrator_requires_a_meaningful_sample():
    result = fit_guarded_calibrator(synthetic_examples(40), min_samples=200, min_class_samples=10)
    assert not result["active"]
    assert "Need at least 200" in result["reason"]


def test_calibrator_activates_only_after_holdout_improvement():
    result = fit_guarded_calibrator(synthetic_examples(), min_samples=200, min_class_samples=20)
    assert result["samples"] == 320
    assert result["holdout_raw_log_loss"] > result["holdout_calibrated_log_loss"]
    assert result["active"]
    assert result["slope"] > 1.0
    calibrated = calibrate_probability(0.80, result)
    expected = float(expit(result["intercept"] + result["slope"] * np.log(0.80 / 0.20)))
    assert calibrated == pytest.approx(expected)
    assert calibrated != pytest.approx(0.80)
    assert calibrate_probability(0.60, {"active": False}) == pytest.approx(0.60)
