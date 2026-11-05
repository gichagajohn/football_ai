from datetime import date, timedelta

import numpy as np
import pytest

from backend.models.dixon_coles import fit_dixon_coles


@pytest.fixture(scope="module")
def fitted_model():
    rng = np.random.default_rng(20261104)
    teams = list(range(1, 9))
    attack = {1: 0.42, 2: 0.28, 3: 0.16, 4: 0.06, 5: -0.04, 6: -0.15, 7: -0.28, 8: -0.45}
    start = date(2025, 8, 1)
    matches = []
    sequence = 0
    for _ in range(2):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                home_rate = 1.30 * np.exp(0.18 + attack[home] - attack[away])
                away_rate = 1.05 * np.exp(attack[away] - attack[home])
                matches.append({
                    "id": sequence,
                    "utcDate": (start + timedelta(days=sequence)).isoformat() + "T15:00:00Z",
                    "homeTeam": {"id": home, "name": f"Team {home}"},
                    "awayTeam": {"id": away, "name": f"Team {away}"},
                    "score": {"fullTime": {
                        "home": int(rng.poisson(home_rate)),
                        "away": int(rng.poisson(away_rate)),
                    }},
                })
                sequence += 1
    return fit_dixon_coles(matches, as_of=start + timedelta(days=sequence + 1))


def test_fit_and_score_matrix_are_valid(fitted_model):
    matrix = fitted_model.score_matrix(1, 8)
    assert matrix.shape == (11, 11)
    assert np.all(matrix >= 0)
    assert matrix.sum() == pytest.approx(1.0)
    assert fitted_model.training_matches == 112
    assert fitted_model.matches_per_team[1] == 28


def test_market_probabilities_are_coherent(fitted_model):
    p = fitted_model.probabilities(1, 8)
    assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(1.0)
    assert p["btts_yes"] + p["btts_no"] == pytest.approx(1.0)
    assert p["over25"] + p["under25"] == pytest.approx(1.0)
    assert p["double_chance_home"] == pytest.approx(p["home_win"] + p["draw"])
    assert 0 < p["draw_no_bet_home"] < 1
    assert p["expected_home_goals"] > 0
    assert p["expected_away_goals"] > 0


def test_model_learns_strength_and_home_advantage(fitted_model):
    strong_at_home = fitted_model.expected_goals(1, 8)
    weak_at_home = fitted_model.expected_goals(8, 1)
    assert strong_at_home[0] > weak_at_home[1]
    assert strong_at_home[0] > strong_at_home[1]


def test_unknown_team_is_not_silently_guessed(fitted_model):
    with pytest.raises(KeyError):
        fitted_model.probabilities(999, 1)


def test_insufficient_history_fails_closed():
    with pytest.raises(ValueError, match="Insufficient history"):
        fit_dixon_coles([], as_of=date(2026, 1, 1))
